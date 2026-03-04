# Human-in-the-Loop 패턴

> LangGraph의 `interrupt()`와 `Command`를 활용하여 AI 에이전트 실행 중 사람의 판단을 개입시키는 방법을 배웁니다.

## 개요

이 섹션에서는 LangGraph가 제공하는 Human-in-the-Loop(HITL) 패턴을 학습합니다. AI 에이전트가 자율적으로 작업을 수행하다가 특정 시점에서 멈추고, 사람의 승인이나 입력을 받은 뒤 다시 실행을 이어가는 워크플로우를 구축하는 방법을 다룹니다.

**선수 지식**: [Ch13: LangGraph 기초](13-langgraph-기초/01-langgraph-소개와-핵심-개념.md)에서 배운 `StateGraph`, 노드(Node), 엣지(Edge)의 개념과 기본적인 그래프 구성 방법
**학습 목표**:
- `interrupt()` 함수로 그래프 실행을 일시 중단하고 외부 입력을 받을 수 있다
- `Command(resume=...)` 를 사용하여 중단된 그래프를 재개할 수 있다
- 도구 실행 전 사용자 승인을 요구하는 워크플로우를 구현할 수 있다
- `interrupt_before`/`interrupt_after` 정적 브레이크포인트의 용도를 이해한다

## 왜 알아야 할까?

AI 에이전트가 이메일을 보내거나, 데이터베이스를 수정하거나, 결제를 처리한다고 상상해 보세요. 만약 에이전트가 잘못된 판단을 내렸는데 아무런 확인 없이 실행된다면 어떻게 될까요? 잘못된 사람에게 기밀 이메일이 전송되거나, 중요한 데이터가 삭제될 수 있습니다.

실제 프로덕션 환경에서 AI 에이전트를 운영할 때, **모든 결정을 AI에게 맡기는 것은 위험**합니다. 특히 되돌리기 어려운 작업(이메일 전송, 금융 거래, 파일 삭제 등)에서는 반드시 사람의 확인이 필요하죠. Human-in-the-Loop 패턴은 이런 문제를 해결합니다.

이 패턴을 익히면:
- 에이전트의 **위험한 행동을 사전에 차단**할 수 있습니다
- 사용자로부터 **추가 정보를 수집**하여 더 정확한 결과를 얻을 수 있습니다
- **감사 로그(Audit Log)** 를 남겨 책임 소재를 명확히 할 수 있습니다
- 프로덕션 환경에서 **신뢰할 수 있는 AI 시스템**을 구축할 수 있습니다

## 핵심 개념

### 개념 1: interrupt() — 그래프에 "일시정지 버튼" 달기

> 💡 **비유**: 넷플릭스로 영화를 보다가 일시정지 버튼을 누르는 상황을 떠올려 보세요. 화면이 멈추고, 여러분이 간식을 가져오거나 화장실을 다녀온 뒤 재생 버튼을 누르면 정확히 멈춘 지점부터 다시 시작되죠. LangGraph의 `interrupt()`가 바로 이 일시정지 버튼입니다. 그래프 실행이 멈추고, 사람이 무언가를 결정한 뒤 `Command(resume=...)`로 재생 버튼을 누르면 이어서 실행됩니다.

`interrupt()` 함수는 `langgraph.types`에서 임포트합니다. 노드 함수 안 어디서든 호출할 수 있으며, 호출 시점에 그래프 실행이 즉시 중단됩니다.

```python
from langgraph.types import interrupt, Command

def review_node(state):
    # interrupt()에 전달하는 값은 호출자에게 표시됩니다
    user_decision = interrupt({
        "question": "이 작업을 승인하시겠습니까?",
        "details": state["action_summary"]
    })
    # Command(resume=...)로 재개하면, 그 값이 여기로 반환됩니다
    if user_decision == "approve":
        return {"status": "approved"}
    else:
        return {"status": "rejected"}
```

`interrupt()`에 전달하는 인자는 **JSON 직렬화 가능한 값**이어야 합니다. 문자열, 딕셔너리, 리스트 등 무엇이든 가능하지만, 함수나 클래스 인스턴스처럼 직렬화할 수 없는 객체는 안 됩니다.

**핵심 동작 원리**:
1. `interrupt()` 호출 시, 특별한 예외(Exception)가 발생하여 그래프 런타임에 "일시정지" 신호를 보냅니다
2. 런타임은 현재 상태를 **체크포인터(Checkpointer)** 에 저장합니다
3. 호출자에게 `__interrupt__` 필드를 통해 중단 정보가 전달됩니다
4. `Command(resume=값)`으로 재개하면, 해당 값이 `interrupt()`의 반환값이 됩니다

### 개념 2: Command(resume=...) — "재생 버튼" 누르기

> 💡 **비유**: 식당에서 주문할 때를 생각해 보세요. 웨이터가 "메인 요리는 무엇으로 하시겠어요?"라고 물으면(`interrupt`), 잠시 메뉴를 보고 "스테이크요"라고 대답합니다(`Command(resume="스테이크")`). 그러면 웨이터는 여러분의 답을 가지고 주방으로 가서 나머지 과정을 이어가죠.

중단된 그래프를 재개하려면 `Command(resume=...)`를 사용합니다. 이때 **반드시 같은 `thread_id`** 를 사용해야 합니다.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 체크포인터와 설정
checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "thread-1"}}

# 1단계: 그래프 실행 → interrupt()에서 멈춤
result = graph.invoke({"input": "중요한 이메일 전송"}, config=config)
print(result["__interrupt__"])  # 중단 정보 확인

# 2단계: 사용자 결정 후 재개
graph.invoke(Command(resume="approve"), config=config)
```

**중요한 점**: 그래프가 재개될 때 `interrupt()`가 포함된 **노드 전체가 처음부터 다시 실행**됩니다. `interrupt()` 호출 이전의 코드도 다시 실행되므로, 그 앞에 부수 효과(side effect)가 있다면 **멱등성(idempotency)** 을 보장해야 합니다.

```python
def risky_node(state):
    # ⚠️ 주의: 이 코드는 재개 시 다시 실행됩니다!
    # 데이터베이스 레코드 생성 같은 작업은 중복될 수 있습니다
    record = create_record(state["data"])  # 재개 시 중복 생성!

    approval = interrupt("승인하시겠습니까?")
    return {"result": approval}

def safe_node(state):
    # ✅ 멱등한 방식: 이미 존재하면 건너뜁니다
    record = get_or_create_record(state["data"])

    approval = interrupt("승인하시겠습니까?")
    return {"result": approval}
```

### 개념 3: 체크포인터 — 일시정지 상태를 기억하는 장치

> 💡 **비유**: 게임을 하다가 세이브 포인트에 저장하는 것과 같습니다. 세이브 없이 전원이 꺼지면 처음부터 다시 해야 하죠. 체크포인터는 그래프의 "세이브 파일"입니다.

`interrupt()`를 사용하려면 **반드시 체크포인터가 필요**합니다. 체크포인터가 그래프의 현재 상태를 저장해야 나중에 재개할 수 있기 때문입니다.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# 인메모리 체크포인터 (개발/테스트용)
checkpointer = InMemorySaver()

# 그래프 컴파일 시 체크포인터 연결
builder = StateGraph(State)
# ... 노드와 엣지 추가 ...
graph = builder.compile(checkpointer=checkpointer)
```

| 체크포인터 | 용도 | 특징 |
|-----------|------|------|
| `InMemorySaver` | 개발/테스트 | 프로세스 종료 시 데이터 소멸 |
| `SqliteSaver` | 로컬 프로덕션 | SQLite 파일에 영구 저장 |
| `PostgresSaver` | 서버 프로덕션 | PostgreSQL에 영구 저장, 다중 서버 지원 |

### 개념 4: interrupt_before / interrupt_after — 정적 브레이크포인트

> 💡 **비유**: 교통 신호등처럼, 특정 교차로(노드)에 항상 빨간불이 켜져 있어서 반드시 멈추게 만드는 장치입니다. `interrupt()`가 운전자가 원하는 곳에서 수동으로 차를 세우는 것이라면, `interrupt_before`/`interrupt_after`는 교차로마다 설치된 고정 신호등이죠.

이 방식은 그래프 **컴파일 시점**에 어떤 노드 전/후에 멈출지 지정합니다.

```python
# 컴파일 시 정적 브레이크포인트 설정
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_action"],   # execute_action 노드 실행 전 멈춤
    interrupt_after=["generate_plan"],     # generate_plan 노드 실행 후 멈춤
)
```

`interrupt_before`/`interrupt_after`는 주로 **디버깅이나 간단한 승인 흐름**에 유용합니다. 하지만 노드 내부에서 세밀하게 제어하거나, 사용자에게 구체적인 정보를 보여주고 입력을 받아야 하는 경우에는 `interrupt()` 함수가 훨씬 유연합니다.

| 비교 항목 | `interrupt()` 함수 | `interrupt_before`/`interrupt_after` |
|-----------|-------------------|--------------------------------------|
| 제어 위치 | 노드 함수 내부 어디서든 | 노드 경계(전/후)에서만 |
| 사용자에게 정보 전달 | 임의의 JSON 페이로드 가능 | 현재 상태만 확인 가능 |
| 사용자 입력 수집 | `Command(resume=값)` 으로 직접 전달 | 상태를 수동으로 업데이트 후 재개 |
| 유연성 | 높음 (조건부 중단, 검증 루프 등) | 낮음 (항상 해당 노드에서 멈춤) |
| 권장 용도 | 프로덕션 HITL 워크플로우 | 디버깅, 단순 승인 |

### 개념 5: 도구 실행 전 승인 워크플로우

실제로 가장 많이 쓰이는 HITL 패턴은 **에이전트가 도구를 호출하기 전에 사용자 승인을 받는 것**입니다. 예를 들어, 에이전트가 이메일을 보내려 할 때 "정말로 이 이메일을 보낼까요?"라고 물어보는 거죠.

```python
from langchain_core.tools import tool
from langgraph.types import interrupt

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """이메일을 전송합니다."""
    # 실행 전 사용자에게 승인 요청
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "이 이메일을 보내시겠습니까?"
    })

    if response.get("decision") == "approve":
        # 실제 이메일 전송 로직
        return f"이메일이 {to}에게 전송되었습니다."
    elif response.get("decision") == "edit":
        # 수정된 내용으로 전송
        new_to = response.get("to", to)
        return f"수정된 이메일이 {new_to}에게 전송되었습니다."
    else:
        return "이메일 전송이 취소되었습니다."
```

이 패턴은 세 가지 선택지를 제공합니다:
- **승인(Approve)**: 원래 계획대로 실행
- **수정(Edit)**: 사용자가 내용을 수정한 뒤 실행
- **거부(Reject)**: 실행을 취소하고 피드백 제공

## 실습: 직접 해보기

아래는 완전히 실행 가능한 예제입니다. AI 에이전트가 작업 계획을 세운 뒤, 사용자 승인을 받고, 승인된 경우에만 실행하는 워크플로우를 구축합니다.

```python
"""
Human-in-the-Loop 패턴 실습
- 계획 수립 → 사용자 승인 → 실행 워크플로우
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# --- 1. 상태 정의 ---
class WorkflowState(TypedDict):
    task: str              # 사용자 요청
    plan: str              # AI가 수립한 계획
    approval: str          # 승인 여부
    result: str            # 최종 결과

# --- 2. 노드 함수 정의 ---
def plan_node(state: WorkflowState) -> dict:
    """작업 계획을 수립하는 노드"""
    task = state["task"]
    # 실제로는 LLM을 호출하여 계획을 수립합니다
    plan = f"[계획] '{task}'을(를) 수행하기 위해 다음 단계를 진행합니다:\n"
    plan += "1. 데이터 수집\n2. 분석 실행\n3. 보고서 생성"
    print(f"📋 계획 수립 완료: {plan}")
    return {"plan": plan}

def approval_node(state: WorkflowState) -> dict:
    """사용자 승인을 요청하는 노드 (여기서 interrupt 발생)"""
    # interrupt()로 실행 중단 → 사용자 입력 대기
    user_response = interrupt({
        "question": "다음 계획을 승인하시겠습니까?",
        "plan": state["plan"],
        "options": ["approve", "reject", "edit"]
    })
    print(f"✅ 사용자 응답: {user_response}")
    return {"approval": user_response}

def execute_node(state: WorkflowState) -> dict:
    """승인된 작업을 실행하는 노드"""
    if state["approval"] == "approve":
        result = f"✨ '{state['task']}' 작업이 성공적으로 완료되었습니다!"
    elif state["approval"] == "reject":
        result = "❌ 작업이 사용자에 의해 취소되었습니다."
    else:
        result = f"📝 수정된 계획으로 작업을 실행합니다: {state['approval']}"
    print(result)
    return {"result": result}

# --- 3. 그래프 구성 ---
builder = StateGraph(WorkflowState)

# 노드 추가
builder.add_node("plan", plan_node)
builder.add_node("approval", approval_node)
builder.add_node("execute", execute_node)

# 엣지 연결: 계획 → 승인 → 실행
builder.add_edge(START, "plan")
builder.add_edge("plan", "approval")
builder.add_edge("approval", "execute")
builder.add_edge("execute", END)

# 체크포인터와 함께 컴파일 (interrupt 사용 시 필수!)
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# --- 4. 실행 ---
config = {"configurable": {"thread_id": "demo-thread-1"}}

# 1단계: 그래프 실행 → approval_node의 interrupt()에서 멈춤
print("=" * 50)
print("🚀 1단계: 그래프 실행 시작")
print("=" * 50)
result = graph.invoke(
    {"task": "월간 매출 보고서 생성"},
    config=config
)

# 중단 정보 확인
print("\n⏸️  그래프가 중단되었습니다!")
print(f"중단 정보: {result['__interrupt__']}")

# 2단계: 사용자가 승인 후 재개
print("\n" + "=" * 50)
print("▶️  2단계: 사용자 승인 후 재개")
print("=" * 50)
final_result = graph.invoke(
    Command(resume="approve"),  # 사용자가 "approve" 입력
    config=config               # 같은 thread_id 사용!
)

print(f"\n🎯 최종 결과: {final_result['result']}")
```

실행 결과:

```
==================================================
🚀 1단계: 그래프 실행 시작
==================================================
📋 계획 수립 완료: [계획] '월간 매출 보고서 생성'을(를) 수행하기 위해 다음 단계를 진행합니다:
1. 데이터 수집
2. 분석 실행
3. 보고서 생성

⏸️  그래프가 중단되었습니다!
중단 정보: [Interrupt(value={'question': '다음 계획을 승인하시겠습니까?', ...})]

==================================================
▶️  2단계: 사용자 승인 후 재개
==================================================
✅ 사용자 응답: approve
✨ '월간 매출 보고서 생성' 작업이 성공적으로 완료되었습니다!

🎯 최종 결과: ✨ '월간 매출 보고서 생성' 작업이 성공적으로 완료되었습니다!
```

이번에는 **입력 검증 루프** 패턴도 만들어 봅시다. 사용자가 올바른 값을 입력할 때까지 반복해서 물어보는 패턴입니다.

```python
"""
입력 검증 루프: 사용자가 유효한 값을 입력할 때까지 반복
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class FormState(TypedDict):
    name: str
    age: int
    confirmed: bool

def collect_age_node(state: FormState) -> dict:
    """나이를 입력받되, 유효한 값이 올 때까지 반복합니다."""
    prompt = "나이를 입력해 주세요 (1~150 사이 숫자):"

    while True:
        # 매번 interrupt()로 사용자 입력을 받습니다
        answer = interrupt({"prompt": prompt})

        # 검증: 숫자이고 유효한 범위인지 확인
        try:
            age = int(answer)
            if 1 <= age <= 150:
                return {"age": age}
            prompt = f"'{answer}'은(는) 유효하지 않습니다. 1~150 사이 숫자를 입력해 주세요:"
        except (ValueError, TypeError):
            prompt = f"'{answer}'은(는) 숫자가 아닙니다. 숫자를 입력해 주세요:"

# 그래프 구성
builder = StateGraph(FormState)
builder.add_node("collect_age", collect_age_node)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 실행
config = {"configurable": {"thread_id": "form-thread-1"}}

# 첫 번째 시도: 잘못된 입력
result = graph.invoke({"name": "홍길동", "age": 0, "confirmed": False}, config=config)
print("프롬프트:", result["__interrupt__"][0].value)

# 잘못된 값으로 재개 → 다시 interrupt 발생
result = graph.invoke(Command(resume="abc"), config=config)
print("프롬프트:", result["__interrupt__"][0].value)

# 올바른 값으로 재개 → 정상 완료
result = graph.invoke(Command(resume="25"), config=config)
print("최종 결과 - 나이:", result["age"])
# 출력: 최종 결과 - 나이: 25
```

## 더 깊이 알아보기

### Human-in-the-Loop의 탄생 배경

Human-in-the-Loop라는 개념은 AI가 등장하기 훨씬 전, **1960년대 항공우주 산업**에서 시작되었습니다. 미국 NASA의 우주 프로그램에서 자동 조종 시스템이 발전하면서, 기계가 모든 것을 제어하되 **핵심 결정은 반드시 조종사(사람)가 내리는** 설계 철학이 확립되었습니다. 아폴로 11호의 달 착륙 때도 자동 시스템이 착륙 지점을 잡았지만, 닐 암스트롱이 바위투성이 지형을 눈으로 확인하고 수동으로 착륙 지점을 변경한 유명한 일화가 있죠.

### LangGraph의 interrupt() 탄생

LangGraph를 만든 Harrison Chase와 LangChain 팀은 2024년 초 LangGraph를 출시하면서 처음에는 `interrupt_before`/`interrupt_after`라는 정적 브레이크포인트만 제공했습니다. 하지만 실제 프로덕션 환경에서 사용자들은 **노드 내부에서 더 세밀한 제어**를 원했습니다. "이 도구 호출 전에만 멈추고 싶은데, 노드 전체를 멈추면 다른 안전한 도구까지 멈춰버려요"라는 피드백이 쏟아졌거든요.

이에 LangChain 팀은 Python의 `input()` 함수에서 영감을 받아 `interrupt()` 함수를 설계했습니다. `input()`이 사용자 입력을 받을 때까지 프로그램을 멈추듯, `interrupt()`도 외부 입력을 받을 때까지 그래프 실행을 멈추는 것이죠. 이 기능은 LangGraph의 **가장 중요한 설계 원칙**인 "제어 가능성(Controllability)"을 실현하는 핵심 메커니즘이 되었습니다.

> 💡 **알고 계셨나요?**: Harrison Chase는 "에이전트의 진정한 가치는 자율성이 아니라 제어 가능성에 있다"고 강조했습니다. 아무리 똑똑한 에이전트라도 사람이 개입할 수 없다면 프로덕션에서 사용할 수 없기 때문이죠. LangGraph는 이 철학을 기반으로 스트리밍, 지속적 실행(durable execution), 그리고 Human-in-the-Loop를 핵심 기능으로 설계되었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "interrupt()를 try/except로 감싸서 예외를 처리해야 한다"
> `interrupt()`는 내부적으로 특별한 예외를 발생시켜 런타임에 신호를 보냅니다. 이것을 `try/except`로 잡아버리면 중단 신호가 전파되지 않아 interrupt가 작동하지 않습니다. **절대로 `interrupt()` 호출을 try/except 블록 안에 넣지 마세요.**

> ⚠️ **흔한 오해**: "interrupt()를 호출한 정확한 줄에서 재개된다"
> 그래프가 재개될 때는 `interrupt()`가 포함된 **노드 전체가 처음부터 다시 실행**됩니다. `interrupt()` 위에 있는 코드도 다시 실행되므로, 데이터베이스 INSERT 같은 비멱등 작업이 있다면 중복 실행 문제가 생깁니다. 반드시 `interrupt()` 이전의 부수 효과를 멱등하게 만드세요.

> 🔥 **실무 팁**: 프로덕션에서는 `InMemorySaver` 대신 `PostgresSaver`나 `SqliteSaver`를 사용하세요. `InMemorySaver`는 프로세스가 종료되면 모든 체크포인트가 사라지기 때문에, 서버 재시작 시 중단된 워크플로우를 재개할 수 없습니다. 사용자가 승인 요청을 받고 10분 뒤에 응답할 수도 있으니, 상태는 반드시 영구 저장소에 보관해야 합니다.

> 🔥 **실무 팁**: `interrupt()`에 전달하는 페이로드에는 사용자가 판단에 필요한 **모든 맥락 정보**를 포함하세요. "승인하시겠습니까?"만 보내면 사용자는 무엇을 승인하는지 알 수 없습니다. 작업 내용, 영향 범위, 위험도 등을 함께 전달하면 더 빠르고 정확한 의사결정이 가능합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `interrupt(payload)` | 그래프 실행을 중단하고, payload를 호출자에게 전달합니다 |
| `Command(resume=value)` | 중단된 그래프를 재개하며, value가 `interrupt()`의 반환값이 됩니다 |
| 체크포인터(Checkpointer) | `interrupt()` 사용 시 필수. 중단 시점의 상태를 저장합니다 |
| `thread_id` | 재개 시 어떤 실행 상태를 이어갈지 식별하는 키입니다 |
| `interrupt_before` | 지정한 노드 **실행 전**에 그래프를 멈추는 정적 브레이크포인트 |
| `interrupt_after` | 지정한 노드 **실행 후**에 그래프를 멈추는 정적 브레이크포인트 |
| 멱등성(Idempotency) | `interrupt()` 이전 코드가 재실행되므로, 부수 효과의 멱등성을 보장해야 합니다 |
| 승인 워크플로우 | approve/edit/reject 세 가지 응답으로 도구 실행을 제어하는 패턴 |

## 다음 섹션 미리보기

이번 섹션에서 `interrupt()`로 그래프를 "일시정지"하는 방법을 배웠다면, 다음 섹션 **체크포인팅과 상태 영속성**에서는 이 일시정지 상태가 **어떻게 저장되고 복원되는지** 깊이 파고듭니다. `InMemorySaver`를 넘어 `SqliteSaver`, `PostgresSaver`를 활용한 프로덕션 수준의 상태 관리와, 스레드(thread) 개념을 통해 여러 대화를 동시에 관리하는 방법을 다룹니다.

## 참고 자료

- [LangGraph Interrupts 공식 문서](https://docs.langchain.com/oss/python/langgraph/interrupts) - `interrupt()`, `Command`, `interrupt_before`/`interrupt_after`의 완전한 API 레퍼런스와 사용 예제
- [interrupt: Simplifying human-in-the-loop (LangChain 공식 블로그)](https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt/) - `interrupt()` 함수의 설계 배경과 기존 방식 대비 개선점을 설명하는 공식 발표
- [LangGraph v0.4: Working with Interrupts (LangChain Changelog)](https://changelog.langchain.com/announcements/langgraph-v0-4-working-with-interrupts) - v0.4에서 추가된 다중 인터럽트 동시 재개 등 최신 기능 변경 사항
- [Interrupts and Commands in LangGraph (DEV Community)](https://dev.to/jamesbmour/interrupts-and-commands-in-langgraph-building-human-in-the-loop-workflows-4ngl) - 단계별 코드 예제와 함께 HITL 워크플로우를 구축하는 실전 튜토리얼
- [LangGraph Human-in-the-Loop 공식 가이드](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) - LangChain 생태계에서의 HITL 패턴 개요와 에이전트 도구 호출 승인 미들웨어 설명

---
### 🔗 Related Sessions
- [node](01-langchain-소개와-개발-환경-설정/05-langchain-생태계-탐색.md) (prerequisite)
- [edge](01-langchain-소개와-개발-환경-설정/05-langchain-생태계-탐색.md) (prerequisite)
- [checkpointer](./02-체크포인팅과-상태-영속성.md) (prerequisite)
