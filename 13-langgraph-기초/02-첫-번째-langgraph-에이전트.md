# 첫 번째 LangGraph 에이전트

> TypedDict로 상태를 설계하고, 노드와 엣지로 연결하여 첫 번째 LangGraph 에이전트를 직접 만들어봅니다.

## 개요

이 섹션에서는 [Session 13.1: LangGraph 소개와 핵심 개념](ch13/session_13_1.md)에서 배운 StateGraph, 노드, 엣지의 개념을 실제 코드로 구현합니다. 이론에서 벗어나 직접 손으로 그래프를 만들고, 컴파일하고, 실행하는 전 과정을 경험하게 됩니다.

**선수 지식**: Session 13.1에서 배운 StateGraph, 노드, 엣지, START/END의 개념
**학습 목표**:
- TypedDict를 사용하여 그래프 상태 스키마를 정의할 수 있다
- 노드 함수를 작성하고 `add_node()`로 그래프에 등록할 수 있다
- `add_edge()`로 노드 간 실행 흐름을 연결할 수 있다
- 그래프를 컴파일(`compile`)하고 실행(`invoke`)할 수 있다

## 왜 알아야 할까?

LangGraph의 개념을 아는 것과 실제로 작동하는 에이전트를 만드는 것은 완전히 다른 이야기입니다. 마치 자동차의 엔진, 변속기, 바퀴가 어떤 역할을 하는지 아는 것과 실제로 자동차를 조립해서 시동을 거는 것의 차이와 같죠.

이 세션에서 만드는 "첫 번째 에이전트"는 앞으로 여러분이 구축할 모든 LangGraph 애플리케이션의 뼈대가 됩니다. RAG 파이프라인, 멀티 에이전트 시스템, 복잡한 워크플로우 — 이 모든 것이 **상태 정의 → 노드 작성 → 엣지 연결 → 컴파일 → 실행**이라는 동일한 패턴 위에 세워집니다. 이 패턴을 한 번 확실히 익혀두면, 나머지는 응용일 뿐입니다.

## 핵심 개념

### 개념 1: 상태 스키마 — 에이전트의 "기억 장치" 설계하기

> 💡 **비유**: 상태 스키마는 **여행 체크리스트**와 같습니다. 여행을 떠나기 전에 "여권 ✓, 항공권 ✓, 호텔 예약 ✓" 같은 체크리스트를 만들죠? 여행의 각 단계(공항 도착, 출국 심사, 탑승)를 거칠 때마다 체크리스트의 항목이 업데이트됩니다. LangGraph의 상태 스키마도 마찬가지입니다 — 그래프가 실행되는 동안 각 노드가 읽고 업데이트할 "공유 체크리스트"를 미리 정의하는 거예요.

LangGraph에서 상태(State)는 Python의 `TypedDict`를 사용하여 정의합니다. `TypedDict`는 딕셔너리의 키와 값의 타입을 명시적으로 선언하는 방법인데요, 이를 통해 그래프의 모든 노드가 어떤 데이터를 주고받을지 계약(contract)을 맺는 셈입니다.

```python
from typing import TypedDict

# 가장 기본적인 상태 스키마
class AgentState(TypedDict):
    query: str          # 사용자의 질문
    response: str       # 에이전트의 응답
    steps: list[str]    # 처리 단계 기록
```

상태 스키마를 정의할 때 기억해야 할 핵심 원칙이 있습니다:

1. **최소한으로 유지**: 정말 필요한 필드만 넣으세요. "혹시 모르니까" 넣는 필드는 복잡성만 증가시킵니다.
2. **명확한 타입 선언**: 각 필드의 타입을 정확히 명시하면 디버깅이 훨씬 쉬워집니다.
3. **직렬화 가능한 타입 사용**: 체크포인트 저장을 위해 JSON으로 변환 가능한 타입(str, int, list, dict 등)을 사용하세요.

#### Annotated와 리듀서(Reducer) — 값을 "누적"하는 마법

기본적으로 노드가 상태의 특정 키에 값을 반환하면, 기존 값을 **덮어씁니다**. 하지만 때로는 값을 덮어쓰는 게 아니라 **누적**하고 싶을 때가 있죠. 예를 들어 메시지 히스토리나 처리 단계 로그처럼요.

이때 `Annotated` 타입과 리듀서 함수를 사용합니다:

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    query: str                                    # 덮어쓰기 (기본 동작)
    response: str                                 # 덮어쓰기
    steps: Annotated[list[str], operator.add]     # 리스트 누적!
```

`Annotated[list[str], operator.add]`는 "이 필드에 새 값이 오면, 기존 리스트에 **이어 붙여라**"라는 뜻입니다. 노드 A가 `{"steps": ["분석"]}`, 노드 B가 `{"steps": ["생성"]}`을 반환하면, 최종 상태의 `steps`는 `["분석", "생성"]`이 됩니다.

### 개념 2: 노드 함수 — 에이전트의 "일꾼" 만들기

> 💡 **비유**: 노드 함수는 **공장의 작업 스테이션**과 같습니다. 자동차 공장에서 한 스테이션은 차체를 조립하고, 다음 스테이션은 도색을 하고, 그 다음은 내장을 설치하죠. 각 스테이션은 컨베이어 벨트(상태)에서 작업물을 받아 자기 일을 한 뒤 다시 벨트에 올려놓습니다.

노드 함수는 **현재 상태를 입력으로 받아, 업데이트할 부분만 딕셔너리로 반환**하는 단순한 Python 함수입니다:

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    query: str
    response: str
    steps: Annotated[list[str], operator.add]

# 노드 함수: 상태를 받아서 업데이트할 부분만 반환
def analyze_query(state: AgentState) -> dict:
    """사용자 질문을 분석하는 노드"""
    query = state["query"]  # 현재 상태에서 읽기
    return {
        "steps": [f"'{query}' 질문 분석 완료"]  # steps에 누적
    }

def generate_response(state: AgentState) -> dict:
    """응답을 생성하는 노드"""
    query = state["query"]
    return {
        "response": f"'{query}'에 대한 답변입니다.",  # response 덮어쓰기
        "steps": ["응답 생성 완료"]                    # steps에 누적
    }
```

노드 함수 작성 시 핵심 규칙:

- **입력**: 전체 상태 딕셔너리(`AgentState`)를 받습니다
- **출력**: 업데이트할 키-값 쌍만 포함된 딕셔너리를 반환합니다
- **누락된 키는 유지**: 반환하지 않은 키는 이전 값이 그대로 유지됩니다
- **순수 함수 권장**: 같은 입력에 같은 출력을 반환하면 디버깅이 쉬워집니다

### 개념 3: 그래프 조립 — add_node와 add_edge

> 💡 **비유**: 이제 레고 블록(노드)을 만들었으니, 조립 설명서(엣지)를 따라 붙여야 합니다. `add_node()`는 "이 블록을 사용할 거야"라고 선언하는 것이고, `add_edge()`는 "이 블록 다음에 저 블록을 연결해"라고 지시하는 것입니다.

```python
from langgraph.graph import StateGraph, START, END

# 1. StateGraph 생성 — 상태 스키마 전달
graph = StateGraph(AgentState)

# 2. 노드 등록 — add_node(이름, 함수)
graph.add_node("analyze", analyze_query)
graph.add_node("generate", generate_response)

# 3. 엣지 연결 — add_edge(시작 노드, 도착 노드)
graph.add_edge(START, "analyze")       # 시작 → 분석
graph.add_edge("analyze", "generate")  # 분석 → 생성
graph.add_edge("generate", END)        # 생성 → 종료
```

실행 흐름을 시각적으로 표현하면:

```
START → [analyze] → [generate] → END
```

**`START`와 `END`의 역할:**
- `START`: 그래프 실행의 진입점입니다. `add_edge(START, "첫_노드")`로 어디서 시작할지 지정합니다.
- `END`: 그래프 실행의 종료점입니다. `add_edge("마지막_노드", END)`로 어디서 끝날지 지정합니다.

`add_node()`의 첫 번째 인자(문자열)는 노드의 **이름**입니다. 이 이름은 `add_edge()`에서 노드를 참조할 때 사용되며, 디버깅과 시각화에서도 표시됩니다. 함수 이름과 노드 이름이 반드시 같을 필요는 없지만, 일관성을 위해 맞추는 것을 권장합니다.

### 개념 4: 컴파일과 실행 — 시동 걸기

> 💡 **비유**: 레고를 다 조립했다고 바로 움직이진 않죠. `compile()`은 조립된 레고에 배터리를 넣고 전원을 켜는 과정입니다. 이 단계에서 LangGraph는 그래프 구조를 검증하고, 실행 가능한 형태로 변환합니다.

```python
# 4. 컴파일 — 실행 가능한 형태로 변환
app = graph.compile()

# 5. 실행 — 초기 상태 전달
result = app.invoke({
    "query": "LangGraph란 무엇인가요?",
    "response": "",
    "steps": []
})

# 결과 확인
print(result)
# {'query': 'LangGraph란 무엇인가요?',
#  'response': "'LangGraph란 무엇인가요?'에 대한 답변입니다.",
#  'steps': ["'LangGraph란 무엇인가요?' 질문 분석 완료", '응답 생성 완료']}
```

`compile()`이 하는 일:
1. **구조 검증**: 모든 노드가 엣지로 연결되어 있는지, START에서 END까지 경로가 존재하는지 확인합니다
2. **`CompiledStateGraph` 반환**: LangChain의 `Runnable` 인터페이스를 구현한 객체를 반환합니다
3. **실행 메서드 제공**: `.invoke()`, `.stream()`, `.batch()`, `.ainvoke()` 등을 사용할 수 있게 됩니다

`invoke()`에 전달하는 딕셔너리는 **초기 상태**입니다. 그래프는 이 상태에서 출발하여 각 노드를 순서대로 실행하면서 상태를 업데이트하고, 최종 상태를 반환합니다.

### 개념 5: 스트리밍 — 실행 과정 들여다보기

`invoke()`는 최종 결과만 반환하지만, `stream()`을 사용하면 각 노드의 실행 결과를 하나씩 받아볼 수 있습니다. 이는 디버깅이나 사용자에게 진행 상황을 보여줄 때 매우 유용합니다.

```python
# 스트리밍 실행 — 노드별 결과를 순차적으로 받기
for event in app.stream({
    "query": "LangGraph란 무엇인가요?",
    "response": "",
    "steps": []
}):
    print(event)
    print("---")

# 출력:
# {'analyze': {'steps': ["'LangGraph란 무엇인가요?' 질문 분석 완료"]}}
# ---
# {'generate': {'response': "'LangGraph란 무엇인가요?'에 대한 답변입니다.",
#               'steps': ['응답 생성 완료']}}
# ---
```

`stream()`의 각 이벤트는 `{노드_이름: 해당_노드의_출력}` 형태입니다. 어떤 노드가 어떤 상태 변경을 했는지 명확하게 추적할 수 있죠.

### 개념 6: 그래프 시각화 — 구조를 눈으로 확인하기

LangGraph는 그래프 구조를 시각적으로 확인할 수 있는 도구를 내장하고 있습니다:

```python
# Mermaid 다이어그램 코드 생성
print(app.get_graph().draw_mermaid())

# 출력 예시:
# %%{init: {'flowchart': {'curve': 'linear'}}}%%
# graph TD;
#     __start__([<p>__start__</p>]):::first
#     analyze(analyze)
#     generate(generate)
#     __end__([<p>__end__</p>]):::last
#     __start__ --> analyze;
#     analyze --> generate;
#     generate --> __end__;
```

생성된 Mermaid 코드를 [Mermaid Live Editor](https://mermaid.live)에 붙여넣으면 그래프를 시각적으로 확인할 수 있습니다. Jupyter 노트북 환경에서는 `draw_mermaid_png()`를 사용하면 노트북 안에서 바로 그래프를 렌더링할 수도 있습니다.

```python
# Jupyter 노트북에서 그래프 이미지 표시
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

## 실습: 직접 해보기

이제 배운 내용을 종합하여, LLM을 연동한 간단한 질의응답 에이전트를 만들어보겠습니다. 이 에이전트는 세 단계로 작동합니다: 질문 분류 → LLM 호출 → 응답 후처리.

```python
"""
LangGraph 첫 번째 에이전트 — 3단계 질의응답 파이프라인
실행 전: pip install langgraph langchain-openai python-dotenv
"""

import os
from typing import TypedDict, Annotated, Literal
import operator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 환경 변수 로드
load_dotenv()

# ── 1단계: 상태 스키마 정의 ──────────────────────────
class QAState(TypedDict):
    question: str                                      # 사용자 질문
    category: str                                      # 질문 카테고리
    answer: str                                        # LLM 응답
    steps: Annotated[list[str], operator.add]           # 처리 로그 (누적)

# ── 2단계: 노드 함수 작성 ────────────────────────────

def classify_question(state: QAState) -> dict:
    """질문을 카테고리별로 분류하는 노드"""
    question = state["question"]

    # 간단한 키워드 기반 분류 (실제로는 LLM을 사용할 수도 있음)
    if any(kw in question for kw in ["코드", "프로그래밍", "함수", "버그"]):
        category = "technical"
    elif any(kw in question for kw in ["추천", "비교", "어떤 것"]):
        category = "recommendation"
    else:
        category = "general"

    return {
        "category": category,
        "steps": [f"질문 분류: {category}"]
    }

def call_llm(state: QAState) -> dict:
    """카테고리에 맞는 시스템 프롬프트로 LLM을 호출하는 노드"""
    # 카테고리별 시스템 프롬프트 설정
    system_prompts = {
        "technical": "당신은 시니어 개발자입니다. 코드 예제와 함께 설명해주세요.",
        "recommendation": "당신은 기술 컨설턴트입니다. 장단점을 비교하여 추천해주세요.",
        "general": "당신은 친절한 AI 어시스턴트입니다. 쉽게 설명해주세요."
    }

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    messages = [
        SystemMessage(content=system_prompts[state["category"]]),
        HumanMessage(content=state["question"])
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "steps": [f"LLM 호출 완료 (모델: gpt-4o, 카테고리: {state['category']})"]
    }

def format_response(state: QAState) -> dict:
    """응답을 보기 좋게 포맷팅하는 노드"""
    formatted = (
        f"📂 카테고리: {state['category']}\n"
        f"❓ 질문: {state['question']}\n"
        f"{'─' * 40}\n"
        f"💬 답변:\n{state['answer']}"
    )

    return {
        "answer": formatted,
        "steps": ["응답 포맷팅 완료"]
    }

# ── 3단계: 그래프 조립 ──────────────────────────────

# StateGraph 생성
graph = StateGraph(QAState)

# 노드 등록
graph.add_node("classify", classify_question)
graph.add_node("llm", call_llm)
graph.add_node("format", format_response)

# 엣지 연결: START → classify → llm → format → END
graph.add_edge(START, "classify")
graph.add_edge("classify", "llm")
graph.add_edge("llm", "format")
graph.add_edge("format", END)

# ── 4단계: 컴파일 ───────────────────────────────────

app = graph.compile()

# ── 5단계: 실행 ─────────────────────────────────────

if __name__ == "__main__":
    # 그래프 구조 시각화
    print("=== 그래프 구조 (Mermaid) ===")
    print(app.get_graph().draw_mermaid())
    print()

    # invoke로 실행
    print("=== invoke 실행 ===")
    result = app.invoke({
        "question": "Python에서 리스트와 튜플의 차이는 뭔가요?",
        "category": "",
        "answer": "",
        "steps": []
    })

    print(result["answer"])
    print()
    print("=== 처리 단계 ===")
    for i, step in enumerate(result["steps"], 1):
        print(f"  {i}. {step}")

    # stream으로 실행 — 각 노드의 진행 상황 확인
    print("\n=== stream 실행 ===")
    for event in app.stream({
        "question": "FastAPI와 Django 중 어떤 것을 추천하시나요?",
        "category": "",
        "answer": "",
        "steps": []
    }):
        # 이벤트에서 노드 이름과 결과 추출
        for node_name, output in event.items():
            print(f"[{node_name}] 완료")
            if "steps" in output:
                for step in output["steps"]:
                    print(f"  → {step}")
        print()
```

**실행 결과 (예시):**
```
=== 그래프 구조 (Mermaid) ===
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__([<p>__start__</p>]):::first
    classify(classify)
    llm(llm)
    format(format)
    __end__([<p>__end__</p>]):::last
    __start__ --> classify;
    classify --> llm;
    llm --> format;
    format --> __end__;

=== invoke 실행 ===
📂 카테고리: technical
❓ 질문: Python에서 리스트와 튜플의 차이는 뭔가요?
────────────────────────────────────────
💬 답변:
리스트(list)와 튜플(tuple)의 가장 큰 차이는 ...

=== 처리 단계 ===
  1. 질문 분류: technical
  2. LLM 호출 완료 (모델: gpt-4o, 카테고리: technical)
  3. 응답 포맷팅 완료

=== stream 실행 ===
[classify] 완료
  → 질문 분류: recommendation

[llm] 완료
  → LLM 호출 완료 (모델: gpt-4o, 카테고리: recommendation)

[format] 완료
  → 응답 포맷팅 완료
```

## 더 깊이 알아보기

### LangGraph의 탄생 — "체인은 충분하지 않았다"

LangGraph는 2023년 여름부터 개발이 시작되어 2024년 초에 공개되었습니다. LangChain의 창시자인 Harrison Chase가 Harvard에서 통계학과 컴퓨터 과학을 전공한 뒤, Kensho(핀테크 스타트업)에서 엔티티 링킹 팀을 이끌고, Robust Intelligence에서 ML 팀을 리드하던 경험이 그 배경이 되었죠.

LangChain의 초기 에이전트(`AgentExecutor`)는 기본적으로 선형적인 체인 구조였습니다. "생각 → 행동 → 관찰"의 루프를 반복할 수는 있었지만, **분기, 병렬 실행, 상태 관리** 같은 복잡한 워크플로우를 구현하기에는 한계가 명확했거든요. 실제로 프로덕션에서 에이전트를 운영하려는 기업들이 가장 많이 요청한 기능이 바로 "세밀한 제어"와 "상태 유지"였습니다.

Harrison Chase와 LangChain 팀은 이 문제를 해결하기 위해 **유한 상태 머신(Finite State Machine)**과 **방향성 그래프(Directed Graph)** 개념을 결합했습니다. 그 결과가 LangGraph입니다. "Graph"라는 이름이 붙은 이유도 여기에 있죠 — 노드(처리 단계)와 엣지(전환)로 구성된 그래프 구조로 에이전트 워크플로우를 표현합니다.

2025년 10월에는 LangGraph 1.0이 정식 출시되면서 첫 안정 버전이 되었고, Uber, LinkedIn, Klarna 등 대기업에서도 프로덕션 환경에서 LangGraph를 활용하고 있습니다. 노드 캐싱, 지연 노드(deferred nodes), pre/post 모델 훅 같은 고급 기능도 추가되면서 프로덕션 레벨의 에이전트 프레임워크로 자리잡았습니다.

### TypedDict를 선택한 이유

LangGraph가 상태 스키마에 `TypedDict`를 기본으로 채택한 것도 흥미로운 결정입니다. Pydantic `BaseModel`이나 `dataclass`도 사용할 수 있지만, `TypedDict`가 기본인 이유는 **런타임 오버헤드가 거의 없기 때문**입니다. `TypedDict`는 순수한 타입 힌트로, 실행 시에는 일반 딕셔너리와 동일하게 동작합니다. 에이전트가 수백, 수천 번 상태를 업데이트하는 프로덕션 환경에서 이 차이는 무시할 수 없죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "노드 함수는 상태 전체를 반환해야 한다" — 아닙니다! 노드 함수는 **변경된 키만 반환**하면 됩니다. 반환하지 않은 키는 이전 값이 그대로 유지됩니다. 예를 들어 `query`를 수정하지 않는 노드는 `query`를 반환에 포함시킬 필요가 없습니다.

> 💡 **알고 계셨나요?**: `compile()`이 반환하는 객체는 LangChain의 `Runnable` 인터페이스를 완벽히 구현합니다. 즉, 기존 LCEL 체인과 동일하게 `.invoke()`, `.stream()`, `.batch()`, `.ainvoke()`, `.astream()` 등을 사용할 수 있습니다. LangGraph 그래프를 LCEL 파이프라인의 한 컴포넌트로 조합하는 것도 가능합니다!

> 🔥 **실무 팁**: 리듀서를 사용하지 않는 필드에 대해 **두 개 이상의 노드가 동시에 같은 키를 업데이트하면 에러가 발생**합니다. 병렬 노드가 같은 필드를 수정해야 할 때는 반드시 `Annotated` 리듀서를 사용하세요. 또한 `invoke()` 시 초기 상태에서 `Annotated` 필드의 초기값을 빈 리스트(`[]`)로 전달하는 것을 잊지 마세요.

> 🔥 **실무 팁**: 디버깅할 때 `stream()`을 적극 활용하세요. `invoke()`는 최종 결과만 보여주지만, `stream()`은 각 노드가 어떤 상태 변경을 했는지 단계별로 확인할 수 있어 문제를 빠르게 찾을 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `TypedDict` 상태 스키마 | 그래프의 공유 상태 구조를 타입 안전하게 정의하는 방법 |
| `Annotated` + 리듀서 | 상태 필드의 값을 덮어쓰기 대신 누적(merge)하는 메커니즘 |
| 노드 함수 | 상태를 입력으로 받아 업데이트할 부분만 딕셔너리로 반환하는 함수 |
| `add_node(이름, 함수)` | 그래프에 노드를 등록하는 메서드 |
| `add_edge(출발, 도착)` | 두 노드 사이의 실행 순서를 정의하는 메서드 |
| `START` / `END` | 그래프의 진입점과 종료점을 나타내는 특수 노드 |
| `compile()` | 그래프를 검증하고 실행 가능한 `Runnable` 객체로 변환 |
| `invoke(초기상태)` | 초기 상태를 전달하여 그래프를 실행하고 최종 상태를 반환 |
| `stream(초기상태)` | 각 노드의 실행 결과를 순차적으로 스트리밍 |

## 다음 섹션 미리보기

이번 세션에서는 `add_edge()`로 노드를 순차적으로 연결하는 **일직선 그래프**를 만들었습니다. 하지만 실제 에이전트는 상황에 따라 다른 경로를 선택해야 하죠 — "질문이 기술적이면 코드 생성 노드로, 일반적이면 대화 노드로" 같은 분기 로직이 필요합니다. 다음 세션에서는 `add_conditional_edges()`를 사용하여 **조건부 분기**를 구현하고, 라우터 함수로 동적 워크플로우를 만드는 방법을 배웁니다.

## 참고 자료

- [LangGraph Graph API Overview — LangChain 공식 문서](https://docs.langchain.com/oss/python/langgraph/graph-api) - StateGraph, 노드, 엣지의 공식 API 레퍼런스. 가장 먼저 참고해야 할 자료입니다.
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph) - 소스 코드와 예제, 이슈 트래커를 확인할 수 있는 공식 저장소
- [LangGraph: Build Stateful AI Agents in Python — Real Python](https://realpython.com/langgraph-python/) - Python 개발자를 위한 LangGraph 실습 튜토리얼. 상태 관리와 그래프 구성을 단계별로 설명합니다.
- [How to Build LangGraph Agents — DataCamp Tutorial](https://www.datacamp.com/tutorial/langgraph-agents) - 에이전트 구축에 초점을 맞춘 실전 튜토리얼
- [LangGraph 1.0 Release — LangChain Blog](https://blog.langchain.com/langchain-langgraph-1dot0/) - LangGraph 1.0 릴리즈 노트와 주요 변경사항

---
### 🔗 Related Sessions
- [langgraph](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [stategraph](../01-langchain-소개와-개발-환경-설정/05-langchain-생태계-탐색.md) (prerequisite)
- [node](../01-langchain-소개와-개발-환경-설정/05-langchain-생태계-탐색.md) (prerequisite)
- [edge](../01-langchain-소개와-개발-환경-설정/05-langchain-생태계-탐색.md) (prerequisite)
