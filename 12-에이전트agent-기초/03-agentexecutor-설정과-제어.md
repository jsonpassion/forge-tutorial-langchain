# AgentExecutor 설정과 제어

> AgentExecutor의 핵심 파라미터를 마스터하여 에이전트의 실행 시간, 반복 횟수, 오류 처리를 정밀하게 제어하는 방법을 배웁니다.

## 개요

이 섹션에서는 AgentExecutor의 **실행 제어 파라미터**를 심층적으로 다룹니다. [세션 12.2: create_react_agent로 에이전트 구축](ch12/session_12_2.md)에서 AgentExecutor의 기본 사용법을 배웠다면, 이번에는 에이전트가 "폭주"하지 않도록 안전장치를 설정하고, 오류 상황을 우아하게 처리하며, 에이전트의 추론 과정을 투명하게 들여다보는 방법을 익힙니다.

**선수 지식**: 세션 12.1의 ReAct 패턴과 Thought→Action→Observation 루프, 세션 12.2의 `create_react_agent` 및 `AgentExecutor` 기본 구성

**학습 목표**:
- `max_iterations`와 `max_execution_time`으로 에이전트 실행을 안전하게 제한할 수 있다
- `early_stopping_method`의 `"force"`와 `"generate"` 차이를 이해하고 상황에 맞게 선택할 수 있다
- `handle_parsing_errors`로 파싱 오류를 유연하게 처리할 수 있다
- `return_intermediate_steps`와 `trim_intermediate_steps`로 에이전트의 추론 과정을 분석할 수 있다

## 왜 알아야 할까?

에이전트는 스스로 판단하고 행동하는 시스템이기 때문에, **통제 없이는 위험할 수 있습니다**. 실제 프로덕션 환경에서 흔히 발생하는 문제들을 떠올려 볼까요?

- LLM이 같은 도구를 무한 반복 호출하면서 **API 비용이 폭발**하는 경우
- 외부 API 응답이 느려서 에이전트가 **몇 분째 멈춰있는** 경우
- LLM이 잘못된 형식으로 응답해서 **파싱 에러로 전체가 중단**되는 경우
- 에이전트가 어떤 경로로 결론에 도달했는지 **추적이 불가능**한 경우

이런 상황을 방지하려면 AgentExecutor의 제어 파라미터를 정확히 이해해야 합니다. 마치 자동차의 브레이크, 속도 제한 장치, 블랙박스처럼 — 에이전트를 안전하고 투명하게 운영하기 위한 필수 도구들이죠.

## 핵심 개념

### 개념 1: max_iterations — 반복 횟수 제한

> 💡 **비유**: `max_iterations`는 마치 **시험의 제한 시간**과 같습니다. 학생(에이전트)이 문제를 풀 때 무한정 고민하게 두면 시험이 끝나지 않겠죠? "최대 15번까지 시도할 수 있어"라고 정해두는 거예요. 15번 안에 답을 못 찾으면, 그때는 다른 방법을 쓰게 됩니다.

`max_iterations`는 에이전트가 Thought→Action→Observation 루프를 **최대 몇 번 반복할 수 있는지** 설정합니다. 기본값은 `15`이며, `None`으로 설정하면 무한 반복이 가능해지므로 **절대 프로덕션에서 사용하지 마세요**.

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

# 도구 정의
@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    return f"'{query}'에 대한 검색 결과: LangChain은 LLM 기반 앱 프레임워크입니다."

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다."""
    try:
        return str(eval(expression))  # 실습용 간단 구현
    except Exception as e:
        return f"계산 오류: {e}"

tools = [search_web, calculator]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ReAct 프롬프트 (세션 12.2에서 배운 형식)
prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

agent = create_react_agent(llm, tools, prompt)

# max_iterations 설정: 최대 5번만 시도
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,  # 기본값 15 → 5로 제한
)

result = executor.invoke({"input": "2의 10승은 얼마인가요?"})
print(result["output"])
# 출력: 1024
```

`max_iterations`가 너무 작으면 복잡한 질문에 답하지 못하고, 너무 크면 비용이 증가합니다. 작업의 복잡도에 따라 적절히 조정하는 것이 핵심이에요.

### 개념 2: max_execution_time — 시간 제한

> 💡 **비유**: 요리 타이머를 생각해보세요. 오븐에 음식을 넣고 "30분 후에 꺼내라"고 타이머를 설정하듯, `max_execution_time`은 에이전트에게 "이 시간 안에 끝내라"고 지시하는 것입니다. 타이머가 울리면 음식이 다 익었든 안 익었든 오븐 문을 열어야 하죠.

`max_execution_time`은 에이전트의 **총 실행 시간을 초 단위**로 제한합니다. 외부 API 호출이 느려지거나, LLM 응답이 지연되는 상황에서 에이전트가 무한 대기하는 것을 방지해줍니다.

```python
import time

@tool
def slow_api_call(query: str) -> str:
    """느린 외부 API를 호출합니다 (시뮬레이션)."""
    time.sleep(3)  # 3초 지연 시뮬레이션
    return f"'{query}'에 대한 API 응답 결과입니다."

tools_with_slow = [search_web, calculator, slow_api_call]
agent_slow = create_react_agent(llm, tools_with_slow, prompt)

# 시간 제한: 최대 10초
executor_timed = AgentExecutor(
    agent=agent_slow,
    tools=tools_with_slow,
    verbose=True,
    max_iterations=10,       # 반복 횟수 제한
    max_execution_time=10,   # 10초 시간 제한
)

result = executor_timed.invoke({"input": "느린 API로 날씨 정보를 검색해주세요"})
print(result["output"])
# 시간 초과 시: "Agent stopped due to iteration limit or time limit."
```

> 🔥 **실무 팁**: `max_iterations`와 `max_execution_time`은 **동시에 설정**하는 것이 좋습니다. 반복 횟수만 제한하면 각 반복이 오래 걸릴 경우 전체 시간이 길어지고, 시간만 제한하면 빠른 반복이 과도하게 많아질 수 있거든요. 둘 중 **하나라도 먼저 도달**하면 에이전트가 멈춥니다.

### 개념 3: early_stopping_method — 조기 종료 전략

> 💡 **비유**: 시험 시간이 끝났을 때 선생님의 대응 방식을 떠올려보세요. **"force"**는 "시간 끝! 답안지 그대로 제출해!"라고 하는 것이고, **"generate"**는 "시간이 다 됐지만, 마지막으로 지금까지 푼 내용을 바탕으로 최종 답을 적어봐"라고 기회를 주는 거예요.

`early_stopping_method`는 `max_iterations`나 `max_execution_time`에 도달했을 때 **어떻게 에이전트를 종료할지** 결정합니다. 두 가지 옵션이 있습니다:

**`"force"` (기본값)**: 즉시 중단하고 고정 메시지를 반환합니다.

```python
# force: 즉시 중단 — "Agent stopped due to iteration limit or time limit."
executor_force = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    early_stopping_method="force",  # 기본값
)

result = executor_force.invoke({"input": "복잡한 다단계 계산을 해주세요"})
print(result["output"])
# 3회 반복 후 강제 종료 → "Agent stopped due to iteration limit or time limit."
```

**`"generate"`**: 지금까지의 중간 단계를 기반으로 LLM에게 **마지막 한 번** 최종 답변 생성을 요청합니다.

```python
# generate: 마지막으로 최선의 답변 생성 시도
executor_generate = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",  # 최종 답변 생성 시도
)

result = executor_generate.invoke({"input": "복잡한 다단계 계산을 해주세요"})
print(result["output"])
# 3회 반복 후 → 지금까지 수집한 정보로 최선의 답변 시도
```

| 방식 | 장점 | 단점 |
|------|------|------|
| `"force"` | 예측 가능, 즉시 종료 | 사용자 경험 나쁨 (고정 문구) |
| `"generate"` | 불완전해도 의미 있는 답변 | 추가 LLM 호출 비용, 잘못된 답 가능성 |

### 개념 4: handle_parsing_errors — 파싱 오류 처리

> 💡 **비유**: 레스토랑에서 웨이터가 주문을 받을 때를 생각해보세요. 손님이 메뉴에 없는 걸 주문하면? **방법 1**: "그런 메뉴 없습니다" 하고 주문을 거부(기본값 — 에러 발생). **방법 2**: "혹시 이것을 말씀하신 건가요?" 하고 다시 물어보기(`True` — LLM에게 다시 전달). **방법 3**: 매니저에게 보내서 판단하게 하기(커스텀 함수).

LLM이 예상한 형식(Action/Action Input)을 지키지 않으면 `OutputParserException`이 발생합니다. `handle_parsing_errors`로 이를 우아하게 처리할 수 있습니다.

```python
# 방법 1: 기본값 (False) — 에러가 그대로 발생
executor_no_handle = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=False,  # 기본값: 파싱 에러 시 예외 발생
)

# 방법 2: True — 에러 메시지를 LLM에게 다시 전달하여 재시도
executor_auto_retry = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # LLM에게 에러를 알려주고 재시도
    verbose=True,
)

# 방법 3: 문자열 — 커스텀 에러 메시지를 LLM에게 전달
executor_custom_msg = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=(
        "출력 형식이 잘못되었습니다. "
        "반드시 'Action: 도구이름' 과 'Action Input: 입력값' 형식을 사용하거나, "
        "'Final Answer: 최종답변' 형식으로 응답하세요."
    ),
    verbose=True,
)

# 방법 4: 콜러블 — 에러를 직접 분석하고 맞춤형 안내 반환
def custom_error_handler(error: Exception) -> str:
    """파싱 에러를 분석하여 맞춤형 안내를 반환합니다."""
    error_msg = str(error)
    if "Could not parse LLM output" in error_msg:
        return (
            "응답 형식을 확인해주세요. "
            "도구를 사용하려면 'Action: 도구이름\\nAction Input: 입력값' 형식을, "
            "최종 답변은 'Final Answer: 답변' 형식을 사용하세요."
        )
    return f"오류가 발생했습니다: {error_msg}. 다시 시도해주세요."

executor_callable = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=custom_error_handler,  # 함수 전달
    verbose=True,
)
```

### 개념 5: return_intermediate_steps — 추론 과정 추적

> 💡 **비유**: 수학 시험에서 "풀이 과정을 보여주세요"라고 하는 것과 같습니다. 최종 답만 보면 맞았는지 알 수 있지만, 풀이 과정을 보면 **어디서 잘못됐는지** 또는 **얼마나 효율적으로 풀었는지** 분석할 수 있죠.

`return_intermediate_steps=True`로 설정하면, 에이전트가 거친 모든 Action과 Observation을 리스트로 반환합니다. 각 단계는 `(AgentAction, str)` 튜플로 구성됩니다.

```python
executor_with_steps = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,  # 중간 단계 반환 활성화
    max_iterations=10,
)

result = executor_with_steps.invoke({"input": "2의 10승을 계산하고 결과를 검색해주세요"})

# 최종 답변
print("최종 답변:", result["output"])

# 중간 단계 분석
for i, (action, observation) in enumerate(result["intermediate_steps"]):
    print(f"\n--- 단계 {i + 1} ---")
    print(f"  도구: {action.tool}")           # 사용한 도구 이름
    print(f"  입력: {action.tool_input}")      # 도구에 전달한 입력
    print(f"  사고: {action.log}")             # LLM의 사고 과정 (Thought)
    print(f"  관찰: {observation}")            # 도구 실행 결과

# 출력 예시:
# --- 단계 1 ---
#   도구: calculator
#   입력: 2**10
#   사고: Thought: 2의 10승을 먼저 계산해야 합니다...
#   관찰: 1024
# --- 단계 2 ---
#   도구: search_web
#   입력: 1024
#   사고: Thought: 이제 1024에 대해 검색해보겠습니다...
#   관찰: '1024'에 대한 검색 결과: ...
```

### 개념 6: trim_intermediate_steps — 중간 단계 트리밍

에이전트가 많은 단계를 거치면 컨텍스트 윈도우가 빠르게 채워집니다. `trim_intermediate_steps`를 사용하면 에이전트에게 전달되는 이전 단계 기록을 줄일 수 있습니다.

```python
# 정수 값: 마지막 N개 단계만 유지
executor_trimmed = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    trim_intermediate_steps=5,  # 최근 5단계만 유지
)

# 콜러블: 커스텀 트리밍 로직
def smart_trim(steps: list) -> list:
    """에러가 발생한 단계는 제거하고 최근 3개만 유지합니다."""
    # 성공한 단계만 필터링
    successful = [(action, obs) for action, obs in steps if "오류" not in obs]
    return successful[-3:]  # 최근 3개만

executor_smart_trim = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    trim_intermediate_steps=smart_trim,  # 함수 전달
)
```

## 실습: 직접 해보기

아래는 모든 제어 파라미터를 조합하여 **안전하고 투명한 에이전트**를 구축하는 완전한 예제입니다. 복사해서 바로 실행해보세요.

```python
"""
AgentExecutor 제어 파라미터 통합 실습
- max_iterations, max_execution_time으로 실행 제한
- early_stopping_method로 조기 종료 전략 설정
- handle_parsing_errors로 파싱 오류 처리
- return_intermediate_steps로 추론 과정 추적
"""
import os
import time
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

# ============================================================
# 1. 도구 정의
# ============================================================
@tool
def search_knowledge(query: str) -> str:
    """내부 지식 베이스에서 정보를 검색합니다."""
    # 시뮬레이션: 실제로는 벡터 스토어 검색 등을 수행
    knowledge = {
        "python": "Python은 1991년 귀도 반 로섬이 만든 범용 프로그래밍 언어입니다.",
        "langchain": "LangChain은 2022년 해리슨 체이스가 만든 LLM 앱 프레임워크입니다.",
        "react": "ReAct는 Reasoning과 Acting을 결합한 에이전트 패턴입니다.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"'{query}'에 대한 정보를 찾을 수 없습니다."

@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다. 예: '2 + 3', '100 * 0.15'"""
    try:
        # 안전한 계산을 위한 허용 문자 검증
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "허용되지 않는 문자가 포함되어 있습니다."
        result = eval(expression)  # 실습용 간단 구현
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"

@tool
def get_current_time(dummy: str = "") -> str:
    """현재 시간을 반환합니다."""
    return f"현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}"

tools = [search_knowledge, calculate, get_current_time]

# ============================================================
# 2. LLM 및 프롬프트 설정
# ============================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, prompt)

# ============================================================
# 3. 커스텀 에러 핸들러 정의
# ============================================================
def parsing_error_handler(error: Exception) -> str:
    """파싱 에러를 분석하여 LLM에게 구체적 가이드를 제공합니다."""
    error_text = str(error)
    if "Could not parse" in error_text:
        return (
            "응답 형식 오류입니다. 반드시 아래 형식 중 하나를 사용하세요:\n"
            "1) Action: 도구이름\\nAction Input: 입력값\n"
            "2) Final Answer: 최종 답변"
        )
    return f"예상치 못한 오류: {error_text}. 다시 시도해주세요."

# ============================================================
# 4. AgentExecutor 구성 — 모든 제어 파라미터 적용
# ============================================================
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                                # 실행 과정 출력
    max_iterations=8,                            # 최대 8회 반복
    max_execution_time=30,                       # 최대 30초
    early_stopping_method="generate",            # 조기 종료 시 최선의 답변 생성
    handle_parsing_errors=parsing_error_handler,  # 커스텀 에러 핸들러
    return_intermediate_steps=True,              # 중간 단계 반환
)

# ============================================================
# 5. 에이전트 실행 및 결과 분석
# ============================================================
print("=" * 60)
print("🤖 에이전트에게 질문합니다...")
print("=" * 60)

query = "LangChain이 뭔지 알려주고, 2024 * 3을 계산한 다음, 현재 시간도 알려줘"
result = executor.invoke({"input": query})

# 최종 답변 출력
print("\n" + "=" * 60)
print("📋 최종 답변:")
print("=" * 60)
print(result["output"])

# 중간 단계 분석
print("\n" + "=" * 60)
print("🔍 에이전트 추론 과정 분석:")
print("=" * 60)

steps = result["intermediate_steps"]
print(f"총 {len(steps)}단계를 거쳤습니다.\n")

for i, (action, observation) in enumerate(steps):
    print(f"📌 단계 {i + 1}:")
    print(f"   🛠️  도구: {action.tool}")
    print(f"   📥 입력: {action.tool_input}")
    print(f"   📤 결과: {observation}")
    print()

# 사용된 도구 통계
tool_usage = {}
for action, _ in steps:
    tool_usage[action.tool] = tool_usage.get(action.tool, 0) + 1

print("📊 도구 사용 통계:")
for tool_name, count in tool_usage.items():
    print(f"   {tool_name}: {count}회")
```

## 더 깊이 알아보기

### AgentExecutor의 탄생 배경

LangChain의 AgentExecutor는 2022년 말 해리슨 체이스(Harrison Chase)가 LangChain을 처음 만들면서 함께 설계했습니다. 초기 에이전트 구현은 매우 단순했는데, LLM에게 도구 사용을 요청하고 결과를 다시 LLM에 전달하는 루프가 전부였죠. 하지만 곧 문제가 드러났습니다 — LLM이 무한 루프에 빠지는 경우가 빈번했던 겁니다.

이 문제를 해결하기 위해 `max_iterations`가 가장 먼저 도입되었고, 이후 실무에서의 다양한 피드백을 반영하여 `max_execution_time`, `early_stopping_method`, `handle_parsing_errors` 등이 순차적으로 추가되었습니다. 특히 `handle_parsing_errors`는 GitHub Issues에서 가장 많이 요청된 기능 중 하나였는데, 초기 LLM들이 정해진 형식을 자주 어겼기 때문이에요.

### AgentExecutor에서 LangGraph로의 진화

LangChain 팀은 AgentExecutor의 한계를 인식하고 **LangGraph**라는 새로운 프레임워크를 개발했습니다. LangChain 1.0부터 AgentExecutor는 공식적으로 **레거시(legacy)** 상태가 되었고, 새로운 프로젝트에는 LangGraph 기반 에이전트를 권장합니다. 그러나 AgentExecutor는 여전히 완전히 동작하며, 에이전트의 **기본 원리를 이해하는 데 최적의 학습 도구**입니다. 실제로 LangGraph의 `create_react_agent`도 내부적으로 같은 제어 개념(반복 제한, 시간 제한, 오류 처리)을 사용하므로, 여기서 배운 개념은 LangGraph로 전환해도 그대로 적용됩니다.

> 💡 **알고 계셨나요?**: `early_stopping_method="generate"`는 사실 초기 구현에서 버그가 있었습니다. [GitHub Issue #16263](https://github.com/langchain-ai/langchain/issues/16263)에서 보고된 바와 같이, 특정 에이전트 타입에서는 `generate` 모드가 제대로 동작하지 않는 문제가 있었습니다. 이 경험에서 얻는 교훈은 — 프로덕션에서는 `"force"`를 기본으로 사용하고, `"generate"`는 충분히 테스트한 후 적용하라는 것입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "`max_iterations=1`이면 도구를 한 번만 호출하겠지?"라고 생각할 수 있지만, 사실 **하나의 iteration은 Thought→Action→Observation 전체 루프**입니다. `max_iterations=1`이면 에이전트는 **딱 한 번** 도구를 호출하고, 그 결과로 최종 답변을 해야 합니다. 복잡한 작업에는 너무 적으니 최소 3 이상을 권장합니다.

> 💡 **알고 계셨나요?**: `handle_parsing_errors=True`로 설정하면 에러 메시지 자체가 LLM의 다음 입력(Observation)으로 들어갑니다. 이 덕분에 LLM이 **자기 수정(self-correction)**을 할 수 있게 됩니다. GPT-4o 같은 최신 모델은 이 피드백을 받으면 거의 대부분 올바른 형식으로 재시도하는데, 이것은 ReAct 패턴의 자기 반성(self-reflection) 능력 덕분이에요.

> 🔥 **실무 팁**: 프로덕션 환경에서의 권장 설정은 다음과 같습니다:
> ```python
> executor = AgentExecutor(
>     agent=agent,
>     tools=tools,
>     max_iterations=10,              # 비용 제어
>     max_execution_time=60,          # 1분 타임아웃
>     early_stopping_method="force",  # 예측 가능한 종료
>     handle_parsing_errors=True,     # 자동 재시도
>     return_intermediate_steps=True, # 디버깅/로깅용
> )
> ```
> `verbose=True`는 개발 중에만 사용하고, 프로덕션에서는 `return_intermediate_steps`로 로그를 남기세요. `verbose`의 출력은 stdout으로 나가므로 로그 시스템과 충돌할 수 있습니다.

## 핵심 정리

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `max_iterations` | `int \| None` | `15` | Thought→Action→Observation 루프의 최대 반복 횟수 |
| `max_execution_time` | `float \| None` | `None` | 총 실행 시간 제한 (초 단위) |
| `early_stopping_method` | `str` | `"force"` | 조기 종료 시 전략: `"force"`(즉시 중단) 또는 `"generate"`(최종 답변 생성) |
| `handle_parsing_errors` | `bool \| str \| Callable` | `False` | 파싱 오류 처리: `True`(자동 재시도), 문자열(커스텀 메시지), 함수(커스텀 로직) |
| `return_intermediate_steps` | `bool` | `False` | 에이전트의 모든 중간 단계를 결과에 포함할지 여부 |
| `trim_intermediate_steps` | `int \| Callable` | `-1` | 에이전트에 전달할 이전 단계 수를 제한 (토큰 절약) |

## 다음 섹션 미리보기

이제 AgentExecutor를 안전하게 제어하는 방법을 배웠으니, 다음 세션에서는 **에이전트의 추론 과정을 실시간으로 모니터링하고 디버깅하는 방법**을 배웁니다. `verbose` 출력을 넘어서 콜백(Callback)을 활용한 정밀한 모니터링, 에이전트가 잘못된 추론을 할 때의 디버깅 전략, 그리고 LangSmith를 사용한 트레이싱까지 다뤄볼 예정입니다. AgentExecutor의 설정을 "왜 그렇게 해야 하는지" 더 깊이 이해할 수 있을 거예요.

## 참고 자료

- [AgentExecutor API Reference — LangChain 공식 문서](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html) - AgentExecutor의 모든 파라미터와 메서드가 정리된 공식 API 레퍼런스
- [Cap the max number of iterations — LangChain How-to](https://python.langchain.com/v0.1/docs/modules/agents/how_to/max_iterations/) - max_iterations 사용법을 다룬 공식 가이드
- [Handle parsing errors — LangChain How-to](https://python.langchain.com/v0.1/docs/modules/agents/how_to/handle_parsing_errors/) - handle_parsing_errors의 세 가지 사용 패턴을 설명하는 공식 가이드
- [early_stopping_method "generate" Issue #16263](https://github.com/langchain-ai/langchain/issues/16263) - early_stopping_method의 알려진 이슈와 해결 과정을 다룬 GitHub 이슈
- [LangChain and LangGraph 1.0 Milestones — 공식 블로그](https://blog.langchain.com/langchain-langgraph-1dot0/) - AgentExecutor에서 LangGraph로의 전환 배경과 향후 방향

---
### 🔗 Related Sessions
- [agent](../12-에이전트agent-기초/01-에이전트-개념과-react-패턴.md) (prerequisite)
- [react_pattern](../12-에이전트agent-기초/01-에이전트-개념과-react-패턴.md) (prerequisite)
- [agent_executor](../12-에이전트agent-기초/01-에이전트-개념과-react-패턴.md) (prerequisite)
- [intermediate_steps](../12-에이전트agent-기초/01-에이전트-개념과-react-패턴.md) (prerequisite)
- [create_react_agent_usage](../12-에이전트agent-기초/02-create-react-agent로-에이전트-구축.md) (prerequisite)
- [react_prompt_variables](../12-에이전트agent-기초/02-create-react-agent로-에이전트-구축.md) (prerequisite)
