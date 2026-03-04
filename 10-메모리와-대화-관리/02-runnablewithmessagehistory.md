# RunnableWithMessageHistory

> LCEL 체인에 대화 히스토리를 자동으로 주입하고 관리하는 핵심 래퍼(Wrapper)

## 개요

이 섹션에서는 LangChain의 `RunnableWithMessageHistory`를 학습합니다. 앞서 [세션 10.1: 메시지 히스토리 기초](./01-메시지-히스토리-기초.md)에서 `InMemoryChatMessageHistory`로 메시지를 수동으로 추가하고 조회하는 방법을 배웠는데요, 이번에는 그 히스토리를 **LCEL 체인에 자동으로 연결**하는 방법을 다룹니다.

**선수 지식**: 세션 10.1의 `BaseChatMessageHistory`, `InMemoryChatMessageHistory`, `session_id` 패턴, 그리고 [챕터 5](05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md)에서 배운 LCEL 파이프 연산자(`|`)와 `Runnable` 인터페이스

**학습 목표**:
- `RunnableWithMessageHistory`의 동작 원리(Load → Run → Save 패턴)를 이해한다
- `input_messages_key`, `history_messages_key`, `output_messages_key`의 역할을 구분할 수 있다
- `session_id` 기반으로 여러 독립 대화를 동시에 관리하는 체인을 구축할 수 있다
- `ConfigurableFieldSpec`을 활용한 커스텀 설정 패턴을 이해한다

## 왜 알아야 할까?

세션 10.1에서 우리는 히스토리 객체에 메시지를 직접 추가하고, 체인을 호출할 때마다 히스토리를 수동으로 꺼내 프롬프트에 넣어야 했습니다. 대화가 한두 번이라면 괜찮지만, 실제 챗봇은 어떨까요?

- 사용자 A의 3번째 질문이 들어오면, A의 이전 2개 메시지를 찾아서 프롬프트에 넣고
- 사용자 B가 동시에 질문하면, B의 별도 히스토리를 찾아서 넣고
- 응답이 나오면, 다시 각 사용자의 히스토리에 저장하고...

이 모든 작업을 매번 수동으로 하면 코드가 금방 복잡해지겠죠? `RunnableWithMessageHistory`는 이 **"히스토리 로드 → 체인 실행 → 히스토리 저장"의 전체 사이클을 자동화**해주는 래퍼입니다. LCEL 체인을 감싸기만 하면, `session_id` 하나로 알아서 올바른 대화 히스토리를 주입하고 새 메시지를 저장해줍니다.

프로덕션 챗봇, 고객 상담 시스템, 멀티턴 QA 등 **대화 맥락이 필요한 모든 LLM 애플리케이션**의 기반이 되는 패턴이기 때문에, LangChain 개발자라면 반드시 알아야 하는 컴포넌트입니다.

## 핵심 개념

### 개념 1: RunnableWithMessageHistory의 동작 원리 — "자동 비서" 패턴

> 💡 **비유**: 회의실에 자동 비서가 있다고 상상해보세요. 회의 참석자(session_id)가 들어오면, 비서는 (1) 그 사람의 이전 회의록을 캐비닛에서 꺼내 테이블 위에 펼쳐놓고 → (2) 회의를 진행한 뒤 → (3) 새로 나온 내용을 기록해서 다시 캐비닛에 넣습니다. 참석자는 회의록 관리를 전혀 신경 쓸 필요가 없죠. `RunnableWithMessageHistory`가 바로 이 자동 비서입니다.

`RunnableWithMessageHistory`는 **Load → Run → Save** 3단계로 동작합니다:

1. **Load(불러오기)**: `session_id`로 해당 세션의 히스토리를 조회하여, 프롬프트의 `MessagesPlaceholder` 위치에 주입합니다
2. **Run(실행)**: 히스토리가 포함된 완전한 프롬프트로 래핑된 체인(LLM 등)을 실행합니다
3. **Save(저장)**: 사용자의 새 입력 메시지와 AI의 응답 메시지를 히스토리에 자동 저장합니다

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 세션별 히스토리 저장소
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """session_id에 해당하는 히스토리를 반환 (없으면 새로 생성)"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# LCEL 체인 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 도우미입니다."),
    MessagesPlaceholder(variable_name="history"),  # 히스토리가 주입될 위치
    ("human", "{input}"),                          # 현재 사용자 입력
])

chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0.7)

# RunnableWithMessageHistory로 래핑
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,               # 히스토리 팩토리 함수
    input_messages_key="input",        # 사용자 입력이 담긴 키
    history_messages_key="history",    # 히스토리가 주입될 키
)

# 실행 — session_id는 config로 전달
response = with_history.invoke(
    {"input": "안녕! 나는 민수야."},
    config={"configurable": {"session_id": "user_001"}},
)
print(response.content)
# 출력: "안녕하세요, 민수님! 무엇을 도와드릴까요?"
```

핵심은 **`config={"configurable": {"session_id": "..."}}`** 형태로 세션을 식별한다는 점입니다. 같은 `session_id`로 다시 호출하면, 이전 대화가 자동으로 포함됩니다.

### 개념 2: 세 가지 핵심 키 파라미터

> 💡 **비유**: 우체국에서 편지를 보낸다고 생각해보세요. `input_messages_key`는 **보내는 사람 칸**(어디에 새 메시지가 있는지), `history_messages_key`는 **이전 서신 파일 칸**(과거 대화를 어디에 끼워 넣을지), `output_messages_key`는 **받는 사람 응답 칸**(AI 응답을 어디서 꺼낼지)입니다.

`RunnableWithMessageHistory`의 생성자에서 가장 중요한 파라미터 세 가지를 살펴볼까요?

| 파라미터 | 역할 | 언제 필요한가 |
|---------|------|-------------|
| `input_messages_key` | 입력 딕셔너리에서 **사용자 메시지**가 담긴 키 | 입력이 딕셔너리일 때 |
| `history_messages_key` | 프롬프트에서 **과거 히스토리**가 주입될 키 | 히스토리와 입력을 분리할 때 |
| `output_messages_key` | 출력 딕셔너리에서 **AI 응답**이 담긴 키 | 출력이 딕셔너리일 때 |

입력/출력 형태에 따라 필요한 키가 달라집니다:

```python
# 패턴 1: 딕셔너리 입력 + 히스토리 별도 키
# → input_messages_key와 history_messages_key 모두 지정
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",       # {"input": "사용자 메시지"}
    history_messages_key="history",   # MessagesPlaceholder("history")에 매핑
)

# 패턴 2: 메시지 리스트 직접 입력 (ChatModel에 직접 연결)
# → 키 지정 불필요, 메시지 자체가 입력이자 히스토리
model = ChatOpenAI(model="gpt-4o")
with_history_simple = RunnableWithMessageHistory(
    model,
    get_session_history,
    # input_messages_key, history_messages_key 모두 생략 가능
)

# 패턴 3: 출력이 딕셔너리인 경우 (예: RunnableParallel 출력)
# → output_messages_key로 응답 위치 지정
with_history_dict_output = RunnableWithMessageHistory(
    chain_with_dict_output,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",     # {"answer": AIMessage(...)} 에서 추출
)
```

> ⚠️ **흔한 오해**: `history_messages_key`와 `MessagesPlaceholder`의 `variable_name`이 **반드시 일치**해야 합니다. 둘이 다르면 히스토리가 주입되지 않고, 매번 첫 대화처럼 동작합니다. 디버깅 시 가장 먼저 확인할 포인트예요!

### 개념 3: session_id 기반 멀티 세션 관리

> 💡 **비유**: 병원의 접수 시스템을 떠올려보세요. 환자 번호(session_id)만 대면, 접수 직원이 알아서 그 환자의 차트(히스토리)를 꺼내옵니다. 다른 환자의 차트와 절대 섞이지 않죠.

하나의 `RunnableWithMessageHistory` 인스턴스로 여러 사용자의 대화를 동시에 관리할 수 있습니다:

```python
# 사용자 A의 대화
response_a1 = with_history.invoke(
    {"input": "나는 파이썬 개발자야."},
    config={"configurable": {"session_id": "user_A"}},
)
# → "안녕하세요! 파이썬 개발자시군요..."

response_a2 = with_history.invoke(
    {"input": "내가 뭘 한다고 했지?"},
    config={"configurable": {"session_id": "user_A"}},
)
# → "파이썬 개발을 하신다고 하셨습니다." (이전 대화 기억!)

# 사용자 B의 대화 (완전히 독립)
response_b1 = with_history.invoke(
    {"input": "오늘 날씨 어때?"},
    config={"configurable": {"session_id": "user_B"}},
)
# → 사용자 A의 대화 내용은 전혀 모름

# 히스토리 확인
print(len(store["user_A"].messages))  # 4 (user 2개 + ai 2개)
print(len(store["user_B"].messages))  # 2 (user 1개 + ai 1개)
```

`session_id`만 다르게 넘기면, 각 세션의 히스토리가 완전히 격리됩니다. 동일한 `session_id`를 재사용하면 이전 대화를 이어갈 수 있고, 새 `session_id`를 생성하면 새 대화가 시작됩니다.

### 개념 4: ConfigurableFieldSpec으로 커스텀 설정하기

기본적으로 `session_id` 하나만 사용하지만, 실무에서는 `user_id`와 `conversation_id`처럼 **복합 키**가 필요할 때가 많습니다:

```python
from langchain_core.runnables import ConfigurableFieldSpec

with_history_custom = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 아래에서 시그니처가 변경됨
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="사용자 ID",
            description="고유 사용자 식별자",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="대화 ID",
            description="대화 세션 식별자",
            default="",
            is_shared=True,
        ),
    ],
)
```

이 경우 `get_session_history` 함수의 시그니처도 바뀝니다:

```python
def get_session_history(user_id: str, conversation_id: str) -> InMemoryChatMessageHistory:
    """user_id + conversation_id 조합으로 히스토리 관리"""
    key = f"{user_id}:{conversation_id}"
    if key not in store:
        store[key] = InMemoryChatMessageHistory()
    return store[key]

# 호출 시 두 개의 설정값을 전달
response = with_history_custom.invoke(
    {"input": "안녕하세요!"},
    config={
        "configurable": {
            "user_id": "user_123",
            "conversation_id": "conv_456",
        }
    },
)
```

이렇게 하면 같은 사용자의 여러 대화 세션을 독립적으로 관리할 수 있어, 실제 채팅 서비스처럼 "대화방" 개념을 구현할 수 있습니다.

## 실습: 직접 해보기

아래는 `RunnableWithMessageHistory`를 활용한 **멀티 세션 챗봇**을 처음부터 끝까지 구축하는 완전한 코드입니다. 복사-붙여넣기로 바로 실행할 수 있습니다.

```python
"""
RunnableWithMessageHistory 실습: 멀티 세션 챗봇
- 여러 사용자의 대화를 독립적으로 관리
- 대화 맥락을 유지하며 응답
"""
import os
from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# ── 1단계: 세션 히스토리 관리 설정 ──────────────────────────
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """session_id에 해당하는 히스토리를 반환합니다."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ── 2단계: 프롬프트 템플릿 구성 ─────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 친절하고 기억력 좋은 AI 도우미입니다. "
     "사용자가 이전에 말한 내용을 잘 기억하고, "
     "자연스럽게 대화를 이어가세요."),
    MessagesPlaceholder(variable_name="history"),  # 히스토리 주입 위치
    ("human", "{input}"),
])

# ── 3단계: LCEL 체인 구성 ───────────────────────────────────
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
chain = prompt | model | StrOutputParser()  # 문자열 출력

# ── 4단계: RunnableWithMessageHistory로 래핑 ────────────────
# StrOutputParser를 사용하면 출력이 str이므로 output_messages_key 불필요
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ── 5단계: 대화 실행 ────────────────────────────────────────
def chat(session_id: str, message: str) -> str:
    """주어진 세션에서 메시지를 보내고 응답을 받습니다."""
    response = with_history.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )
    return response

# ── 대화 시나리오 ───────────────────────────────────────────
# 사용자 A: 자기소개 대화
print("=== 사용자 A ===")
print(chat("alice", "안녕! 나는 앨리스야. 백엔드 개발자로 일하고 있어."))
# 출력 예: "안녕하세요 앨리스님! 백엔드 개발자시군요. 반갑습니다..."

print(chat("alice", "요즘 FastAPI를 배우고 있는데, 팁 있어?"))
# 출력 예: "백엔드 개발자시니 FastAPI 배우시는 건 좋은 선택이에요..."

# 사용자 B: 완전히 다른 대화
print("\n=== 사용자 B ===")
print(chat("bob", "LangChain으로 RAG 시스템을 만들고 싶어."))
# 출력 예: "RAG 시스템 구축이요! 좋은 주제네요..."

# 사용자 A: 이전 대화 맥락 유지 확인
print("\n=== 사용자 A (이어서) ===")
print(chat("alice", "내가 뭘 배우고 있다고 했지?"))
# 출력 예: "FastAPI를 배우고 계신다고 하셨어요!"

# ── 히스토리 상태 확인 ──────────────────────────────────────
print("\n=== 히스토리 상태 ===")
for sid, history in store.items():
    print(f"  세션 '{sid}': 메시지 {len(history.messages)}개")
    for msg in history.messages:
        role = "👤" if msg.type == "human" else "🤖"
        # 메시지 내용을 50자로 잘라서 출력
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"    {role} {content}")
```

**실행 결과 예시**:
```
=== 사용자 A ===
안녕하세요 앨리스님! 백엔드 개발자시군요, 반갑습니다. 무엇을 도와드릴까요?
백엔드 개발자시니 FastAPI를 배우시는 건 정말 좋은 선택이에요! 몇 가지 팁을 드릴게요...

=== 사용자 B ===
RAG 시스템 구축이요! 좋은 주제네요. 어떤 종류의 문서를 다루실 건가요?

=== 사용자 A (이어서) ===
FastAPI를 배우고 계신다고 하셨어요!

=== 히스토리 상태 ===
  세션 'alice': 메시지 6개
    👤 안녕! 나는 앨리스야. 백엔드 개발자로 일하고 있어.
    🤖 안녕하세요 앨리스님! 백엔드 개발자시군요, 반갑습니다...
    👤 요즘 FastAPI를 배우고 있는데, 팁 있어?
    🤖 백엔드 개발자시니 FastAPI를 배우시는 건 정말 좋은 선...
    👤 내가 뭘 배우고 있다고 했지?
    🤖 FastAPI를 배우고 계신다고 하셨어요!
  세션 'bob': 메시지 2개
    👤 LangChain으로 RAG 시스템을 만들고 싶어.
    🤖 RAG 시스템 구축이요! 좋은 주제네요. 어떤 종류의 문서...
```

## 더 깊이 알아보기

### RunnableWithMessageHistory의 탄생 배경

LangChain의 메모리 시스템은 상당한 변천사를 거쳤습니다. 초기 LangChain(v0.0.x 시절)에는 `ConversationBufferMemory`, `ConversationSummaryMemory` 같은 **레거시 Memory 클래스**들이 있었는데요, 이들은 `LLMChain`이라는 구식 체인 인터페이스에 묶여 있었습니다. 문제는 LangChain이 LCEL(LangChain Expression Language)이라는 새로운 선언적 체인 구성 방식을 도입하면서 시작됐어요.

LCEL의 핵심 철학은 **모든 컴포넌트가 `Runnable` 인터페이스를 구현**하여, `invoke`, `stream`, `batch` 같은 통합 메서드를 사용할 수 있게 하는 것이었습니다. 그런데 레거시 Memory 클래스들은 이 `Runnable` 프로토콜과 맞지 않았어요. 체인 외부에서 상태를 관리하는 방식이라 LCEL의 "합성 가능한 파이프라인" 패턴과 충돌했죠.

LangChain 창립자 해리슨 체이스(Harrison Chase)와 팀은 2023년 말~2024년 초에 이 문제를 해결하기 위해 `RunnableWithMessageHistory`를 설계했습니다. 핵심 아이디어는 간단했어요: **메모리를 체인 내부에 포함시키지 말고, 체인을 감싸는 래퍼로 만들자.** 이렇게 하면 어떤 LCEL 체인이든 — 단순한 `ChatModel`이든, 복잡한 RAG 파이프라인이든 — 히스토리 관리 기능을 **비침투적으로** 추가할 수 있게 됩니다.

이후 LangGraph가 등장하면서 상태 관리의 패러다임이 한 번 더 바뀌었는데요, `RunnableWithMessageHistory`는 여전히 **"간단한 체인에 대화 메모리를 추가하는 가장 빠른 방법"**으로서 핵심적인 위치를 차지하고 있습니다. 복잡한 에이전트 워크플로우에는 LangGraph의 `StateGraph`와 체크포인트가 더 적합하지만, 일반적인 멀티턴 챗봇이라면 `RunnableWithMessageHistory`가 최적의 선택입니다.

### 내부 동작: _enter_history와 _exit_history

`RunnableWithMessageHistory`의 소스 코드를 들여다보면, 두 개의 핵심 내부 메서드가 있습니다:

- **`_enter_history`**: 체인 실행 전에 호출됩니다. `get_session_history`로 히스토리를 불러와 기존 메시지 목록을 복사하고, `history_messages_key`가 있으면 해당 위치에 주입합니다.
- **`_exit_history`**: 체인 실행 후에 호출됩니다. 입력 메시지와 출력 메시지를 추출하여, 중복을 제거한 뒤 히스토리에 추가합니다.

이 "입장-퇴장" 패턴 덕분에, 래핑된 체인은 히스토리의 존재를 전혀 몰라도 됩니다. 체인 입장에서는 그냥 "이전 메시지가 포함된 프롬프트"를 받을 뿐이에요.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "RunnableWithMessageHistory는 자체적으로 메시지를 저장한다" — 아닙니다! 실제 저장은 `get_session_history`가 반환하는 `BaseChatMessageHistory` 구현체가 담당합니다. `RunnableWithMessageHistory`는 **오케스트레이터**일 뿐, 저장 로직은 히스토리 객체에 위임합니다. 따라서 저장소를 바꾸고 싶다면(예: Redis, PostgreSQL), `get_session_history` 함수만 교체하면 됩니다.

> 💡 **알고 계셨나요?**: `RunnableWithMessageHistory`는 `stream()`도 지원합니다. 스트리밍 모드에서도 히스토리 관리가 자동으로 이루어지기 때문에, 실시간 채팅 UX를 구현할 때도 똑같은 패턴을 사용할 수 있어요:
> ```python
> for chunk in with_history.stream(
>     {"input": "스트리밍 테스트!"},
>     config={"configurable": {"session_id": "stream_session"}},
> ):
>     print(chunk, end="", flush=True)
> ```

> 🔥 **실무 팁**: `config`에 `session_id`를 넘기지 않으면 `ValueError`가 발생합니다. 프로덕션에서는 미들웨어나 데코레이터로 `config` 주입을 자동화하는 것이 좋습니다. 예를 들어, FastAPI의 `Depends`로 요청 헤더에서 `session_id`를 추출하여 자동 주입하는 패턴이 일반적입니다:
> ```python
> @app.post("/chat")
> async def chat_endpoint(
>     request: ChatRequest,
>     session_id: str = Header(...),
> ):
>     response = with_history.invoke(
>         {"input": request.message},
>         config={"configurable": {"session_id": session_id}},
>     )
>     return {"response": response}
> ```

> 🔥 **실무 팁**: `StrOutputParser`를 체인 마지막에 사용하면 출력이 문자열이 됩니다. 이 경우 `RunnableWithMessageHistory`는 출력 문자열을 `AIMessage`로 자동 변환하여 히스토리에 저장합니다. 반면, `ChatModel`만 사용하면 `AIMessage` 객체가 직접 반환되므로 변환 없이 저장됩니다. 어느 쪽이든 히스토리 관리는 자동으로 처리되니 걱정하지 마세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `RunnableWithMessageHistory` | LCEL 체인을 감싸 히스토리 Load → Run → Save를 자동화하는 래퍼 |
| `get_session_history` | `session_id`를 받아 `BaseChatMessageHistory` 인스턴스를 반환하는 팩토리 함수 |
| `input_messages_key` | 입력 딕셔너리에서 사용자 메시지가 담긴 키 이름 |
| `history_messages_key` | 프롬프트의 `MessagesPlaceholder`와 매핑되는 키 이름 (일치 필수!) |
| `output_messages_key` | 출력이 딕셔너리일 때 AI 응답이 담긴 키 이름 |
| `session_id` | `config["configurable"]`로 전달하며, 세션을 고유하게 식별하는 문자열 |
| `ConfigurableFieldSpec` | `user_id + conversation_id` 같은 복합 키를 사용할 때 설정 스펙을 정의하는 객체 |
| Load → Run → Save | 히스토리 불러오기 → 체인 실행 → 새 메시지 저장의 3단계 자동화 사이클 |

## 다음 섹션 미리보기

이번 섹션에서는 인메모리(`InMemoryChatMessageHistory`) 기반의 히스토리 관리를 다뤘는데요, 서버가 재시작되면 모든 대화가 사라진다는 치명적인 한계가 있습니다. 다음 섹션 **[세션 10.3: 영구 메모리 저장소](./03-영구-메시지-저장소.md)**에서는 Redis, SQLite, PostgreSQL 등 **영구 저장소**에 대화 히스토리를 저장하고 복원하는 방법을 학습합니다. `get_session_history` 함수만 교체하면 된다는 이번 섹션의 설계가 어떻게 빛을 발하는지 직접 확인해보세요.

## 참고 자료

- [How to add message history — LangChain 공식 가이드](https://python.langchain.com/docs/how_to/message_history/) - RunnableWithMessageHistory 사용법을 단계별로 설명하는 공식 How-to 가이드
- [RunnableWithMessageHistory API Reference — LangChain](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html) - 전체 파라미터와 타입 정의를 확인할 수 있는 API 레퍼런스
- [RunnableWithMessageHistory 소스 코드 — GitHub](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/history.py) - 내부 동작을 이해하고 싶다면 소스 코드를 직접 읽어보세요
- [LangChain Memory Tutorial — LangChain Tutorials](https://langchain-tutorials.com/lessons/langchain-essentials/lesson-8) - RunnableWithMessageHistory를 활용한 실전 챗봇 구축 튜토리얼

---
### 🔗 Related Sessions
- [lcel](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [runnable](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [chatprompttemplate](01-langchain-소개와-개발-환경-설정/04-첫-번째-langchain-애플리케이션.md) (prerequisite)
- [messagesplaceholder](03-프롬프트-엔지니어링과-템플릿/02-고급-프롬프트-패턴.md) (prerequisite)
- [basechatmessagehistory](./01-메시지-히스토리-기초.md) (prerequisite)
- [inmemorychatmessagehistory](./01-메시지-히스토리-기초.md) (prerequisite)
- [session_id_pattern](./01-메시지-히스토리-기초.md) (prerequisite)
- [get_session_history](09-ragretrieval-augmented-generation-구축/03-대화형-rag.md) (prerequisite)
