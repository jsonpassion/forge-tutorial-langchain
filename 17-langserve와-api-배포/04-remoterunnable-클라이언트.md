# RemoteRunnable 클라이언트

> LangServe API를 로컬 Runnable처럼 호출하는 강력한 클라이언트 SDK

## 개요

이 섹션에서는 LangServe로 배포된 체인을 원격에서 호출하는 `RemoteRunnable` 클라이언트의 모든 것을 다룹니다. 단순한 `invoke` 호출부터 실시간 스트리밍, 대량 배치 처리, 그리고 프로덕션 환경에서 필수적인 에러 핸들링과 재시도 전략까지 종합적으로 학습합니다.

**선수 지식**: [17.1 LangServe 기초](./01-langserve-기초.md)에서 배운 `add_routes`와 기본 `RemoteRunnable` 개념, [17.2 FastAPI 통합과 서버 구성](./02-fastapi-통합과-서버-구성.md)에서 다룬 다중 체인 배포, [17.3 인증과 보안](./03-인증과-보안.md)에서 학습한 API 키/JWT 인증

**학습 목표**:
- RemoteRunnable의 전체 초기화 옵션(타임아웃, 인증, 헤더 등)을 설정할 수 있다
- 동기/비동기 스트리밍으로 실시간 응답을 처리할 수 있다
- 배치 요청으로 여러 입력을 병렬 처리할 수 있다
- httpx 기반 에러 핸들링과 재시도 패턴을 구현할 수 있다

## 왜 알아야 할까?

서버에 체인을 배포했다면, 그 다음은 당연히 "어떻게 호출하지?"겠죠. 물론 `requests` 라이브러리로 HTTP POST를 보내도 됩니다. 하지만 그러면 입출력 직렬화, 스트리밍 파싱, 에러 처리를 전부 직접 구현해야 합니다.

`RemoteRunnable`은 이 모든 복잡성을 감춰줍니다. **원격 서버의 체인을 마치 로컬에서 실행하는 것처럼** `invoke()`, `stream()`, `batch()`를 그대로 호출할 수 있거든요. 더 놀라운 건 `RemoteRunnable` 자체가 `Runnable`이기 때문에, 다른 로컬 컴포넌트와 파이프 연산자(`|`)로 조합할 수도 있다는 점입니다.

실제 프로덕션에서는 마이크로서비스 아키텍처에서 여러 LangServe 서버를 조합하거나, 프론트엔드 백엔드가 분리된 환경에서 AI 체인을 호출하는 경우가 대부분입니다. 이때 `RemoteRunnable`은 단순한 HTTP 클라이언트가 아닌, LangChain 생태계의 **일급 시민**으로서 역할합니다.

## 핵심 개념

### 개념 1: RemoteRunnable 초기화와 설정

> 💡 **비유**: RemoteRunnable은 마치 TV 리모컨과 같습니다. 리모컨(클라이언트)은 TV(서버)의 모든 기능을 버튼 하나로 조작하죠. 적외선 주파수(URL)만 맞추면, 채널 변경(invoke), 볼륨 조절(stream), 여러 TV 동시 제어(batch) 모두 가능합니다. 리모컨 설정에서 타임아웃(반응 대기 시간)이나 보안 페어링(인증)도 조정할 수 있고요.

`RemoteRunnable`은 httpx 기반의 HTTP 클라이언트를 내부에 품고 있습니다. 초기화 시 다양한 옵션을 설정할 수 있는데, 그 전체 시그니처를 살펴보겠습니다:

```python
from langserve import RemoteRunnable

# 기본 초기화 — URL만 지정
chain = RemoteRunnable("http://localhost:8000/my-chain/")

# 프로덕션용 전체 옵션 초기화
chain = RemoteRunnable(
    url="https://api.example.com/my-chain/",
    timeout=30.0,                    # 요청 타임아웃 (초)
    headers={"X-API-Key": "my-key"}, # 커스텀 헤더
    auth=("user", "password"),       # HTTP 기본 인증
    verify=True,                     # SSL 인증서 검증
    cookies={"session": "abc123"},   # 쿠키
    client_kwargs={                  # httpx 추가 설정
        "follow_redirects": True,
        "max_redirects": 5,
    },
)
```

주요 매개변수를 정리하면:

| 매개변수 | 타입 | 설명 |
|----------|------|------|
| `url` | `str` | 서버 엔드포인트 URL (끝에 `/` 자동 추가) |
| `timeout` | `Optional[float]` | 요청 타임아웃 (초 단위) |
| `headers` | `Optional[dict]` | HTTP 헤더 (API 키 등) |
| `auth` | `Optional[tuple]` | HTTP 인증 정보 |
| `verify` | `bool \| str` | SSL 검증 (`True`, `False`, 또는 인증서 경로) |
| `cookies` | `Optional[dict]` | HTTP 쿠키 |
| `client_kwargs` | `Optional[dict]` | httpx 클라이언트 추가 옵션 |

내부적으로 `RemoteRunnable`은 **동기 클라이언트**(`httpx.Client`)와 **비동기 클라이언트**(`httpx.AsyncClient`)를 동시에 생성합니다. 객체가 가비지 컬렉션될 때 `weakref.finalize`를 통해 자동으로 클라이언트를 정리하므로, 메모리 누수 걱정 없이 사용할 수 있습니다.

### 개념 2: invoke와 ainvoke — 단일 호출

> 💡 **비유**: `invoke`는 카페에서 커피를 주문하고 완성될 때까지 카운터 앞에서 기다리는 것과 같습니다. 반면 `ainvoke`는 진동벨을 받고 자리에서 다른 일을 하다가 벨이 울리면 가져오는 방식이죠.

가장 기본적인 호출 방법입니다. 동기 `invoke`는 응답이 올 때까지 블로킹하고, 비동기 `ainvoke`는 이벤트 루프를 차단하지 않습니다.

```python
from langserve import RemoteRunnable

# 서버 연결
chain = RemoteRunnable("http://localhost:8000/translate/")

# 동기 호출 — 응답 완료까지 대기
result = chain.invoke({"text": "Hello, world!", "language": "Korean"})
print(result)  # "안녕하세요, 세계!"

# 비동기 호출 — async 컨텍스트에서 사용
import asyncio

async def main():
    result = await chain.ainvoke({"text": "Hello!", "language": "Japanese"})
    print(result)  # "こんにちは！"

asyncio.run(main())
```

`invoke`와 `ainvoke` 모두 내부적으로 서버의 `/invoke` 엔드포인트에 POST 요청을 보냅니다. 입력은 자동으로 JSON 직렬화되고, 응답은 자동으로 역직렬화됩니다.

> ⚠️ **흔한 오해**: `invoke`와 `ainvoke`의 차이는 성능이 아니라 **동시성 모델**입니다. 단일 호출의 응답 속도는 동일하지만, `ainvoke`를 사용하면 하나의 호출이 진행되는 동안 다른 작업을 동시에 처리할 수 있습니다.

### 개념 3: stream과 astream — 실시간 스트리밍

> 💡 **비유**: 스트리밍은 영화 다운로드와 넷플릭스의 차이입니다. 다운로드(`invoke`)는 영화 전체가 받아져야 볼 수 있지만, 스트리밍(`stream`)은 데이터가 도착하는 즉시 화면에 보여줍니다. LLM 응답을 토큰 단위로 실시간 표시하고 싶을 때 바로 이 스트리밍이 필요합니다.

LLM 기반 체인은 응답 생성에 수 초가 걸릴 수 있습니다. 사용자 경험을 위해 토큰이 생성되는 대로 실시간으로 보여주는 게 훨씬 좋겠죠.

```python
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chat/")

# 동기 스트리밍
for chunk in chain.stream({"question": "양자 컴퓨터를 쉽게 설명해주세요"}):
    print(chunk, end="", flush=True)
# 출력: 양자 컴퓨터는... (토큰 단위로 실시간 출력)
```

비동기 스트리밍은 웹 서버나 GUI 애플리케이션처럼 이벤트 루프가 돌아가는 환경에서 특히 유용합니다:

```python
import asyncio
from langserve import RemoteRunnable

async def stream_response():
    chain = RemoteRunnable("http://localhost:8000/chat/")
    
    # 비동기 스트리밍
    full_response = ""
    async for chunk in chain.astream({"question": "파이썬의 장점은?"}):
        print(chunk, end="", flush=True)
        full_response += str(chunk)
    
    print(f"\n\n--- 전체 응답 길이: {len(full_response)}자 ---")

asyncio.run(stream_response())
```

더 세밀한 제어가 필요하다면 `astream_events`를 사용할 수 있습니다. 이 메서드는 체인 내부의 각 단계에서 발생하는 이벤트를 스트리밍합니다:

```python
async def stream_with_events():
    chain = RemoteRunnable("http://localhost:8000/rag/")
    
    # 이벤트 스트리밍 — 체인 내부 동작까지 관찰
    async for event in chain.astream_events(
        {"question": "LangChain이란?"},
        version="v2",  # 이벤트 스키마 버전
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # LLM 토큰 출력
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_retriever_end":
            # 검색 완료 시 문서 수 표시
            docs = event["data"]["output"]
            print(f"\n[검색 완료: {len(docs)}개 문서 발견]")
```

### 개념 4: batch와 abatch — 배치 처리

> 💡 **비유**: 배치 처리는 세탁기에 빨래를 한 번에 넣는 것과 같습니다. 셔츠 하나씩 세탁하는 것(개별 invoke)보다 여러 벌을 한꺼번에 돌리는 게(batch) 훨씬 효율적이죠. 네트워크 오버헤드를 줄이고 서버 측 병렬 처리도 가능합니다.

여러 입력을 한 번에 처리할 때는 `batch`를 사용합니다. 서버의 `/batch` 엔드포인트를 호출하므로 개별 `invoke`를 반복하는 것보다 효율적입니다:

```python
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/summarize/")

# 여러 문서를 동시에 요약
documents = [
    {"text": "인공지능의 역사는 1956년 다트머스 회의에서..."},
    {"text": "딥러닝의 핵심은 역전파 알고리즘으로..."},
    {"text": "트랜스포머 아키텍처는 2017년 구글이 발표한..."},
]

# 동기 배치 — 모든 결과를 한 번에 반환
results = chain.batch(documents)
for i, result in enumerate(results):
    print(f"문서 {i+1} 요약: {result[:50]}...")
```

비동기 배치도 지원됩니다:

```python
import asyncio
from langserve import RemoteRunnable

async def batch_process():
    chain = RemoteRunnable("http://localhost:8000/translate/")
    
    inputs = [
        {"text": "Hello", "language": "Korean"},
        {"text": "Goodbye", "language": "Korean"},
        {"text": "Thank you", "language": "Korean"},
    ]
    
    # 비동기 배치
    results = await chain.abatch(inputs)
    for original, translated in zip(inputs, results):
        print(f"{original['text']} → {translated}")

asyncio.run(batch_process())
```

> 🔥 **실무 팁**: `batch` 메서드는 내부적으로 서버의 `/batch` 엔드포인트를 단일 HTTP 요청으로 호출합니다. 따라서 100개의 입력을 `invoke` 100번 호출하는 것보다 네트워크 왕복 횟수가 극적으로 줄어듭니다. 다만 서버 측의 메모리와 처리 능력을 고려하여 적절한 배치 크기를 설정하세요.

### 개념 5: 체인 합성 — 로컬과 원격의 결합

RemoteRunnable의 진정한 힘은 **LangChain의 Runnable 프로토콜을 완전히 구현**한다는 점입니다. 덕분에 파이프 연산자(`|`)로 로컬 컴포넌트와 원격 컴포넌트를 자유롭게 조합할 수 있습니다:

```python
from langserve import RemoteRunnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 서로 다른 서버의 체인 연결
translator = RemoteRunnable("http://server-a:8000/translate/")
summarizer = RemoteRunnable("http://server-b:8000/summarize/")

# 로컬 프롬프트 + 원격 체인 조합
prompt = ChatPromptTemplate.from_template(
    "다음 텍스트를 {language}로 번역해주세요: {text}"
)

# 파이프라인: 로컬 프롬프트 → 원격 번역 → 원격 요약
pipeline = (
    {"text": RunnablePassthrough(), "language": lambda _: "English"}
    | prompt
    | translator
    | {"text": RunnablePassthrough()}
    | summarizer
)

result = pipeline.invoke("LangChain은 LLM 애플리케이션 개발 프레임워크입니다...")
print(result)
```

이렇게 하면 마이크로서비스 아키텍처에서 각 서비스가 담당하는 체인을 하나의 파이프라인으로 연결할 수 있습니다. 각 서비스는 독립적으로 스케일링되면서도, 클라이언트 측에서는 마치 단일 체인처럼 사용할 수 있죠.

### 개념 6: 에러 핸들링과 재시도

> 💡 **비유**: 네트워크 에러 핸들링은 택배 시스템과 비슷합니다. 첫 배달 시도에 부재중이면 재배달을 요청하고(재시도), 주소가 잘못됐으면 반송하며(클라이언트 에러), 물류센터 화재가 나면 다른 센터에서 보내야 합니다(서버 에러). 각 상황에 맞는 대응 전략이 필요하죠.

`RemoteRunnable`은 내부적으로 httpx를 사용하므로, HTTP 에러는 `httpx.HTTPStatusError`로 발생합니다. 서버 측 에러는 응답 본문에 에러 메시지가 포함되어 `_raise_for_status()` 메서드가 이를 파싱하여 더 상세한 에러 정보를 제공합니다.

```python
import httpx
from langserve import RemoteRunnable

chain = RemoteRunnable(
    "http://localhost:8000/my-chain/",
    timeout=10.0,
)

# 기본 에러 핸들링
try:
    result = chain.invoke({"query": "안녕하세요"})
    print(result)
except httpx.ConnectError:
    print("서버에 연결할 수 없습니다. URL과 서버 상태를 확인하세요.")
except httpx.TimeoutException:
    print("요청 시간이 초과되었습니다. timeout 값을 늘려보세요.")
except httpx.HTTPStatusError as e:
    status = e.response.status_code
    if status == 401:
        print("인증 실패: API 키를 확인하세요.")
    elif status == 422:
        print(f"입력 형식 오류: {e.response.text}")
    elif status == 429:
        print("요청 한도 초과: 잠시 후 다시 시도하세요.")
    elif status >= 500:
        print(f"서버 오류 ({status}): {e.response.text}")
```

프로덕션 환경에서는 `tenacity` 라이브러리를 활용하여 자동 재시도 로직을 구현하는 것이 일반적입니다:

```python
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langserve import RemoteRunnable

chain = RemoteRunnable(
    "http://localhost:8000/my-chain/",
    timeout=30.0,
)

@retry(
    stop=stop_after_attempt(3),                    # 최대 3회 시도
    wait=wait_exponential(multiplier=1, max=10),   # 지수 백오프 (1s, 2s, 4s...)
    retry=retry_if_exception_type((                # 재시도할 예외 타입
        httpx.ConnectError,
        httpx.TimeoutException,
    )),
)
def invoke_with_retry(input_data: dict) -> str:
    """재시도 로직이 포함된 체인 호출"""
    return chain.invoke(input_data)

# 사용
try:
    result = invoke_with_retry({"query": "LangChain 소개해줘"})
    print(result)
except Exception as e:
    print(f"3회 재시도 후에도 실패: {e}")
```

## 실습: 직접 해보기

앞서 배운 모든 개념을 종합하여, 여러 LangServe 서버를 조합하고 스트리밍과 에러 핸들링을 갖춘 프로덕션급 클라이언트를 구현해봅시다.

> 이 실습은 [17.1 LangServe 기초](./01-langserve-기초.md)와 [17.3 인증과 보안](./03-인증과-보안.md)에서 배포한 서버가 실행 중이라고 가정합니다.

```python
"""
LangServe RemoteRunnable 클라이언트 종합 실습
- 다양한 호출 패턴 (invoke, stream, batch)
- 에러 핸들링과 재시도
- 원격 체인 합성
"""

import asyncio
import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langserve import RemoteRunnable

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# 1. 클라이언트 초기화
# ──────────────────────────────────────────

# API 키 인증이 적용된 서버에 연결
chat_chain = RemoteRunnable(
    url="http://localhost:8000/chat/",
    timeout=30.0,                              # 30초 타임아웃
    headers={"X-API-Key": "my-secret-key"},    # 인증 헤더
)

translate_chain = RemoteRunnable(
    url="http://localhost:8000/translate/",
    timeout=15.0,
)

logger.info("RemoteRunnable 클라이언트 초기화 완료")

# ──────────────────────────────────────────
# 2. 기본 호출 (invoke)
# ──────────────────────────────────────────

def demo_invoke():
    """동기 invoke 데모"""
    print("=== 동기 invoke ===")
    try:
        result = chat_chain.invoke({"question": "파이썬이란 무엇인가요?"})
        print(f"응답: {result}\n")
    except httpx.HTTPStatusError as e:
        print(f"HTTP 에러 {e.response.status_code}: {e.response.text}")

# ──────────────────────────────────────────
# 3. 스트리밍 응답
# ──────────────────────────────────────────

async def demo_streaming():
    """비동기 스트리밍 데모"""
    print("=== 비동기 스트리밍 ===")
    
    token_count = 0
    async for chunk in chat_chain.astream(
        {"question": "LangChain의 핵심 컴포넌트를 설명해주세요"}
    ):
        print(chunk, end="", flush=True)  # 토큰 단위 실시간 출력
        token_count += 1
    
    print(f"\n[총 {token_count}개 청크 수신]\n")

# ──────────────────────────────────────────
# 4. 배치 처리
# ──────────────────────────────────────────

async def demo_batch():
    """비동기 배치 데모"""
    print("=== 비동기 배치 ===")
    
    questions = [
        {"question": "RAG란?"},
        {"question": "LCEL이란?"},
        {"question": "LangGraph란?"},
    ]
    
    # 여러 질문을 한 번에 처리
    results = await chat_chain.abatch(questions)
    
    for q, r in zip(questions, results):
        print(f"Q: {q['question']}")
        print(f"A: {str(r)[:80]}...\n")

# ──────────────────────────────────────────
# 5. 재시도 로직이 포함된 안전한 호출
# ──────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        httpx.ConnectError,
        httpx.TimeoutException,
    )),
    before_sleep=lambda retry_state: logger.warning(
        f"재시도 {retry_state.attempt_number}/3 — "
        f"{retry_state.outcome.exception().__class__.__name__}"
    ),
)
async def safe_ainvoke(
    chain: RemoteRunnable,
    input_data: dict[str, Any],
) -> Any:
    """재시도와 에러 핸들링이 포함된 안전한 비동기 호출"""
    return await chain.ainvoke(input_data)

async def demo_safe_call():
    """안전한 호출 데모"""
    print("=== 안전한 호출 (재시도 포함) ===")
    
    try:
        result = await safe_ainvoke(
            chat_chain,
            {"question": "에이전트(Agent)를 설명해주세요"},
        )
        print(f"응답: {result}\n")
    except httpx.ConnectError:
        print("3회 재시도 후에도 서버 연결 실패\n")
    except httpx.HTTPStatusError as e:
        print(f"HTTP 에러 (재시도 불가): {e.response.status_code}\n")

# ──────────────────────────────────────────
# 6. 원격 체인 합성
# ──────────────────────────────────────────

async def demo_composition():
    """원격 체인을 로컬 파이프라인에 합성하는 데모"""
    print("=== 원격 체인 합성 ===")
    from langchain_core.runnables import RunnableLambda
    
    # 후처리 단계를 로컬 함수로 정의
    def format_output(text: str) -> str:
        """응답을 포맷팅하는 로컬 함수"""
        return f"📝 번역 결과:\n{text}\n{'─' * 40}"
    
    # 원격 번역 → 로컬 포맷팅 파이프라인
    pipeline = translate_chain | RunnableLambda(format_output)
    
    result = await pipeline.ainvoke({
        "text": "LangServe makes deployment easy.",
        "language": "Korean",
    })
    print(result)

# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────

async def main():
    """모든 데모를 순차 실행"""
    demo_invoke()             # 동기 호출
    await demo_streaming()    # 스트리밍
    await demo_batch()        # 배치
    await demo_safe_call()    # 재시도 포함 호출
    await demo_composition()  # 체인 합성

if __name__ == "__main__":
    asyncio.run(main())
```

## 더 깊이 알아보기

### RemoteRunnable의 탄생 — "투명한 원격 호출"이라는 꿈

RemoteRunnable의 설계 철학은 분산 컴퓨팅의 오래된 꿈에서 비롯됩니다. 1984년 앤드류 버렐(Andrew Birrell)과 브루스 넬슨(Bruce Nelson)이 발표한 RPC(Remote Procedure Call) 논문에서 처음 제안된 "원격 함수를 로컬 함수처럼 호출한다"는 개념이 그 뿌리입니다.

LangChain 팀은 이 철학을 AI 시대에 맞게 재해석했습니다. 2023년 10월 LangServe를 발표하면서, 핵심 목표를 "개발자가 배포와 호출의 복잡성을 의식하지 않도록 하는 것"으로 설정했죠. `RemoteRunnable`이 `Runnable` 인터페이스를 완전히 구현하는 이유도 바로 여기에 있습니다 — `invoke()`, `stream()`, `batch()` 등 로컬에서 쓰던 메서드를 그대로 사용할 수 있으니까요.

내부적으로 `RemoteRunnable`은 httpx 라이브러리를 선택했는데, 이는 requests 대비 네이티브 비동기 지원, HTTP/2 호환, 그리고 스트리밍 응답 처리에 강점이 있기 때문입니다. 동기(`httpx.Client`)와 비동기(`httpx.AsyncClient`) 클라이언트를 동시에 생성하는 이중 구조도 LangChain의 "sync와 async를 모두 지원한다"는 설계 원칙을 반영한 것입니다.

### LangGraph Platform으로의 진화

흥미롭게도, LangChain 팀은 현재 신규 프로젝트에 LangServe 대신 **LangGraph Platform**을 권장하고 있습니다. LangGraph Platform은 상태 기반 에이전트 워크플로우의 배포에 최적화되어 있으며, LangServe의 핵심 패턴(RemoteRunnable 포함)을 흡수하면서 더 강력한 기능을 제공합니다. 하지만 LangServe와 RemoteRunnable의 패턴을 이해하는 것은 LangGraph Platform으로 전환할 때도 큰 도움이 됩니다 — 근본적인 설계 철학이 동일하거든요.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "RemoteRunnable은 단순한 HTTP 클라이언트다"라고 생각하기 쉽지만, 실제로는 LangChain의 `Runnable` 프로토콜을 완전히 구현한 일급 객체입니다. 따라서 파이프 연산자(`|`)로 다른 Runnable과 자유롭게 합성할 수 있고, `RunnableParallel`, `RunnablePassthrough` 등과도 조합됩니다. 단순 HTTP 래퍼가 아닌, LangChain 생태계의 완전한 구성원이죠.

> 💡 **알고 계셨나요?**: RemoteRunnable이 내부에서 사용하는 httpx 라이브러리는 원래 requests의 차세대 대안으로 개발되었습니다. Tom Christie가 Django REST framework를 만든 후, 비동기 지원과 HTTP/2를 기본 제공하는 현대적 HTTP 클라이언트가 필요하다고 느껴 시작한 프로젝트입니다. LangServe가 requests 대신 httpx를 선택한 이유는 동기/비동기 이중 지원이 LangChain의 핵심 설계 원칙과 정확히 일치했기 때문입니다.

> 🔥 **실무 팁**: 프로덕션에서 RemoteRunnable의 `timeout`을 설정하지 않으면 httpx의 기본 타임아웃(5초)이 적용됩니다. LLM 체인은 응답 생성에 10초 이상 걸릴 수 있으므로, **반드시 충분한 타임아웃을 설정**하세요. 스트리밍의 경우 첫 번째 토큰까지의 시간(TTFT)을 고려하여 설정하되, 전체 응답 시간이 아닌 연결 수립 시간 기준으로 판단하는 것이 좋습니다.

> 🔥 **실무 팁**: 배치 크기 결정에 고민이 있다면, 서버의 동시 처리 능력과 메모리를 기준으로 판단하세요. 일반적으로 10~50개가 적절한 배치 크기입니다. 너무 크면 서버 메모리 부족이나 타임아웃이 발생할 수 있고, 너무 작으면 배치의 이점이 줄어듭니다. 큰 데이터셋은 배치를 나누어 순차 처리하는 것이 안전합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `RemoteRunnable` | LangServe 서버의 체인을 로컬 Runnable처럼 호출하는 클라이언트 SDK |
| `invoke` / `ainvoke` | 단일 입력에 대한 동기/비동기 호출 |
| `stream` / `astream` | 토큰 단위 실시간 스트리밍 (동기/비동기) |
| `batch` / `abatch` | 여러 입력을 한 번에 처리하는 배치 호출 |
| `astream_events` | 체인 내부 이벤트를 세밀하게 관찰하는 스트리밍 |
| `timeout` | httpx 요청 타임아웃 (초 단위, 기본값 5초) |
| `headers` | 커스텀 HTTP 헤더 (API 키 인증 등) |
| `client_kwargs` | httpx 클라이언트에 전달할 추가 설정 |
| 체인 합성 | 파이프 연산자(`\|`)로 로컬·원격 컴포넌트 결합 |
| 에러 핸들링 | httpx 예외 + tenacity 재시도로 안정적 호출 |

## 다음 섹션 미리보기

이제 RemoteRunnable로 서버를 자유자재로 호출할 수 있게 되었습니다. 다음 섹션 **[17.5 배포 전략과 모니터링](./05-배포와-운영.md)**에서는 Docker 컨테이너화, 클라우드 배포, 로드 밸런싱, 그리고 LangSmith를 활용한 프로덕션 모니터링까지 — LangServe 애플리케이션을 실제 서비스로 운영하는 전체 과정을 다룹니다.

## 참고 자료

- [LangServe GitHub 리포지토리](https://github.com/langchain-ai/langserve) - RemoteRunnable 소스 코드, 전체 API, 클라이언트/서버 예제를 포함한 공식 리포지토리
- [LangChain 공식 문서 — LangServe](https://python.langchain.com/docs/langserve/) - LangServe 설치, 설정, 배포에 대한 공식 가이드
- [LangServe 클라이언트 예제 노트북](https://github.com/langchain-ai/langserve/blob/main/examples/llm/client.ipynb) - invoke, stream, batch를 실제로 시연하는 공식 Jupyter 노트북
- [httpx 공식 문서 — Timeouts](https://www.python-httpx.org/advanced/timeouts/) - RemoteRunnable이 내부에서 사용하는 httpx의 타임아웃 설정 상세 문서
- [LangServe 에러 핸들링 Discussion](https://github.com/langchain-ai/langserve/discussions/433) - 커뮤니티에서 논의된 예외 처리와 HTTP 응답 패턴

---
### 🔗 Related Sessions
- [langserve](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [add_routes](01-langchain-소개와-개발-환경-설정/05-langchain-생태계-탐색.md) (prerequisite)
- [remote_runnable](./01-langserve-기초.md) (prerequisite)
- [cors_middleware](./02-fastapi-통합과-서버-구성.md) (prerequisite)
- [api_key_auth](./02-fastapi-통합과-서버-구성.md) (prerequisite)
