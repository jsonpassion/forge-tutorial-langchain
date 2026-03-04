# 핵심 Runnable 컴포넌트

> LCEL의 데이터 흐름을 자유자재로 제어하는 RunnablePassthrough, RunnableParallel, RunnableLambda를 마스터합니다.

## 개요

이 섹션에서는 LCEL 체인의 데이터 흐름을 정밀하게 제어하는 핵심 Runnable 컴포넌트들을 학습합니다. [5.1 LCEL 기초와 파이프 연산자](ch05/session_01.md)에서 배운 파이프 연산자(`|`)와 Runnable 프로토콜을 기반으로, 이제 데이터를 통과시키고, 분기하고, 변환하는 방법을 익힙니다.

**선수 지식**: Runnable 프로토콜과 파이프 연산자(`|`)의 기본 동작 원리, `invoke`/`stream`/`batch` 메서드 사용법 ([5.1 LCEL 기초와 파이프 연산자](ch05/session_01.md))

**학습 목표**:
- `RunnablePassthrough`로 입력 데이터를 유지하면서 중간 처리를 수행할 수 있다
- `RunnableParallel`로 여러 작업을 동시에 병렬 실행할 수 있다
- `RunnableLambda`로 일반 Python 함수를 Runnable로 변환하여 체인에 삽입할 수 있다
- `itemgetter`를 활용하여 딕셔너리에서 특정 키를 추출할 수 있다

## 왜 알아야 할까?

실제 LLM 애플리케이션을 만들다 보면, 단순한 "프롬프트 → 모델 → 파서" 직선 체인만으로는 부족한 순간이 금방 찾아옵니다. RAG 시스템을 예로 들어볼까요? 사용자 질문을 받으면, 한쪽에서는 벡터 DB를 검색하고, 다른 한쪽에서는 원래 질문을 그대로 유지해서, 두 결과를 합쳐 프롬프트에 넣어야 합니다. 이런 "Y자 분기", "데이터 통과", "중간 변환" 같은 패턴이 없으면 체인이 금세 스파게티 코드가 되어버리거든요.

이번 섹션에서 배울 세 가지 Runnable — `RunnablePassthrough`, `RunnableParallel`, `RunnableLambda` — 은 바로 이 문제를 해결하는 LCEL의 핵심 도구입니다. 이 컴포넌트들을 익히면 어떤 복잡한 데이터 흐름도 선언적으로, 깔끔하게 표현할 수 있습니다.

## 핵심 개념

### 개념 1: RunnablePassthrough — 데이터를 그대로 통과시키기

> 💡 **비유**: 공항의 환승 게이트를 떠올려보세요. 승객(데이터)이 아무런 검사 없이 그대로 통과하는 게이트가 있고, 통과하면서 면세품(추가 정보)을 붙여주는 게이트가 있죠. `RunnablePassthrough`가 바로 이 환승 게이트입니다. 기본적으로는 입력을 그대로 내보내고, `.assign()`을 쓰면 추가 정보를 붙여서 내보냅니다.

`RunnablePassthrough`는 입력을 아무 변환 없이 그대로 다음 단계로 전달합니다. "아무것도 안 하는 게 왜 유용하지?"라고 생각할 수 있는데, 이것이 빛을 발하는 건 **병렬 실행과 조합**할 때입니다.

```python
from langchain_core.runnables import RunnablePassthrough

# 기본 사용: 입력을 그대로 반환
passthrough = RunnablePassthrough()
result = passthrough.invoke("안녕하세요")
print(result)  # 출력: 안녕하세요
```

#### RunnablePassthrough.assign() — 통과시키면서 필드 추가하기

진짜 위력은 `.assign()` 메서드에 있습니다. 원래 입력은 그대로 유지하면서, 새로운 키-값 쌍을 추가할 수 있거든요.

```python
from langchain_core.runnables import RunnablePassthrough

# assign으로 새 필드 추가
chain = RunnablePassthrough.assign(
    # 원래 입력(dict)은 그대로 유지하면서 'length' 키를 추가
    length=lambda x: len(x["text"])
)

result = chain.invoke({"text": "LangChain은 강력합니다"})
print(result)
# 출력: {'text': 'LangChain은 강력합니다', 'length': 13}
```

핵심은 이겁니다 — `assign()`은 **원래 딕셔너리의 모든 키를 보존**하면서 새 키를 추가합니다. 데이터를 잃어버리지 않고 점진적으로 정보를 쌓아가는 패턴이죠.

### 개념 2: RunnableParallel — 여러 작업을 동시에 실행하기

> 💡 **비유**: 식당 주방을 상상해보세요. 주문이 들어오면 한 셰프는 스테이크를 굽고, 다른 셰프는 샐러드를 만들고, 또 다른 셰프는 수프를 끓입니다. 각각 독립적으로 작업하다가, 다 끝나면 하나의 접시에 담아 내보냅니다. `RunnableParallel`이 바로 이 "동시 조리" 시스템입니다.

`RunnableParallel`은 **같은 입력**을 여러 Runnable에 동시에 전달하고, 각각의 결과를 딕셔너리로 모아줍니다.

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 세 가지 작업을 병렬로 실행
chain = RunnableParallel(
    upper=RunnableLambda(lambda x: x.upper()),        # 대문자 변환
    length=RunnableLambda(lambda x: len(x)),           # 길이 계산
    reversed=RunnableLambda(lambda x: x[::-1])         # 문자열 뒤집기
)

result = chain.invoke("hello")
print(result)
# 출력: {'upper': 'HELLO', 'length': 5, 'reversed': 'olleh'}
```

#### 딕셔너리 리터럴 축약 문법

LangChain은 `RunnableParallel`을 더 간결하게 쓸 수 있는 문법을 제공합니다. 파이프 연산자 안에서 딕셔너리를 쓰면 자동으로 `RunnableParallel`로 변환되거든요.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "다음 맥락을 참고하여 질문에 답하세요.\n맥락: {context}\n질문: {question}"
)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 딕셔너리 리터럴 → 자동으로 RunnableParallel 변환
chain = (
    {"context": lambda x: "LangChain은 LLM 프레임워크입니다",
     "question": RunnablePassthrough()}
    | prompt
    | model
)

# "question" 키에 입력이 그대로 들어가고, "context"에는 고정값이 들어감
response = chain.invoke("LangChain이 뭔가요?")
print(response.content)
```

이 패턴이 왜 중요하냐면, RAG 체인에서 가장 많이 쓰이는 패턴이기 때문입니다. `context`에는 검색 결과를, `question`에는 원래 질문을 넣는 거죠.

### 개념 3: RunnableLambda — 일반 함수를 Runnable로 변환하기

> 💡 **비유**: USB 어댑터를 생각해보세요. 여러분이 가진 다양한 충전기(일반 함수)를 USB-C 포트(Runnable 인터페이스)에 꽂을 수 있게 해주는 어댑터가 `RunnableLambda`입니다. 어떤 Python 함수든 LCEL 체인에 끼워 넣을 수 있게 변환해줍니다.

`RunnableLambda`는 일반 Python 함수를 Runnable 프로토콜을 따르는 객체로 감싸줍니다. 덕분에 `invoke`, `batch`, `stream` 같은 통합 인터페이스를 사용할 수 있게 되죠.

```python
from langchain_core.runnables import RunnableLambda

# 일반 함수 정의
def add_greeting(text: str) -> str:
    """입력 텍스트 앞에 인사를 추가합니다."""
    return f"안녕하세요! {text}"

def count_words(text: str) -> dict:
    """단어 수를 세서 딕셔너리로 반환합니다."""
    words = text.split()
    return {"text": text, "word_count": len(words)}

# RunnableLambda로 감싸기
greet = RunnableLambda(add_greeting)
counter = RunnableLambda(count_words)

# 체인 구성
chain = greet | counter
result = chain.invoke("LangChain을 배우고 있습니다")
print(result)
# 출력: {'text': '안녕하세요! LangChain을 배우고 있습니다', 'word_count': 4}
```

#### 비동기 함수도 지원

`RunnableLambda`는 비동기 함수도 자연스럽게 지원합니다. `afunc` 매개변수로 비동기 버전을 별도 지정할 수도 있어요.

```python
import asyncio
from langchain_core.runnables import RunnableLambda

# 동기 함수와 비동기 함수를 함께 지정
def sync_process(x: str) -> str:
    return x.upper()

async def async_process(x: str) -> str:
    # 비동기 I/O 작업 시뮬레이션
    await asyncio.sleep(0.1)
    return x.upper()

# 동기/비동기 함수를 동시에 등록
runnable = RunnableLambda(func=sync_process, afunc=async_process)

# 동기 호출
print(runnable.invoke("hello"))  # 출력: HELLO

# 비동기 호출
result = asyncio.run(runnable.ainvoke("hello"))
print(result)  # 출력: HELLO
```

### 개념 4: itemgetter — 딕셔너리에서 키를 깔끔하게 추출하기

> 💡 **비유**: 우편 분류함을 떠올려보세요. 편지(딕셔너리)가 도착하면, 분류함(itemgetter)이 특정 칸(키)에 있는 내용물만 꺼내줍니다. `lambda x: x["question"]`과 같은 역할이지만, 훨씬 깔끔하고 직관적이죠.

Python 표준 라이브러리의 `operator.itemgetter`는 LCEL에서 딕셔너리의 특정 키 값을 추출할 때 매우 유용합니다.

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# itemgetter 기본 사용법
data = {"name": "LangChain", "version": "0.3", "type": "framework"}

# lambda 방식 vs itemgetter 방식
get_name_lambda = lambda x: x["name"]          # lambda 방식
get_name_getter = itemgetter("name")            # itemgetter 방식

print(get_name_lambda(data))  # 출력: LangChain
print(get_name_getter(data))  # 출력: LangChain

# 여러 키를 한 번에 추출
get_name_and_version = itemgetter("name", "version")
print(get_name_and_version(data))  # 출력: ('LangChain', '0.3')
```

#### LCEL 체인에서 itemgetter 활용

```python
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "{language}로 '{concept}' 개념을 설명해주세요."
)
model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# itemgetter로 딕셔너리에서 필요한 키만 추출하여 프롬프트에 전달
chain = (
    {
        "language": itemgetter("language"),
        "concept": itemgetter("concept")
    }
    | prompt
    | model
    | StrOutputParser()
)

result = chain.invoke({
    "language": "한국어",
    "concept": "Runnable",
    "extra_data": "이 값은 무시됩니다"  # 필요 없는 키는 자동으로 걸러짐
})
print(result)
```

`itemgetter`의 장점은 **불필요한 키를 자연스럽게 필터링**한다는 겁니다. 입력 딕셔너리에 많은 키가 있어도, 필요한 것만 골라서 다음 단계로 넘길 수 있죠.

### 개념 5: 컴포넌트 조합 — 모든 것을 연결하기

이제 네 가지 도구를 모두 조합해봅시다. 실전에서 가장 많이 보이는 패턴인 RAG 스타일 체인의 데이터 흐름을 구성합니다.

```python
from operator import itemgetter
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. 가짜 검색 함수 (실제로는 벡터 검색이 들어갈 자리)
def fake_retriever(query: str) -> str:
    """쿼리에 기반한 가짜 문서 검색 시뮬레이션"""
    docs = {
        "LangChain": "LangChain은 LLM 기반 앱 개발 프레임워크입니다.",
        "LCEL": "LCEL은 파이프 연산자로 체인을 구성하는 선언적 언어입니다.",
    }
    # 간단한 키워드 매칭
    for key, doc in docs.items():
        if key.lower() in query.lower():
            return doc
    return "관련 문서를 찾을 수 없습니다."

# 2. 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template(
    "맥락: {context}\n\n질문: {question}\n\n위 맥락을 참고하여 질문에 답하세요."
)

model = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. 전체 체인 조합
chain = (
    # RunnableParallel: 검색과 질문 보존을 병렬로 실행
    RunnableParallel(
        context=RunnableLambda(fake_retriever),  # 질문 → 검색 결과
        question=RunnablePassthrough()            # 질문 → 그대로 유지
    )
    | prompt          # 딕셔너리 → 프롬프트
    | model           # 프롬프트 → LLM 응답
    | StrOutputParser()  # 응답 → 문자열
)

# 실행
answer = chain.invoke("LangChain이 뭔가요?")
print(answer)
```

데이터 흐름을 정리하면 이렇습니다:

| 단계 | 입력 | 처리 | 출력 |
|------|------|------|------|
| 1. RunnableParallel | `"LangChain이 뭔가요?"` | 병렬 분기 | `{"context": "LangChain은...", "question": "LangChain이 뭔가요?"}` |
| 2. prompt | `dict` | 템플릿 렌더링 | `ChatPromptValue` |
| 3. model | `ChatPromptValue` | LLM 호출 | `AIMessage` |
| 4. StrOutputParser | `AIMessage` | 문자열 추출 | `str` |

## 실습: 직접 해보기

이번 실습에서는 여러 관점에서 동시에 분석하는 "다면 분석 체인"을 만들어봅니다. 하나의 텍스트를 입력받아 감정 분석, 키워드 추출, 요약을 동시에 수행하는 체인입니다.

```python
"""
다면 분석 체인 — RunnableParallel + RunnableLambda + RunnablePassthrough 통합 실습
"""
from operator import itemgetter
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일에 OPENAI_API_KEY 필요)
load_dotenv()

# ── 모델 설정 ──
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.3

model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
parser = StrOutputParser()

# ── 전처리 함수: RunnableLambda로 감쌀 예정 ──
def preprocess(text: str) -> str:
    """텍스트 전처리: 공백 정리 및 길이 제한"""
    cleaned = " ".join(text.split())  # 연속 공백 제거
    return cleaned[:2000]  # 최대 2000자로 제한

# ── 분석별 프롬프트 템플릿 ──
sentiment_prompt = ChatPromptTemplate.from_template(
    "다음 텍스트의 감정을 분석하세요. 긍정/부정/중립 중 하나와 이유를 한 줄로 답하세요.\n\n텍스트: {text}"
)

keyword_prompt = ChatPromptTemplate.from_template(
    "다음 텍스트에서 핵심 키워드 5개를 쉼표로 구분하여 추출하세요.\n\n텍스트: {text}"
)

summary_prompt = ChatPromptTemplate.from_template(
    "다음 텍스트를 한 문장으로 요약하세요.\n\n텍스트: {text}"
)

# ── 개별 분석 체인 ──
sentiment_chain = sentiment_prompt | model | parser
keyword_chain = keyword_prompt | model | parser
summary_chain = summary_prompt | model | parser

# ── 전체 체인 조합 ──
# 1단계: 전처리 (RunnableLambda)
# 2단계: text 키로 감싸기 (RunnablePassthrough.assign 활용)
# 3단계: 세 가지 분석을 동시에 실행 (RunnableParallel)
analysis_chain = (
    # 일반 함수 → Runnable 변환
    RunnableLambda(preprocess)
    # 전처리된 텍스트를 딕셔너리로 감싸기
    | RunnableLambda(lambda text: {"text": text})
    # 세 분석을 병렬 실행
    | RunnableParallel(
        sentiment=sentiment_chain,    # 감정 분석
        keywords=keyword_chain,       # 키워드 추출
        summary=summary_chain,        # 요약
    )
)

# ── 실행 ──
sample_text = """
LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한
오픈소스 프레임워크입니다. 2022년 Harrison Chase에 의해 만들어졌으며,
모듈식 설계 덕분에 개발자들이 복잡한 AI 파이프라인을 쉽게 구축할 수 있습니다.
특히 LCEL의 도입으로 선언적 체인 구성이 가능해져, 코드 가독성과
유지보수성이 크게 향상되었습니다.
"""

result = analysis_chain.invoke(sample_text)

# 결과 출력
print("=" * 60)
print("📊 다면 분석 결과")
print("=" * 60)
print(f"\n🎭 감정 분석:\n  {result['sentiment']}")
print(f"\n🔑 키워드:\n  {result['keywords']}")
print(f"\n📝 요약:\n  {result['summary']}")
print("=" * 60)

# ── 배치 실행도 가능! ──
texts = [
    "이 프레임워크 정말 편리하고 직관적이에요!",
    "문서가 부족하고 버전 호환성 문제가 너무 많습니다.",
    "LangChain 0.3 버전이 출시되었습니다.",
]

batch_results = analysis_chain.batch(texts)
for i, (text, res) in enumerate(zip(texts, batch_results), 1):
    print(f"\n--- 텍스트 {i}: {text[:30]}... ---")
    print(f"  감정: {res['sentiment']}")
    print(f"  키워드: {res['keywords']}")
```

## 더 깊이 알아보기

### LCEL은 왜 이런 모양이 됐을까? — Unix 파이프의 영향

LCEL의 파이프 연산자(`|`)와 Runnable 컴포넌트들의 설계는 사실 1973년 Ken Thompson이 Unix에 도입한 파이프(pipe) 개념에서 깊은 영향을 받았습니다. Unix의 철학 — "한 가지 일을 잘 하는 작은 프로그램을 만들고, 파이프로 연결하라" — 이 LCEL에 고스란히 녹아 있는 거죠.

Harrison Chase가 2022년 10월 LangChain을 처음 공개했을 때, 체인 구성은 `LLMChain`, `SequentialChain` 같은 클래스 기반이었습니다. 하지만 이 방식은 복잡한 데이터 흐름을 표현하기 어려웠고, 새로운 패턴이 등장할 때마다 새 클래스를 만들어야 했죠. 2023년 3분기에 도입된 LCEL은 이 문제를 해결하기 위해, 모든 컴포넌트를 `Runnable`이라는 하나의 프로토콜로 통일하고, 파이프 연산자로 조합하는 선언적 접근법을 택했습니다.

특히 `RunnableParallel`의 "딕셔너리 리터럴 축약" 문법은 개발자 경험을 극적으로 개선한 설계 결정이었는데요. `RunnableParallel(context=..., question=...)`을 `{"context": ..., "question": ...}`으로 줄일 수 있게 하면서, 체인 코드가 마치 데이터의 구조를 선언하는 것처럼 읽히게 만들었습니다. 이는 React의 JSX가 HTML처럼 보이게 하여 UI 구조를 직관적으로 만든 것과 비슷한 발상이라고 할 수 있습니다.

### RunnableLambda — "어댑터 패턴"의 현대적 구현

소프트웨어 공학의 고전적 디자인 패턴 중 **어댑터 패턴(Adapter Pattern)** 이 있습니다. 1994년 "Gang of Four"의 『Design Patterns』에서 소개된 이 패턴은 "호환되지 않는 인터페이스를 연결해주는 중간 계층"을 만드는 것인데, `RunnableLambda`가 정확히 이 역할을 합니다. 어떤 Python 함수든 Runnable 프로토콜에 맞게 "어댑팅"해주는 거죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: `RunnablePassthrough`와 `RunnablePassthrough.assign()`은 같다?
> 아닙니다! `RunnablePassthrough()`는 입력을 **그대로** 반환합니다. 반면 `RunnablePassthrough.assign(key=fn)`은 원래 입력(반드시 딕셔너리)에 **새 키를 추가**하여 반환합니다. `.assign()`을 쓸 때는 입력이 딕셔너리여야 한다는 점을 꼭 기억하세요. 문자열을 넣으면 에러가 납니다.

> 💡 **알고 계셨나요?**: LCEL에서 딕셔너리 리터럴(`{...}`)을 파이프 체인 안에 쓰면, LangChain이 내부적으로 `RunnableParallel`로 자동 변환합니다. 즉, `{"a": chain_a, "b": chain_b}`는 `RunnableParallel(a=chain_a, b=chain_b)`와 **완전히 동일**합니다. 공식 문서에서도 딕셔너리 축약 문법을 더 권장하는데, 코드가 더 짧고 직관적이기 때문이죠.

> 🔥 **실무 팁**: `RunnableLambda`에 함수를 넘길 때, 함수의 **인자는 반드시 하나**여야 합니다. 여러 값을 받아야 한다면 딕셔너리로 묶어서 전달하고, 함수 안에서 언패킹하세요:
> ```python
> # ❌ 잘못된 방법
> def process(a, b): return a + b
>
> # ✅ 올바른 방법
> def process(inputs: dict) -> str:
>     return inputs["a"] + inputs["b"]
> ```

> 🔥 **실무 팁**: `itemgetter`는 `lambda`보다 성능이 미세하게 빠르고, 무엇보다 **의도가 명확하게 드러납니다**. `lambda x: x["question"]` 대신 `itemgetter("question")`을 쓰면 "이건 딕셔너리에서 question 키를 꺼내는 거구나"라고 바로 알 수 있죠. 또한 `itemgetter`는 `pickle` 직렬화가 가능하지만, `lambda`는 불가능한 경우가 많다는 실무적 차이도 있습니다.

## 핵심 정리

| 컴포넌트 | 역할 | 핵심 포인트 |
|----------|------|-------------|
| `RunnablePassthrough` | 입력을 그대로 통과 | 병렬 체인에서 원본 데이터 보존에 사용 |
| `RunnablePassthrough.assign()` | 입력 보존 + 새 키 추가 | 딕셔너리 입력 필수, 점진적 정보 축적 패턴 |
| `RunnableParallel` | 여러 Runnable을 동시 실행 | 같은 입력 → 여러 결과를 딕셔너리로 수집 |
| 딕셔너리 리터럴 `{...}` | `RunnableParallel` 축약 문법 | 체인 안에서 딕셔너리 쓰면 자동 변환 |
| `RunnableLambda` | 일반 함수를 Runnable로 변환 | 인자는 반드시 1개, 비동기 함수도 지원 |
| `itemgetter` | 딕셔너리 키 추출 | `lambda`보다 명시적이고 직렬화 가능 |

## 다음 섹션 미리보기

이번 섹션에서 배운 핵심 Runnable 컴포넌트로 데이터 흐름을 제어하는 방법을 익혔습니다. 하지만 실전에서는 "조건에 따라 다른 체인을 실행해야 하는" 상황이 반드시 등장합니다. 다음 섹션에서는 **조건 분기와 동적 라우팅** — `RunnableBranch`와 커스텀 라우팅 로직을 사용하여 입력에 따라 체인이 다르게 동작하는 패턴을 학습합니다.

## 참고 자료

- [LangChain LCEL 공식 개념 문서](https://python.langchain.com/docs/concepts/lcel/) - LCEL의 핵심 개념과 설계 철학을 공식 문서에서 확인할 수 있습니다
- [RunnablePassthrough API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html) - `RunnablePassthrough`와 `.assign()` 메서드의 상세 API 문서
- [langchain_core.runnables API Reference](https://python.langchain.com/api_reference/core/runnables.html) - 모든 Runnable 컴포넌트의 공식 API 레퍼런스
- [LangChain Expression Language Explained - Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/) - LCEL의 실전 활용법을 풍부한 예제로 설명하는 튜토리얼
- [LangChain Runnables 개요 - 공식 문서](https://python.langchain.com/docs/concepts/runnables/) - Runnable 프로토콜의 전체 그림을 이해할 수 있는 공식 가이드

---
### 🔗 Related Sessions
- [lcel_pipe_operator](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [runnable_protocol](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [runnable_sequence](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [invoke_method](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [batch_method](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
