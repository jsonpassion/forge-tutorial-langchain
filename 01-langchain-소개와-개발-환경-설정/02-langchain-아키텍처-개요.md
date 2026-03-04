# LangChain 아키텍처 개요

> LangChain은 하나의 거대한 덩어리가 아닙니다. 정교하게 분리된 패키지 구조와 통합 인터페이스가 그 힘의 비밀입니다.

## 개요

이 섹션에서는 LangChain의 내부 아키텍처를 해부합니다. 앞서 [LLM 애플리케이션의 진화와 LangChain](01-llm-애플리케이션의-진화와-langchain.md)에서 LangChain이 "LLM 애플리케이션의 레고 시스템"이라고 했는데요, 이번에는 그 레고 블록이 실제로 **어떤 상자에 담겨 있고**, **어떤 규격으로 만들어져 있으며**, **어떻게 끼워 맞추는지** 구체적으로 들여다봅니다.

**선수 지식**: Session 1.1에서 배운 LangChain의 핵심 철학(통합 인터페이스, 모듈식 구성), 파이프 연산자(`|`) 기본 개념
**학습 목표**:
- `langchain-core`, `langchain-community`, `langchain`, 통합 패키지의 역할과 관계를 설명할 수 있다
- Runnable 인터페이스의 핵심 메서드(`invoke`, `stream`, `batch`)를 이해한다
- LCEL(LangChain Expression Language)의 파이프 연산자가 내부적으로 어떻게 동작하는지 설명할 수 있다
- 필요한 패키지만 골라서 설치하는 최소 의존성 전략을 세울 수 있다

## 왜 알아야 할까?

LangChain을 처음 접하는 개발자들이 가장 많이 겪는 혼란이 바로 이것입니다: **"대체 뭘 설치해야 하는 거지?"**

`pip install langchain`을 하면 되는 건지, `langchain-core`가 따로 있는 건지, `langchain-openai`는 또 뭔지... 패키지 이름만 봐도 머리가 아파지죠. 실제로 LangChain 관련 Stack Overflow 질문 중 상당수가 "어떤 패키지에서 임포트해야 하나요?"입니다.

이 혼란을 겪는 이유는, LangChain이 2024년 초에 **대대적인 아키텍처 분리**를 단행했기 때문입니다. 과거에는 하나의 거대한 `langchain` 패키지에 모든 것이 들어 있었지만, 지금은 목적별로 명확하게 분리되어 있거든요. 이 구조를 이해하면 "어떤 패키지에서 뭘 가져와야 하는지"가 직관적으로 보이게 됩니다.

더 중요한 것은, 이 아키텍처를 이해해야 **Runnable 인터페이스**와 **LCEL**이 왜 그렇게 설계되었는지 납득이 간다는 점입니다. LangChain의 모든 컴포넌트가 같은 규격(Runnable)으로 만들어져 있기 때문에 파이프 연산자로 자유롭게 조합할 수 있는 건데, 이 원리를 모르면 LCEL이 그저 "신기한 문법"으로만 보이게 됩니다.

## 핵심 개념

### 개념 1: 패키지 아키텍처 — 층층이 쌓인 건축물

> 💡 **비유**: LangChain의 패키지 구조는 **아파트 건물**과 비슷합니다. `langchain-core`가 철근 콘크리트 **골조**(기초 구조)이고, `langchain-community`는 여러 업체가 시공한 **인테리어**(서드파티 통합)이며, `langchain-openai` 같은 통합 패키지는 **프리미엄 브랜드 가구**(공식 파트너 통합)입니다. 그리고 `langchain`은 이 모든 것을 연결하는 **배관과 전기 시스템**(체인, 에이전트 등 고수준 기능)이죠.

LangChain 생태계는 크게 네 개의 계층으로 나뉩니다.

**1단계: `langchain-core` — 기초 골조**

모든 것의 뿌리입니다. 추상 클래스, 인터페이스, 그리고 LCEL이 여기에 정의되어 있습니다.

```python
# langchain-core에서 가져오는 것들
from langchain_core.prompts import ChatPromptTemplate       # 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser    # 출력 파서
from langchain_core.runnables import RunnablePassthrough     # Runnable 유틸리티
from langchain_core.documents import Document                # 문서 객체
from langchain_core.messages import HumanMessage, AIMessage  # 메시지 타입
```

`langchain-core`의 특징은 **의존성이 극도로 가볍다**는 것입니다. 외부 API나 무거운 라이브러리에 의존하지 않으므로, 어떤 프로젝트에서든 부담 없이 사용할 수 있습니다.

**2단계: 통합 패키지(Partner Packages) — 프리미엄 가구**

OpenAI, Anthropic, Google 등 주요 LLM 제공자별로 **별도의 패키지**가 존재합니다.

```python
# 각 LLM 제공자별 전용 패키지
from langchain_openai import ChatOpenAI          # pip install langchain-openai
from langchain_anthropic import ChatAnthropic    # pip install langchain-anthropic
from langchain_google_genai import ChatGoogleGenerativeAI  # pip install langchain-google-genai
```

이 패키지들은 각 제공자와 **공식적으로 관리**됩니다. 버전 관리와 품질 보증이 독립적으로 이루어지죠.

**3단계: `langchain-community` — 커뮤니티 인테리어**

수백 개의 서드파티 통합이 모여 있는 곳입니다. 문서 로더, 벡터 스토어, 도구 등이 이 패키지에 포함됩니다.

```python
# langchain-community에서 가져오는 것들
from langchain_community.document_loaders import PyPDFLoader      # PDF 로더
from langchain_community.vectorstores import FAISS                # FAISS 벡터 스토어
from langchain_community.tools import WikipediaQueryRun           # Wikipedia 도구
```

**4단계: `langchain` — 배관과 전기 시스템**

체인(Chain), 에이전트(Agent), 메모리(Memory) 등 **고수준 구성 요소**를 제공합니다. `langchain-core`의 기초 위에 실용적인 기능을 얹은 계층이죠.

```python
# langchain에서 가져오는 것들 (고수준 기능)
from langchain.chains import create_retrieval_chain    # 검색 체인
from langchain.agents import create_tool_calling_agent # 에이전트 생성
```

이 네 계층의 관계를 표로 정리하면 이렇습니다:

| 패키지 | 역할 | 의존성 | 예시 |
|--------|------|--------|------|
| `langchain-core` | 기초 추상화, LCEL, Runnable | 최소 | 프롬프트, 파서, 메시지 |
| `langchain-openai` 등 | 특정 LLM 제공자 통합 | core + 제공자 SDK | ChatOpenAI, OpenAIEmbeddings |
| `langchain-community` | 서드파티 통합 모음 | core + 각 라이브러리 | FAISS, PyPDFLoader |
| `langchain` | 고수준 체인, 에이전트 | core + community | create_retrieval_chain |

> ⚠️ **흔한 오해**: "`pip install langchain` 하나면 다 되는 거 아닌가요?" — 아닙니다! `langchain`을 설치하면 `langchain-core`는 자동으로 딸려오지만, `langchain-openai`나 `langchain-community`는 **별도 설치**가 필요합니다. 필요한 것만 설치하는 것이 패키지 충돌을 줄이고 프로젝트를 가볍게 유지하는 비결입니다.

### 개념 2: Runnable 인터페이스 — 모든 블록의 공통 규격

> 💡 **비유**: USB-C 포트를 떠올려보세요. 충전기, 이어폰, 외장 하드, 모니터... 모양과 용도는 전혀 다르지만, 모두 같은 USB-C 포트로 연결됩니다. LangChain의 **Runnable 인터페이스**가 바로 이 USB-C입니다. 프롬프트 템플릿이든, LLM이든, 출력 파서든, 모두 같은 인터페이스를 구현하기 때문에 서로 자유롭게 연결할 수 있죠.

Runnable은 `langchain-core`에 정의된 **핵심 추상 클래스**로, LangChain의 모든 컴포넌트가 이 인터페이스를 구현합니다. Runnable이 제공하는 핵심 메서드는 세 가지입니다:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# 1. invoke — 단일 입력 → 단일 출력
result = llm.invoke("LangChain이 뭔가요?")
print(result.content)
# 출력: "LangChain은 LLM 기반 애플리케이션 개발을 위한 프레임워크입니다..."

# 2. stream — 단일 입력 → 토큰 단위 스트리밍 출력
for chunk in llm.stream("Python의 장점 3가지"):
    print(chunk.content, end="", flush=True)
# 출력: (한 글자씩 실시간으로 출력됩니다)

# 3. batch — 여러 입력 → 여러 출력 (병렬 처리)
results = llm.batch([
    "Python이란?",
    "JavaScript란?",
    "Rust란?"
])
for r in results:
    print(r.content[:50])  # 각 결과의 앞 50자 출력
```

그리고 이 세 가지 각각에 대응하는 **비동기 버전**이 있습니다:

| 동기 메서드 | 비동기 메서드 | 용도 |
|------------|-------------|------|
| `invoke()` | `ainvoke()` | 단일 입력/출력 |
| `stream()` | `astream()` | 스트리밍 출력 |
| `batch()` | `abatch()` | 병렬 일괄 처리 |

여기서 핵심은, **프롬프트 템플릿도, LLM도, 출력 파서도, 심지어 사용자 정의 함수도** 모두 Runnable이라는 점입니다:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 이 세 가지 모두 Runnable입니다!
prompt = ChatPromptTemplate.from_template("{topic}에 대해 설명해주세요.")
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# 각각 독립적으로 invoke 가능
formatted = prompt.invoke({"topic": "인공지능"})
# → ChatPromptValue(messages=[HumanMessage(content='인공지능에 대해 설명해주세요.')])

response = model.invoke(formatted)
# → AIMessage(content='인공지능은...')

text = parser.invoke(response)
# → '인공지능은...' (순수 문자열)
```

모든 컴포넌트가 Runnable이니까, 하나의 출력이 다음의 입력으로 자연스럽게 흘러갈 수 있는 거죠. 이것이 바로 LCEL의 토대입니다.

### 개념 3: LCEL — 파이프 연산자의 마법

> 💡 **비유**: 공장의 **컨베이어 벨트**를 상상해보세요. 원재료(사용자 입력)가 벨트에 올라가면, 첫 번째 공정(프롬프트 포맷팅)을 거치고, 두 번째 공정(LLM 처리)을 통과하고, 세 번째 공정(출력 파싱)에서 완제품이 나옵니다. LCEL의 파이프 연산자(`|`)는 이 컨베이어 벨트의 **연결 고리**입니다.

LCEL(LangChain Expression Language)은 Runnable 컴포넌트를 **파이프 연산자**(`|`)로 연결하는 선언적 구문입니다. Session 1.1에서 이미 간단히 보았는데, 이번에는 내부 동작을 이해해봅시다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LCEL 체인 구성
chain = ChatPromptTemplate.from_template("{topic}을 한 문장으로 요약해주세요.") \
        | ChatOpenAI(model="gpt-4o") \
        | StrOutputParser()

# 이 한 줄이 내부적으로 하는 일:
# 1. {"topic": "양자역학"} → ChatPromptTemplate → 포맷된 프롬프트
# 2. 포맷된 프롬프트 → ChatOpenAI → AIMessage
# 3. AIMessage → StrOutputParser → 순수 문자열
result = chain.invoke({"topic": "양자역학"})
print(result)
# 출력: "양자역학은 원자 이하 입자들의 행동을 확률적으로 설명하는 물리학 이론입니다."
```

**파이프 연산자의 비밀: `__or__` 메서드**

파이프 연산자(`|`)가 마법처럼 보이지만, 사실 Python의 **연산자 오버로딩**을 활용한 것입니다. `Runnable` 클래스에서 `__or__` 메서드를 오버라이드하여, `|` 연산이 `RunnableSequence`를 반환하도록 구현했습니다.

```python
# 내부적으로 이런 일이 일어납니다
# prompt | model | parser
# ↕ (동일한 동작)
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence(first=prompt, middle=[model], last=parser)
```

`RunnableSequence` 자체도 Runnable이므로, `invoke`, `stream`, `batch`를 모두 사용할 수 있습니다. 체인을 만들었을 뿐인데 스트리밍과 비동기 처리가 **자동으로** 지원되는 거죠:

```python
# 체인에서도 invoke, stream, batch 모두 사용 가능!
chain = prompt | model | parser

# 동기 호출
result = chain.invoke({"topic": "머신러닝"})

# 스트리밍 — 토큰이 생성될 때마다 실시간 출력
for chunk in chain.stream({"topic": "머신러닝"}):
    print(chunk, end="", flush=True)

# 배치 처리 — 여러 입력을 병렬로 처리
results = chain.batch([
    {"topic": "딥러닝"},
    {"topic": "강화학습"},
    {"topic": "자연어처리"}
])
```

### 개념 4: Runnable 유틸리티 — 체인의 빌딩 블록

> 💡 **비유**: 컨베이어 벨트에는 제품을 가공하는 기계만 있는 게 아닙니다. **분기 장치**(여러 라인으로 나누기), **합류 장치**(여러 결과를 합치기), **통과 장치**(그대로 전달하기) 같은 보조 장비도 필요하죠. LangChain의 Runnable 유틸리티가 바로 이런 역할을 합니다.

LCEL 체인을 구성할 때 자주 사용하는 세 가지 핵심 유틸리티가 있습니다.

**`RunnablePassthrough` — 입력을 그대로 통과**

입력을 변환 없이 다음 단계로 전달합니다. "왜 아무것도 안 하는 게 필요하지?"라고 생각할 수 있는데, 병렬 실행에서 원본 입력을 보존할 때 필수적입니다.

```python
from langchain_core.runnables import RunnablePassthrough

# 입력을 그대로 통과시킵니다
passthrough = RunnablePassthrough()
result = passthrough.invoke("안녕하세요")
print(result)  # "안녕하세요"
```

**`RunnableParallel` — 여러 작업을 동시에**

하나의 입력을 받아 여러 Runnable을 **병렬로** 실행하고, 결과를 딕셔너리로 묶어줍니다.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# 같은 주제에 대해 요약과 키워드를 동시에 생성
summary_chain = (
    ChatPromptTemplate.from_template("{topic}을 한 문장으로 요약해주세요.")
    | model | parser
)

keyword_chain = (
    ChatPromptTemplate.from_template("{topic}의 핵심 키워드 3개를 쉼표로 나열해주세요.")
    | model | parser
)

# RunnableParallel로 병렬 실행
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    original=RunnablePassthrough()  # 원본 입력도 보존
)

result = parallel_chain.invoke({"topic": "트랜스포머 아키텍처"})
print(result["summary"])    # "트랜스포머는 셀프 어텐션 메커니즘을 기반으로..."
print(result["keywords"])   # "어텐션, 인코더-디코더, 병렬처리"
print(result["original"])   # {"topic": "트랜스포머 아키텍처"}
```

**`RunnableLambda` — 일반 함수를 Runnable로**

Python 함수나 람다를 Runnable로 감싸서 LCEL 체인에 끼워 넣을 수 있습니다.

```python
from langchain_core.runnables import RunnableLambda

# 일반 함수를 Runnable로 변환
def add_emoji(text: str) -> str:
    """텍스트 앞에 이모지를 붙이는 함수"""
    return f"🤖 {text}"

emoji_runnable = RunnableLambda(add_emoji)

# 체인에 자연스럽게 끼워 넣기
chain = prompt | model | parser | emoji_runnable
result = chain.invoke({"topic": "LangChain"})
print(result)  # "🤖 LangChain은 LLM 기반 애플리케이션 개발을..."
```

> 🔥 **실무 팁**: `RunnableLambda`를 쓰지 않고도, LCEL 체인에서 일반 함수를 파이프할 수 있습니다. LangChain이 자동으로 `RunnableLambda`로 감싸주거든요. `chain = prompt | model | parser | add_emoji` 이렇게 쓰면 됩니다. 다만 명시적으로 `RunnableLambda`를 사용하면 타입 힌트와 디버깅이 더 수월합니다.

## 실습: 직접 해보기

아키텍처를 직접 체감해봅시다. 아래 코드는 패키지 구조를 확인하고, Runnable 인터페이스의 세 가지 호출 방식과 LCEL 체인, 그리고 병렬 실행까지 한 번에 실습합니다.

```python
"""
LangChain 아키텍처 실습
- 패키지 구조 확인
- Runnable 인터페이스 (invoke / stream / batch)
- LCEL 체인 구성
- RunnableParallel 병렬 실행
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일에 OPENAI_API_KEY 필요)
load_dotenv()

# ============================================
# 1단계: 패키지 구조 확인 — 어디서 뭘 가져오는지
# ============================================
from langchain_core.prompts import ChatPromptTemplate       # core: 기초 추상화
from langchain_core.output_parsers import StrOutputParser    # core: 출력 파서
from langchain_core.runnables import (                       # core: Runnable 유틸리티
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_openai import ChatOpenAI                      # 통합 패키지: OpenAI

# 모든 컴포넌트가 Runnable인지 확인
from langchain_core.runnables import Runnable

prompt = ChatPromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해주세요.")
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
parser = StrOutputParser()

print("=== Runnable 인터페이스 확인 ===")
print(f"ChatPromptTemplate은 Runnable인가? {isinstance(prompt, Runnable)}")  # True
print(f"ChatOpenAI는 Runnable인가?        {isinstance(model, Runnable)}")   # True
print(f"StrOutputParser는 Runnable인가?   {isinstance(parser, Runnable)}")  # True


# ============================================
# 2단계: invoke / stream / batch 체험
# ============================================
print("\n=== invoke: 단일 호출 ===")
chain = prompt | model | parser
result = chain.invoke({"topic": "LangChain"})
print(f"결과: {result}")

print("\n=== stream: 실시간 스트리밍 ===")
print("결과: ", end="")
for chunk in chain.stream({"topic": "LCEL"}):
    print(chunk, end="", flush=True)  # 토큰이 생성될 때마다 출력
print()  # 줄바꿈

print("\n=== batch: 일괄 처리 ===")
topics = [
    {"topic": "Runnable"},
    {"topic": "RAG"},
    {"topic": "Agent"}
]
results = chain.batch(topics)
for topic, result in zip(topics, results):
    print(f"  {topic['topic']}: {result[:60]}...")  # 앞 60자만 출력


# ============================================
# 3단계: RunnableParallel로 병렬 체인
# ============================================
print("\n=== RunnableParallel: 병렬 실행 ===")

# 요약 체인
summary_prompt = ChatPromptTemplate.from_template(
    "{topic}을 초등학생도 이해할 수 있게 한 문장으로 요약해주세요."
)
summary_chain = summary_prompt | model | parser

# 활용 사례 체인
usecase_prompt = ChatPromptTemplate.from_template(
    "{topic}의 실제 활용 사례를 하나만 간단히 말해주세요."
)
usecase_chain = usecase_prompt | model | parser

# 두 체인을 병렬로 실행
parallel_chain = RunnableParallel(
    summary=summary_chain,
    usecase=usecase_chain,
)

parallel_result = parallel_chain.invoke({"topic": "벡터 데이터베이스"})
print(f"  요약: {parallel_result['summary']}")
print(f"  활용: {parallel_result['usecase']}")


# ============================================
# 4단계: RunnableLambda로 커스텀 변환 추가
# ============================================
print("\n=== RunnableLambda: 커스텀 변환 ===")

def format_as_card(data: dict) -> str:
    """병렬 실행 결과를 카드 형태로 포맷팅하는 함수"""
    return (
        f"┌──────────────────────────────┐\n"
        f"│ 📝 요약: {data['summary'][:30]}...\n"
        f"│ 💼 활용: {data['usecase'][:30]}...\n"
        f"└──────────────────────────────┘"
    )

# 병렬 체인 뒤에 포맷팅 함수를 연결
full_chain = parallel_chain | RunnableLambda(format_as_card)
card = full_chain.invoke({"topic": "임베딩"})
print(card)

print("\n✅ 실습 완료!")
print("  - langchain-core: 프롬프트, 파서, Runnable 유틸리티")
print("  - langchain-openai: ChatOpenAI 모델")
print("  - invoke / stream / batch: 세 가지 호출 패턴")
print("  - RunnableParallel + RunnableLambda: 병렬 실행과 커스텀 변환")
```

```
# 예상 출력:
# === Runnable 인터페이스 확인 ===
# ChatPromptTemplate은 Runnable인가? True
# ChatOpenAI는 Runnable인가?        True
# StrOutputParser는 Runnable인가?   True
#
# === invoke: 단일 호출 ===
# 결과: LangChain은 LLM 기반 애플리케이션 개발을 위한 오픈소스 프레임워크입니다.
#
# === stream: 실시간 스트리밍 ===
# 결과: LCEL은 파이프 연산자(|)를 사용하여 LangChain 컴포넌트를 조합하는 선언적 언어입니다.
#
# === batch: 일괄 처리 ===
#   Runnable: Runnable은 LangChain의 핵심 추상화로, invoke/stream/batch 인터...
#   RAG: RAG는 외부 지식을 검색하여 LLM 응답의 정확성을 높이는 기법입...
#   Agent: Agent는 LLM이 도구를 자율적으로 선택하여 복잡한 작업을 수행하...
#
# === RunnableParallel: 병렬 실행 ===
#   요약: 벡터 데이터베이스는 글이나 그림의 의미를 숫자로 바꿔서 비슷한 것끼리 빠르게 찾아주는 창고예요.
#   활용: 넷플릭스가 사용자 취향에 맞는 영화를 추천할 때 벡터 데이터베이스를 활용합니다.
#
# === RunnableLambda: 커스텀 변환 ===
# ┌──────────────────────────────┐
# │ 📝 요약: 임베딩은 텍스트를 숫자 벡터로 변환하여...
# │ 💼 활용: 구글 검색엔진이 검색어와 웹페이지의...
# └──────────────────────────────┘
#
# ✅ 실습 완료!
```

## 더 깊이 알아보기

### 아키텍처 분리의 배경 — "거대한 하나"에서 "잘 나눈 여럿"으로

LangChain은 처음부터 지금 같은 패키지 구조가 아니었습니다. 2022년 10월 Harrison Chase가 최초 커밋을 했을 때는 모든 것이 하나의 `langchain` 패키지에 담겨 있었습니다. 프롬프트 템플릿도, OpenAI 연동도, 벡터 스토어도, 에이전트도 전부 하나의 덩어리였죠.

문제는 LangChain이 폭발적으로 성장하면서 드러났습니다. 700개가 넘는 서드파티 통합이 하나의 패키지에 몰리다 보니, 사용자가 FAISS 벡터 스토어 하나만 쓰고 싶어도 수십 개의 불필요한 의존성이 따라왔습니다. 한 통합의 버그 수정이 다른 통합에 영향을 미치기도 했고, 테스트와 릴리스 속도도 느려졌죠.

2024년 1월, LangChain 팀은 **v0.1 릴리스**와 함께 대대적인 아키텍처 분리를 단행합니다. 핵심 추상화를 `langchain-core`로, 커뮤니티 통합을 `langchain-community`로, 주요 파트너 통합을 `langchain-openai`, `langchain-anthropic` 등 독립 패키지로 분리한 것이죠. LangChain 블로그에서는 이를 "LangChain의 가장 중요한 아키텍처 변화"라고 표현했습니다.

이 결정의 영감은 사실 **JavaScript 생태계**에서 왔습니다. Node.js의 npm 패키지가 작은 단위로 분리되어 있어 필요한 것만 골라 쓸 수 있는 것처럼, LangChain도 같은 철학을 채택한 것입니다. "Do one thing well"이라는 Unix 철학의 현대적 적용이라고도 할 수 있겠죠.

### LCEL과 Unix 파이프 — 50년 된 아이디어의 부활

LCEL의 파이프 연산자(`|`)가 어디서 영감을 받았는지 아시나요? 1973년, Bell Labs의 **Doug McIlroy**가 Unix에 도입한 **파이프(pipe)** 개념입니다. `cat file.txt | grep "error" | sort | uniq` — 이 Unix 명령어에서 `|`는 앞 명령의 출력을 뒤 명령의 입력으로 전달합니다. LCEL이 하는 것과 정확히 같은 원리죠.

McIlroy는 "프로그램은 한 가지 일을 잘 하도록 만들고, 프로그램끼리 협력하도록 만들라"고 했습니다. 50년이 지난 2023년, Harrison Chase는 이 철학을 LLM 애플리케이션에 되살린 셈입니다. 각 Runnable이 "한 가지 일을 잘 하는 프로그램"이고, 파이프 연산자가 이들을 "협력하게 만드는 접착제"인 거죠.

> 💡 **알고 계셨나요?**: LCEL이 처음부터 있었던 것은 아닙니다. 초기 LangChain은 `LLMChain`, `SequentialChain` 같은 **클래스 기반 체인**을 사용했습니다. LCEL은 2023년 중반에 도입되었고, 더 직관적이고 유연한 체인 구성 방식으로 기존의 클래스 기반 체인을 점차 대체하고 있습니다. 현재는 LCEL이 공식적으로 권장되는 체인 구성 방법입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "LCEL은 그냥 문법 설탕(syntactic sugar)이다" — LCEL의 파이프 연산자가 단순한 편의 기능처럼 보일 수 있지만, 실제로는 그 이상입니다. 파이프로 연결하는 순간 **자동 스트리밍**, **자동 비동기 지원**, **자동 배치 처리**, **LangSmith 트레이싱 통합**이 모두 활성화됩니다. 같은 기능을 수동으로 구현하려면 상당한 보일러플레이트 코드가 필요합니다.

> 💡 **알고 계셨나요?**: `langchain-core`의 PyPI 다운로드 수가 `langchain`보다 훨씬 많습니다. 이는 많은 통합 패키지(`langchain-openai`, `langchain-anthropic` 등)가 `langchain-core`에만 의존하고, 메인 `langchain` 패키지 없이도 동작할 수 있기 때문입니다. 즉, 간단한 LCEL 체인이라면 `langchain-core` + 통합 패키지만으로도 충분합니다.

> 🔥 **실무 팁**: 프로젝트를 시작할 때 **최소 설치 전략**을 사용하세요. 처음에는 `pip install langchain-core langchain-openai`만 설치하고, 벡터 스토어가 필요하면 `langchain-community`를, 에이전트가 필요하면 `langchain`을 추가하는 식으로 점진적으로 의존성을 늘리는 것이 좋습니다. `requirements.txt`에 패키지별 버전을 명시하면 재현 가능한 환경을 유지할 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `langchain-core` | 기초 추상화(Runnable, LCEL, 프롬프트, 파서)를 정의하는 경량 핵심 패키지 |
| `langchain-openai` 등 | 특정 LLM 제공자와의 공식 통합을 담당하는 파트너 패키지 |
| `langchain-community` | 수백 개의 서드파티 통합(문서 로더, 벡터 스토어 등)을 모은 커뮤니티 패키지 |
| `langchain` | 체인, 에이전트, 메모리 등 고수준 기능을 제공하는 메인 패키지 |
| Runnable 인터페이스 | 모든 컴포넌트가 구현하는 통합 인터페이스 — `invoke`, `stream`, `batch` + 비동기 버전 |
| LCEL | 파이프 연산자(`\|`)로 Runnable을 연결하는 선언적 체인 구성 언어 |
| `RunnableSequence` | `\|`로 연결된 체인의 내부 표현 — 자동 스트리밍/배치/비동기 지원 |
| `RunnableParallel` | 여러 Runnable을 병렬로 실행하고 결과를 딕셔너리로 반환 |
| `RunnablePassthrough` | 입력을 변환 없이 그대로 통과시키는 유틸리티 |
| `RunnableLambda` | 일반 Python 함수를 Runnable로 감싸는 래퍼 |

## 다음 섹션 미리보기

이번 세션에서 LangChain의 패키지 구조와 Runnable/LCEL의 원리를 이해했습니다. 하지만 아직 실제로 코드를 실행해보지는 못했죠(API 키 설정이 아직이니까요). 다음 세션 **[개발 환경 설정](03-개발-환경-설정.md)**에서는 Python 가상환경을 만들고, 오늘 배운 패키지 구조에 맞춰 필요한 패키지를 설치하며, API 키를 안전하게 관리하는 방법을 실습합니다. 이 세션에서 배운 "최소 설치 전략"을 직접 적용해볼 수 있을 거예요.

## 참고 자료

- [Architecture | LangChain 공식 문서](https://python.langchain.com/docs/concepts/architecture/) — LangChain의 패키지 구조와 계층을 설명하는 공식 아키텍처 문서
- [Towards LangChain 0.1: LangChain-Core and LangChain-Community](https://blog.langchain.com/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/) — 아키텍처 분리의 이유와 과정을 설명한 LangChain 공식 블로그 포스트
- [LangChain Expression Language (LCEL) | 공식 문서](https://python.langchain.com/docs/concepts/lcel/) — LCEL의 개념, 문법, 사용법을 다루는 공식 가이드
- [Runnable interface | LangChain 공식 문서](https://python.langchain.com/docs/concepts/runnables/) — Runnable 프로토콜의 모든 메서드와 사용 패턴을 다루는 레퍼런스
- [LangChain GitHub 저장소](https://github.com/langchain-ai/langchain) — 소스 코드, 패키지 구조, 최신 릴리스 확인
- [LangChain Expression Language Explained | Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/) — LCEL의 개념과 활용을 시각적으로 설명하는 튜토리얼

---

```context_meta
{
  "title": "LangChain 아키텍처 개요",
  "key_concepts": ["패키지 아키텍처", "Runnable 인터페이스", "LCEL", "RunnableSequence", "RunnableParallel", "RunnablePassthrough", "RunnableLambda"],
  "defines": ["langchain-core", "langchain-community", "langchain-openai", "Runnable", "RunnableSequence", "RunnableParallel", "RunnablePassthrough", "RunnableLambda", "invoke", "stream", "batch", "ainvoke", "astream", "abatch", "LCEL", "__or__"],
  "requires": ["LangChain", "LCEL", "Chain", "Runnable"],
  "code_patterns": ["from langchain_core.runnables import RunnablePassthrough", "from langchain_core.runnables import RunnableParallel", "from langchain_core.runnables import RunnableLambda", "chain = prompt | model | parser", "chain.invoke()", "chain.stream()", "chain.batch()", "isinstance(component, Runnable)"],
  "code_imports": ["from langchain_core.prompts import ChatPromptTemplate", "from langchain_core.output_parsers import StrOutputParser", "from langchain_core.runnables import RunnablePassthrough", "from langchain_core.runnables import RunnableParallel", "from langchain_core.runnables import RunnableLambda", "from langchain_core.runnables import Runnable", "from langchain_openai import ChatOpenAI"],
  "difficulty_level": 3,
  "connects_to_next": "Python 가상환경 설정, LangChain 패키지 설치(최소 의존성 전략 적용), API 키 관리, 프로젝트 디렉토리 구조 설계",
  "summary": "LangChain의 4계층 패키지 아키텍처(langchain-core, 통합 패키지, langchain-community, langchain)를 분석하고, 모든 컴포넌트가 공유하는 Runnable 인터페이스(invoke/stream/batch)의 원리를 설명합니다. LCEL의 파이프 연산자가 내부적으로 RunnableSequence를 생성하는 메커니즘과, RunnableParallel/RunnablePassthrough/RunnableLambda 등 체인 구성 유틸리티의 역할과 사용법을 다룹니다."
}
```
