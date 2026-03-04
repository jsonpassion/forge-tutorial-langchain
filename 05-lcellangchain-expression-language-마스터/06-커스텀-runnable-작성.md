# 커스텀 Runnable 작성

> 나만의 Runnable을 만들어 LCEL 체인의 무한한 확장성을 경험하세요

## 개요

이 섹션에서는 LCEL의 마지막 퍼즐 조각인 **커스텀 Runnable 작성**을 다룹니다. 지금까지 LangChain이 제공하는 내장 Runnable들을 조합하는 법을 배웠다면, 이제는 여러분만의 로직을 Runnable로 만들어 체인에 자연스럽게 끼워 넣는 방법을 익힙니다. 더불어 복잡해진 체인을 디버깅하고 시각화하는 도구도 함께 살펴봅니다.

**선수 지식**: 앞서 [5.1: LCEL 기초와 파이프 연산자](5.1)에서 배운 Runnable 프로토콜(invoke/stream/batch), [5.2: 핵심 Runnable 컴포넌트](5.2)에서 배운 RunnableLambda, [5.5: 체인 구성과 바인딩](5.5)에서 배운 with_fallbacks/with_retry 등의 개념이 필요합니다.

**학습 목표**:
- `@chain` 데코레이터로 함수를 Runnable 체인으로 변환할 수 있다
- `RunnableGenerator`로 스트리밍을 보존하는 커스텀 변환 로직을 구현할 수 있다
- `Runnable` 기반 클래스를 상속하여 완전한 커스텀 Runnable을 작성할 수 있다
- `get_graph()`, `print_ascii()`, `input_schema/output_schema`로 체인을 디버깅하고 시각화할 수 있다

## 왜 알아야 할까?

LangChain이 제공하는 내장 컴포넌트만으로 모든 문제를 해결할 수 있을까요? 현실은 그렇지 않습니다. 실제 프로젝트에서는 이런 상황이 빈번하게 발생하거든요:

- LLM 응답에서 특정 패턴을 **스트리밍 중에 실시간으로 변환**해야 할 때
- 외부 API를 호출하면서도 LCEL 체인의 **invoke/stream/batch 인터페이스**를 유지해야 할 때
- 비즈니스 로직이 복잡해서 단순 RunnableLambda로는 **스트리밍이 깨지는** 문제가 생길 때
- 체인이 20단계 이상으로 커져서 **어디서 문제가 생기는지 추적**이 어려울 때

커스텀 Runnable을 만들 수 있다는 건, LCEL이라는 레고 시스템에 **나만의 블록을 추가**할 수 있다는 뜻입니다. 이것이 바로 프레임워크 사용자에서 프레임워크 확장자로 성장하는 전환점이죠.

## 핵심 개념

### 개념 1: `@chain` 데코레이터 — 함수를 체인으로 바꾸는 마법

> 💡 **비유**: 일반 요리사가 만든 레시피를 호텔 주방 시스템에 등록하면, 갑자기 주문 처리, 재고 추적, 품질 관리 등 호텔 시스템의 모든 기능을 자동으로 이용할 수 있게 되는 것과 비슷합니다. `@chain`은 여러분의 평범한 Python 함수를 LCEL 생태계에 등록해주는 데코레이터입니다.

[5.2: 핵심 Runnable 컴포넌트](5.2)에서 `RunnableLambda`로 일반 함수를 Runnable로 감쌀 수 있다는 걸 배웠죠? `@chain` 데코레이터는 그 과정을 더 우아하게 만들어줍니다. 핵심 차이는 **함수 내부에서 다른 Runnable을 호출하면 자동으로 트레이싱 의존성이 기록**된다는 점입니다.

```python
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 프롬프트 템플릿 정의
prompt_joke = ChatPromptTemplate.from_template(
    "'{topic}'에 대한 재미있는 한 줄 농담을 만들어주세요."
)
prompt_explain = ChatPromptTemplate.from_template(
    "다음 농담이 왜 재미있는지 설명해주세요: {joke}"
)

model = ChatOpenAI(model="gpt-4o", temperature=0.7)
parser = StrOutputParser()

@chain
def joke_and_explain(topic: str) -> str:
    """농담을 생성하고, 그 농담을 설명하는 커스텀 체인"""
    # 1단계: 농담 생성
    joke_chain = prompt_joke | model | parser
    joke = joke_chain.invoke({"topic": topic})

    # 2단계: 농담 설명 (1단계 결과를 활용)
    explain_chain = prompt_explain | model | parser
    explanation = explain_chain.invoke({"joke": joke})

    return f"🎭 농담: {joke}\n\n📖 해설: {explanation}"

# 일반 함수처럼 보이지만, Runnable의 모든 기능을 갖춤!
result = joke_and_explain.invoke("프로그래밍")
print(result)

# 배치 처리도 가능
results = joke_and_explain.batch(["파이썬", "자바스크립트", "러스트"])
```

`@chain`의 진짜 위력은 **LangSmith 트레이싱**에서 빛납니다. 함수 이름(`joke_and_explain`)이 자동으로 Runnable의 이름이 되고, 내부에서 호출하는 `joke_chain`과 `explain_chain`이 하위 의존성으로 기록됩니다. 디버깅할 때 "어떤 체인이 어떤 순서로 실행됐는지" 한눈에 파악할 수 있죠.

**`@chain` vs `RunnableLambda` — 언제 뭘 쓸까?**

| 상황 | 추천 |
|------|------|
| 단순 데이터 변환 (문자열 가공 등) | `RunnableLambda` |
| 내부에서 다른 Runnable을 호출 | `@chain` |
| 트레이싱/관찰 가능성이 중요 | `@chain` |
| 기존 함수를 인라인으로 감싸기 | `RunnableLambda` |

### 개념 2: `RunnableGenerator` — 스트리밍을 지키는 수호자

> 💡 **비유**: 공장의 컨베이어 벨트를 떠올려보세요. 일반 작업자(RunnableLambda)는 물건이 전부 도착할 때까지 기다렸다가 한 번에 처리합니다. 하지만 컨베이어 벨트 위의 스마트 작업자(RunnableGenerator)는 물건이 하나씩 올 때마다 바로바로 처리해서 다음 공정으로 넘깁니다. 이것이 스트리밍을 보존한다는 의미입니다.

[5.4: 병렬 실행과 성능 최적화](5.4)에서 `astream`의 중요성을 배웠는데요, `RunnableLambda`에는 치명적인 약점이 하나 있습니다. **입력을 모두 버퍼링한 후에야 처리를 시작**한다는 점이죠. LLM이 토큰을 하나씩 스트리밍하는데, 중간에 `RunnableLambda`가 끼어 있으면 스트리밍이 끊깁니다.

`RunnableGenerator`는 이 문제를 해결합니다. 제너레이터 함수의 시그니처가 `Iterator[A] → Iterator[B]`인 경우, 입력 청크가 도착하는 대로 출력 청크를 내보낼 수 있거든요.

```python
from langchain_core.runnables import RunnableGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import Iterator

# 스트리밍 중에 실시간으로 텍스트를 변환하는 제너레이터
def censor_streaming(chunks: Iterator[str]) -> Iterator[str]:
    """스트리밍 중 민감한 단어를 실시간으로 마스킹"""
    buffer = ""
    sensitive_words = ["비밀번호", "주민등록번호", "계좌번호"]

    for chunk in chunks:
        buffer += chunk
        # 민감 단어가 완성될 수 있으므로 버퍼 유지
        can_flush = True
        for word in sensitive_words:
            # 민감 단어의 일부가 버퍼 끝에 있으면 대기
            for i in range(1, len(word)):
                if buffer.endswith(word[:i]):
                    can_flush = False
                    break
            if not can_flush:
                break

        if can_flush:
            # 민감 단어 치환 후 출력
            for word in sensitive_words:
                buffer = buffer.replace(word, "***")
            yield buffer
            buffer = ""

    # 남은 버퍼 처리
    if buffer:
        for word in sensitive_words:
            buffer = buffer.replace(word, "***")
        yield buffer

# RunnableGenerator로 감싸서 체인에 삽입
censor = RunnableGenerator(censor_streaming)

prompt = ChatPromptTemplate.from_template("{question}에 대해 설명해주세요.")
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# 스트리밍이 보존되는 체인!
chain = prompt | model | parser | censor

# stream()으로 실시간 출력 확인
for chunk in chain.stream({"question": "개인정보 보호"}):
    print(chunk, end="", flush=True)
```

**핵심 포인트**: `RunnableGenerator`의 시그니처는 `Iterator[입력타입] → Iterator[출력타입]`입니다. 이 시그니처 덕분에 이전 단계의 스트리밍 청크를 하나씩 받아서 바로 변환 후 다음 단계로 넘길 수 있습니다.

비동기 버전도 간단합니다:

```python
from typing import AsyncIterator

# 비동기 제너레이터도 지원
async def async_censor(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """비동기 스트리밍 변환"""
    async for chunk in chunks:
        # 간단한 변환 예시
        yield chunk.replace("secret", "***")

# transform과 atransform을 둘 다 지정 가능
censor = RunnableGenerator(
    transform=censor_streaming,
    atransform=async_censor
)
```

### 개념 3: Runnable 서브클래스 — 완전한 커스텀 컴포넌트

> 💡 **비유**: `@chain`이 기성복의 약간의 수선이라면, `RunnableGenerator`는 원단을 골라 맞춤 제작하는 것이고, `Runnable` 서브클래스는 아예 새로운 의류 브랜드를 만드는 것입니다. 가장 자유도가 높지만, 그만큼 알아야 할 것도 많습니다.

정말 복잡한 로직이 필요하거나, `input_schema`/`output_schema`를 정확히 제어하고 싶을 때는 `Runnable` 기반 클래스를 직접 상속합니다. 핵심은 `invoke` 메서드를 반드시 구현하고, 스트리밍이 필요하면 `transform` 메서드도 오버라이드하는 것입니다.

```python
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel
from typing import Any, Optional, Iterator

class TextStatistics(Runnable[str, dict]):
    """텍스트 통계를 분석하는 커스텀 Runnable

    입력: 분석할 텍스트 (str)
    출력: 통계 정보 (dict)
    """
    name: str = "TextStatistics"

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> dict:
        """텍스트 통계를 계산하여 반환"""
        words = input.split()
        sentences = [s.strip() for s in input.split('.') if s.strip()]
        return {
            "char_count": len(input),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": round(
                sum(len(w) for w in words) / max(len(words), 1), 2
            ),
            "unique_words": len(set(w.lower() for w in words)),
        }

# 사용법: 다른 Runnable과 동일!
stats = TextStatistics()

# invoke
result = stats.invoke("LangChain은 LLM 기반 앱을 만드는 프레임워크입니다. LCEL로 체인을 구성합니다.")
print(result)
# 출력: {'char_count': 49, 'word_count': 7, 'sentence_count': 2, ...}

# 파이프 연산자로 체인에 삽입
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("{topic}에 대해 3문장으로 설명해주세요.")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
    | stats  # 커스텀 Runnable을 파이프에 연결!
)

result = chain.invoke({"topic": "인공지능"})
print(result)  # LLM 출력의 텍스트 통계가 나옴
```

스트리밍까지 지원하는 더 고급 예제를 보겠습니다:

```python
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Optional, Iterator

class StreamingWordCounter(Runnable[str, str]):
    """스트리밍 중 단어 수를 실시간 추적하는 커스텀 Runnable"""
    name: str = "StreamingWordCounter"

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> str:
        word_count = len(input.split())
        return f"{input}\n\n---\n📊 총 단어 수: {word_count}"

    def transform(
        self,
        input: Iterator[str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """스트리밍 입력을 받아 실시간으로 통과시키며 카운팅"""
        word_count = 0
        for chunk in input:
            # 청크를 그대로 통과시킴 (스트리밍 유지!)
            word_count += len(chunk.split())
            yield chunk
        # 스트리밍 끝나면 통계 추가
        yield f"\n\n---\n📊 총 단어 수: {word_count}"
```

> ⚠️ **흔한 오해**: `invoke`만 구현하면 `stream`이 자동으로 되는 거 아닌가요? 아닙니다! 기본 `transform` 구현은 **입력을 전부 버퍼링한 후 `invoke`를 호출**합니다. 진짜 스트리밍을 원하면 `transform`(동기)이나 `atransform`(비동기)을 반드시 오버라이드해야 합니다.

### 개념 4: 체인 디버깅과 시각화

> 💡 **비유**: 복잡한 배관 공사를 할 때, 배관도(도면) 없이 문제를 찾으려면 벽을 다 뜯어야 합니다. LCEL의 시각화 도구는 여러분의 체인에 대한 배관도를 자동으로 그려주는 겁니다.

체인이 복잡해질수록 "지금 이 체인이 정확히 어떤 구조인지" 파악하기 어려워집니다. LangChain은 모든 Runnable에 검사(inspection) 메서드를 제공하여 이 문제를 해결합니다.

**1. `get_graph().print_ascii()` — 체인 구조 시각화**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("'{topic}'을 설명해주세요.")
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

chain = prompt | model | parser

# ASCII 아트로 체인 구조 출력
chain.get_graph().print_ascii()
# 출력 예시:
#      +-------------+
#      | PromptInput |
#      +-------------+
#             *
#             *
#             *
#   +-----------------+
#   | ChatPromptTemplate |
#   +-----------------+
#             *
#             *
#             *
#      +------------+
#      | ChatOpenAI  |
#      +------------+
#             *
#             *
#             *
#   +-----------------+
#   | StrOutputParser |
#   +-----------------+
#             *
#             *
#             *
#   +-----------------+
#   | StrOutputParserOutput |
#   +-----------------+
```

**2. `input_schema` / `output_schema` — 입출력 타입 확인**

```python
# 체인이 기대하는 입력 스키마
print(chain.input_schema.model_json_schema())
# 출력: {'properties': {'topic': {'title': 'Topic', 'type': 'string'}},
#        'required': ['topic'], 'title': 'PromptInput', 'type': 'object'}

# 체인이 반환하는 출력 스키마
print(chain.output_schema.model_json_schema())
# 출력: {'title': 'StrOutputParserOutput', 'type': 'string'}
```

**3. `get_prompts()` — 체인 내 프롬프트 확인**

```python
# 체인에 포함된 모든 프롬프트 목록
prompts = chain.get_prompts()
for p in prompts:
    print(p.template)
# 출력: '{topic}'을 설명해주세요.
```

**4. 복잡한 체인의 그래프 시각화**

병렬 실행이나 분기가 포함된 체인의 구조도 볼 수 있습니다:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 병렬 + 순차 조합 체인
analysis_chain = (
    RunnableParallel(
        summary=ChatPromptTemplate.from_template("요약: {text}") | model | parser,
        keywords=ChatPromptTemplate.from_template("키워드: {text}") | model | parser,
        original=RunnablePassthrough()
    )
    | ChatPromptTemplate.from_messages([
        ("system", "요약과 키워드를 결합하여 보고서를 작성하세요."),
        ("human", "요약: {summary}\n키워드: {keywords}\n원본: {original}")
    ])
    | model
    | parser
)

# 복잡한 체인도 한눈에!
analysis_chain.get_graph().print_ascii()
```

> 🔥 **실무 팁**: `get_graph()`가 반환하는 `Graph` 객체에서 `.draw_mermaid()`를 호출하면 Mermaid.js 형식의 다이어그램 코드를 얻을 수 있습니다. 이걸 Jupyter Notebook이나 GitHub README에 붙여넣으면 시각적으로 더 아름다운 체인 구조도를 만들 수 있죠.

```python
# Mermaid 다이어그램 생성
mermaid_code = chain.get_graph().draw_mermaid()
print(mermaid_code)

# Jupyter에서 바로 렌더링하려면:
# from IPython.display import display, Image
# img_bytes = chain.get_graph().draw_mermaid_png()
# display(Image(img_bytes))
```

## 실습: 직접 해보기

지금까지 배운 모든 기법을 결합하여, **스트리밍 가능한 감정 분석 파이프라인**을 만들어보겠습니다.

```python
"""
LCEL 커스텀 Runnable 종합 실습
- @chain 데코레이터로 멀티스텝 체인 구성
- RunnableGenerator로 스트리밍 변환
- 체인 시각화로 구조 확인
"""
import os
from typing import Iterator
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    chain,
    RunnableGenerator,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

load_dotenv()

# --- 모델 설정 ---
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.3

model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
parser = StrOutputParser()

# --- 1단계: RunnableGenerator로 스트리밍 이모지 태거 ---
def emoji_tagger(chunks: Iterator[str]) -> Iterator[str]:
    """스트리밍 중 감정 키워드에 이모지를 실시간 삽입"""
    # 감정-이모지 매핑
    emoji_map = {
        "긍정": "😊",
        "부정": "😞",
        "분노": "😡",
        "기쁨": "🎉",
        "슬픔": "😢",
        "놀라움": "😲",
        "중립": "😐",
    }

    for chunk in chunks:
        result = chunk
        for keyword, emoji in emoji_map.items():
            result = result.replace(keyword, f"{keyword}{emoji}")
        yield result

streaming_tagger = RunnableGenerator(emoji_tagger)

# --- 2단계: @chain으로 감정 분석 + 요약 체인 ---
prompt_sentiment = ChatPromptTemplate.from_template(
    """다음 텍스트의 감정을 분석해주세요.

텍스트: {text}

다음 형식으로 응답해주세요:
- 전체 감정: (긍정/부정/중립)
- 감정 강도: (1-10)
- 핵심 감정 키워드: (기쁨, 슬픔, 분노, 놀라움 등에서 선택)
- 근거: (왜 그렇게 판단했는지 2-3문장)"""
)

prompt_advice = ChatPromptTemplate.from_template(
    """다음 감정 분석 결과를 바탕으로, 이 텍스트 작성자에게 도움이 될 조언을 해주세요.

감정 분석:
{analysis}

원본 텍스트:
{text}

공감하는 톤으로 2-3문장의 조언을 작성해주세요."""
)

@chain
def sentiment_pipeline(text: str) -> str:
    """감정 분석 → 조언 생성 → 이모지 태깅 파이프라인"""
    # 1단계: 감정 분석
    analysis_chain = prompt_sentiment | model | parser
    analysis = analysis_chain.invoke({"text": text})

    # 2단계: 조언 생성 (감정 분석 결과 활용)
    advice_chain = prompt_advice | model | parser | streaming_tagger
    advice = advice_chain.invoke({"analysis": analysis, "text": text})

    return f"📊 감정 분석 결과:\n{analysis}\n\n💬 조언:\n{advice}"

# --- 3단계: 실행 및 시각화 ---
# 체인 실행
sample_text = "오늘 프로젝트 발표가 정말 잘 됐어요! 팀원들도 모두 칭찬해줬고, 상사도 만족해하셨습니다."
result = sentiment_pipeline.invoke(sample_text)
print(result)

print("\n" + "=" * 60 + "\n")

# 별도의 시각화 가능한 체인 구성 (파이프 연산자 조합)
visual_chain = (
    RunnableParallel(
        analysis=prompt_sentiment | model | parser,
        text=RunnablePassthrough()
    )
    | prompt_advice
    | model
    | parser
    | streaming_tagger
)

# 체인 구조 시각화
print("📐 체인 구조도:")
visual_chain.get_graph().print_ascii()

# 입출력 스키마 확인
print("\n📥 입력 스키마:")
print(visual_chain.input_schema.model_json_schema())

print("\n📤 출력 스키마:")
print(visual_chain.output_schema.model_json_schema())

# 스트리밍으로 실행
print("\n🔄 스트리밍 출력:")
for chunk in visual_chain.stream({"text": sample_text}):
    print(chunk, end="", flush=True)
```

이 실습에서 핵심적으로 확인할 부분은 다음과 같습니다:

1. `@chain`으로 감싼 `sentiment_pipeline`이 일반 Runnable처럼 `.invoke()`, `.batch()` 등을 지원하는 것
2. `RunnableGenerator`인 `streaming_tagger`가 파이프 연산자(|)로 자연스럽게 체인에 결합되는 것
3. `get_graph().print_ascii()`로 복잡한 체인의 구조를 한눈에 파악할 수 있는 것
4. 스트리밍 중에 이모지가 실시간으로 삽입되는 것

## 더 깊이 알아보기

### LCEL의 탄생 — Harrison Chase의 "추상화 고민"

LangChain의 창시자 Harrison Chase는 2022년 10월, Robust Intelligence에서 일하면서 사이드 프로젝트로 LangChain을 시작했습니다. 초기 LangChain은 Python의 `formatter.format`을 감싸는 아주 단순한 프롬프트 템플릿이었죠.

그런데 사용자가 폭발적으로 늘면서 문제가 생겼습니다. 모든 사람이 **서로 다른 방식으로 체인을 구성**하고 있었거든요. 어떤 사람은 함수를 연결하고, 어떤 사람은 클래스를 상속하고, 또 어떤 사람은 딕셔너리를 중첩했습니다.

2023년 3분기, LangChain 팀은 이 혼란을 해결하기 위해 **LCEL(LangChain Expression Language)**을 도입합니다. 핵심 아이디어는 Unix의 파이프 연산자(`|`)에서 왔습니다. `cat file | grep pattern | sort`처럼, LLM 파이프라인도 `prompt | model | parser`로 표현할 수 있어야 한다는 것이죠.

이 설계의 핵심에 `Runnable` 프로토콜이 있습니다. **"invoke, stream, batch만 구현하면 어떤 컴포넌트든 파이프로 연결할 수 있다"**는 단순한 약속이, 오늘날 우리가 배운 `@chain`, `RunnableGenerator`, 커스텀 서브클래스까지 모든 확장의 기반이 된 것입니다.

### 제너레이터 패턴의 계보

Python의 제너레이터(`yield`)는 2001년 PEP 255에서 도입되었습니다. CLU 언어와 Icon 언어의 영향을 받았죠. `RunnableGenerator`는 이 오래된 Python 패턴을 LLM 스트리밍이라는 현대적 요구에 연결한 아름다운 사례입니다. LLM이 토큰을 하나씩 생성하는 것은 본질적으로 제너레이터이며, 이 스트림을 변환하는 가장 Pythonic한 방법이 바로 제너레이터 합성(generator composition)입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "RunnableLambda로 충분한데 왜 RunnableGenerator를 써야 하나요?"
> `RunnableLambda`의 기본 `transform`은 입력을 전부 모은 후 `invoke`를 호출합니다. 즉, 체인 중간에 `RunnableLambda`가 있으면 **스트리밍이 끊깁니다**. LLM → RunnableLambda → 출력 구조에서 사용자는 LLM 응답이 전부 생성될 때까지 아무것도 못 보게 되죠. `RunnableGenerator`는 `Iterator → Iterator` 시그니처로 **청크 단위 실시간 변환**을 가능하게 합니다.

> 💡 **알고 계셨나요?**: `@chain` 데코레이터는 내부적으로 `RunnableLambda`를 생성합니다. 다만 함수 이름을 Runnable의 `name` 속성으로 설정하고, 내부에서 호출되는 Runnable들의 트레이싱 의존성을 자동 기록합니다. 소스 코드를 보면 `chain = RunnableLambda`와 동일한 객체임을 확인할 수 있습니다!

> 🔥 **실무 팁**: 프로덕션에서 체인 디버깅이 막막할 때는 이 순서로 접근하세요:
> 1. `chain.get_graph().print_ascii()`로 구조 확인
> 2. `chain.input_schema.model_json_schema()`로 입력 형식 확인
> 3. `LANGCHAIN_TRACING_V2=true`로 LangSmith 트레이싱 활성화
> 4. 문제 구간을 분리하여 `.invoke()`로 단독 테스트
> 5. [5.5: 체인 구성과 바인딩](5.5)에서 배운 `with_listeners()`로 중간 단계 로깅

> 🔥 **실무 팁**: 커스텀 Runnable 서브클래스를 만들 때, `invoke`만 구현하면 `batch`는 자동으로 `invoke`를 반복 호출합니다. 하지만 외부 API를 호출하는 Runnable이라면 `abatch`를 오버라이드하여 `asyncio.gather`로 병렬 호출하는 것이 성능상 훨씬 유리합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `@chain` 데코레이터 | 함수를 Runnable로 변환. 내부 Runnable 호출이 트레이싱에 자동 기록됨 |
| `RunnableGenerator` | `Iterator → Iterator` 시그니처로 스트리밍을 보존하는 커스텀 변환기 |
| `Runnable` 서브클래스 | `invoke` 필수 구현, `transform`/`atransform` 오버라이드로 스트리밍 지원 |
| `get_graph().print_ascii()` | 체인 구조를 ASCII 아트로 시각화 |
| `input_schema` / `output_schema` | 체인의 입출력 타입을 Pydantic 모델로 자동 생성 |
| `get_prompts()` | 체인에 포함된 모든 프롬프트 템플릿 목록 조회 |
| `draw_mermaid()` | Mermaid.js 형식의 다이어그램 코드 생성 |
| 스트리밍 보존 원칙 | `RunnableLambda`는 버퍼링, `RunnableGenerator`/`transform` 오버라이드는 실시간 처리 |

## 다음 섹션 미리보기

축하합니다! Chapter 5의 모든 LCEL 개념을 마스터했습니다. 파이프 연산자, 핵심 Runnable 컴포넌트, 조건 분기, 병렬 실행, 체인 바인딩, 그리고 커스텀 Runnable까지 — 이 모든 도구가 여러분의 손에 있습니다.

다음 [Chapter 6: 문서 로더와 텍스트 분할](ch6)에서는 LCEL로 구축한 파이프라인에 **외부 데이터를 연결하는 방법**을 배웁니다. PDF, 웹페이지, 데이터베이스 등 다양한 소스에서 문서를 불러오고, RAG를 위해 적절한 크기로 분할하는 기법을 다룹니다. LCEL 체인의 진정한 위력은 외부 데이터와 결합할 때 발휘되거든요!

## 참고 자료

- [How to inspect runnables | LangChain 공식 문서](https://python.langchain.com/docs/how_to/inspect/) - `get_graph()`, `input_schema`, `get_prompts()` 등 Runnable 검사 메서드의 공식 가이드
- [RunnableGenerator API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableGenerator.html) - RunnableGenerator의 상세 API 문서와 사용 예제
- [Create a runnable with the @chain decorator](https://python.langchain.com/v0.1/docs/expression_language/how_to/decorator/) - `@chain` 데코레이터의 공식 사용 가이드와 RunnableLambda와의 비교
- [LangChain Expression Language (LCEL) Concepts](https://python.langchain.com/docs/concepts/lcel/) - LCEL의 전체 개념을 정리한 공식 문서
- [Runnable Interface API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html) - Runnable 기반 클래스의 전체 메서드 목록과 서브클래싱 가이드

---
### 🔗 Related Sessions
- [lcel_pipe_operator](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [runnable_protocol](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [invoke_method](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [batch_method](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [runnable_lambda](../05-lcellangchain-expression-language-마스터/02-핵심-runnable-컴포넌트.md) (prerequisite)
- [runnable_parallel](../05-lcellangchain-expression-language-마스터/02-핵심-runnable-컴포넌트.md) (prerequisite)
- [stream_method](../05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md) (prerequisite)
- [async_execution](../05-lcellangchain-expression-language-마스터/04-병렬-실행과-성능-최적화.md) (prerequisite)
- [with_fallbacks_method](../05-lcellangchain-expression-language-마스터/05-체인-구성과-바인딩.md) (prerequisite)
- [with_listeners_method](../05-lcellangchain-expression-language-마스터/05-체인-구성과-바인딩.md) (prerequisite)
