# LCEL 기초와 파이프 연산자

> LangChain Expression Language의 핵심인 Runnable 프로토콜과 파이프 연산자(|)를 이해하고, invoke/stream/batch 통합 인터페이스를 마스터합니다.

## 개요

이 섹션에서는 LangChain의 가장 강력한 추상화인 **LCEL(LangChain Expression Language)**의 기초를 다룹니다. LCEL은 프롬프트, 모델, 파서 같은 컴포넌트를 파이프 연산자(`|`)로 연결하여 선언적으로 체인을 구성하는 방법인데요, 이를 통해 동기/비동기, 스트리밍, 배치 처리를 별도 코드 없이 자동으로 지원받을 수 있습니다.

**선수 지식**: [Ch4: 출력 파서와 구조화된 출력]에서 배운 Output Parser 개념, [Ch2: LLM과 Chat Model 다루기]에서 다룬 ChatModel 사용법, 그리고 [Ch3: 프롬프트 엔지니어링과 템플릿]에서 익힌 PromptTemplate 구성 방법이 필요합니다.

**학습 목표**:
- Runnable 프로토콜의 개념과 통합 인터페이스(`invoke`, `stream`, `batch`)를 이해한다
- 파이프 연산자(`|`)의 동작 원리와 `RunnableSequence`와의 관계를 파악한다
- LCEL로 프롬프트 → 모델 → 파서를 연결하는 기본 체인을 구성할 수 있다
- `invoke`, `stream`, `batch`, 그리고 비동기 메서드를 상황에 맞게 사용할 수 있다

## 왜 알아야 할까?

앞선 챕터들에서 우리는 프롬프트 템플릿, Chat Model, 출력 파서를 각각 개별적으로 다뤘습니다. 하지만 실제 애플리케이션에서는 이 컴포넌트들을 **하나의 파이프라인으로 엮어야** 하죠. 예를 들어 "사용자 질문을 받아 → 프롬프트에 채워 넣고 → 모델에 전달하고 → 결과를 파싱한다"는 흐름을 매번 수동으로 작성한다면 어떨까요?

```python
# LCEL 없이 수동으로 연결하는 방식 (번거롭다!)
prompt_value = prompt.format_messages(topic="AI")
response = model.invoke(prompt_value)
result = parser.parse(response.content)
```

이 코드는 동작하지만, 스트리밍은요? 배치 처리는요? 비동기 실행은요? 각각을 위해 별도 코드를 작성해야 합니다. LCEL은 이 문제를 단 한 줄로 해결합니다:

```python
# LCEL로 구성한 체인 — 스트리밍, 배치, 비동기 모두 자동 지원!
chain = prompt | model | parser
```

LCEL은 2023년 3분기에 도입된 이후 LangChain의 **공식 권장 체인 구성 방식**이 되었습니다. 레거시 `LLMChain`, `SequentialChain` 등을 대체하며, 현재 LangChain 생태계의 모든 예제와 문서가 LCEL 기반으로 작성되고 있거든요. LCEL을 모르면 LangChain의 최신 기능을 활용할 수 없다고 해도 과언이 아닙니다.

## 핵심 개념

### 개념 1: Runnable 프로토콜 — 모든 컴포넌트의 공통 언어

> 💡 **비유**: USB 규격을 떠올려 보세요. 키보드, 마우스, 외장하드 — 모양도 기능도 다르지만, USB 포트에 꽂기만 하면 동작합니다. Runnable은 LangChain의 USB 규격과 같습니다. 프롬프트, 모델, 파서 등 서로 다른 컴포넌트들이 **동일한 인터페이스**를 구현하기 때문에, 어떤 순서로든 자유롭게 연결할 수 있죠.

LangChain에서 `Runnable`은 모든 실행 가능한 컴포넌트가 따르는 **프로토콜(protocol)**입니다. `ChatPromptTemplate`, `ChatOpenAI`, `StrOutputParser` 등 우리가 앞서 배운 거의 모든 클래스가 `Runnable`을 상속하고 있어요.

Runnable이 보장하는 핵심 메서드는 다음과 같습니다:

| 메서드 | 설명 | 동기/비동기 |
|--------|------|------------|
| `invoke(input)` | 단일 입력을 처리하여 출력 반환 | 동기 |
| `ainvoke(input)` | `invoke`의 비동기 버전 | 비동기 |
| `stream(input)` | 출력을 생성되는 대로 스트리밍 | 동기 |
| `astream(input)` | `stream`의 비동기 버전 | 비동기 |
| `batch(inputs)` | 여러 입력을 병렬로 처리 | 동기 |
| `abatch(inputs)` | `batch`의 비동기 버전 | 비동기 |

이게 왜 대단한 걸까요? 어떤 Runnable이든 동일한 방식으로 호출할 수 있다는 뜻이거든요:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 세 가지 모두 Runnable — 동일한 인터페이스를 가짐
prompt = ChatPromptTemplate.from_template("{topic}에 대한 짧은 농담을 해줘")
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
parser = StrOutputParser()

# 각각 독립적으로 invoke 가능
prompt_value = prompt.invoke({"topic": "프로그래밍"})  # -> ChatPromptValue
response = model.invoke(prompt_value)                    # -> AIMessage
text = parser.invoke(response)                           # -> str
print(text)
# 출력 예시: "프로그래머가 숲에서 길을 잃으면? 트리를 순회합니다!"
```

### 개념 2: 파이프 연산자(|) — 레고 블록처럼 연결하기

> 💡 **비유**: 공장의 컨베이어 벨트를 상상해 보세요. 원재료가 들어가면 첫 번째 기계가 가공하고, 그 결과물이 벨트를 타고 두 번째 기계로 이동하고, 또 다음 기계로... 최종 제품이 나올 때까지 자동으로 흘러갑니다. LCEL의 파이프 연산자(`|`)가 바로 이 컨베이어 벨트입니다. 각 컴포넌트의 출력이 자동으로 다음 컴포넌트의 입력이 되죠.

파이프 연산자는 Unix 셸의 파이프(`|`)에서 영감을 받았습니다. Unix에서 `cat file.txt | grep "error" | wc -l`처럼 명령어를 연결하듯, LCEL에서는 컴포넌트를 연결합니다:

```python
# Unix 파이프와의 유사성
# Unix:  cat file.txt | grep "error" | wc -l
# LCEL:  prompt       | model       | parser

chain = prompt | model | parser
```

이 한 줄이 내부적으로는 `RunnableSequence`라는 객체를 생성합니다. 어떻게 가능한 걸까요? Python의 **연산자 오버로딩(operator overloading)** 덕분입니다:

```python
# 파이프 연산자의 내부 동작 원리
# prompt | model 은 사실 이것과 같다:
# prompt.__or__(model)
# 결과는 RunnableSequence([prompt, model])

# 따라서 prompt | model | parser 는:
# RunnableSequence([prompt, model]).__or__(parser)
# = RunnableSequence([prompt, model, parser])
```

`Runnable` 클래스에 정의된 `__or__` 메서드가 `|` 연산자를 가로채서, 두 Runnable을 `RunnableSequence`로 묶어주는 것이죠. `__ror__` 메서드도 구현되어 있어서 일반 함수나 딕셔너리도 파이프라인에 넣을 수 있습니다.

같은 결과를 `.pipe()` 메서드로도 얻을 수 있습니다:

```python
# pipe() 메서드 — | 연산자와 동일한 결과
chain = prompt.pipe(model).pipe(parser)

# 또는 한번에
chain = prompt.pipe(model, parser)
```

> ⚠️ **흔한 오해**: "파이프 연산자는 즉시 실행된다"고 생각하는 분이 많은데요, 사실 `|`는 **체인을 정의(구성)**할 뿐, 실행하지 않습니다. 실제 실행은 `invoke()`, `stream()` 등을 호출할 때 일어납니다. 이 점이 함수형 프로그래밍의 **합성(composition)**과 같은 개념이에요.

### 개념 3: invoke — 가장 기본적인 실행

`invoke`는 체인에 단일 입력을 넣고 최종 출력을 받는 가장 기본적인 실행 방법입니다:

```python
from dotenv import load_dotenv
load_dotenv()  # .env에서 OPENAI_API_KEY 로드

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 체인 구성
prompt = ChatPromptTemplate.from_template(
    "{country}의 수도는 어디인가요? 한 문장으로 답해주세요."
)
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

chain = prompt | model | parser

# invoke로 실행 — 입력은 딕셔너리
result = chain.invoke({"country": "프랑스"})
print(result)
# 출력: "프랑스의 수도는 파리(Paris)입니다."
print(type(result))
# 출력: <class 'str'>
```

데이터가 체인을 통과하는 흐름을 정리하면:

```
{"country": "프랑스"}
    ↓ prompt.invoke()
ChatPromptValue(messages=[HumanMessage("프랑스의 수도는 어디인가요? ...")])
    ↓ model.invoke()
AIMessage(content="프랑스의 수도는 파리(Paris)입니다.")
    ↓ parser.invoke()
"프랑스의 수도는 파리(Paris)입니다."
```

### 개념 4: stream — 실시간 스트리밍

> 💡 **비유**: 영화를 다운로드 완료 후 보는 것과 스트리밍으로 즉시 보는 것의 차이를 생각해 보세요. `invoke`가 전체 응답을 기다리는 "다운로드"라면, `stream`은 생성되는 즉시 받아보는 "스트리밍"입니다. ChatGPT가 글자를 하나씩 출력하는 것도 바로 이 스트리밍이에요.

```python
# stream — 토큰 단위로 실시간 출력
chain = prompt | model | parser

for chunk in chain.stream({"country": "일본"}):
    print(chunk, end="", flush=True)
# 출력 (한 글자씩 점진적으로 나타남):
# 일본의 수도는 도쿄(Tokyo)입니다.
print()  # 줄바꿈
```

스트리밍은 사용자 경험에서 매우 중요합니다. LLM의 응답이 수 초 걸릴 수 있는데, `invoke`를 쓰면 사용자는 그 동안 빈 화면만 보게 되거든요. `stream`을 쓰면 첫 번째 토큰이 생성되는 즉시 화면에 표시되어, 체감 대기 시간이 크게 줄어듭니다.

LCEL 체인에서 스트리밍이 특히 강력한 이유는, 체인 내의 **각 컴포넌트가 자동으로 스트리밍을 지원**하기 때문입니다. 프롬프트와 파서는 입력을 그대로 통과시키고, 모델만 토큰 단위로 스트리밍하면 전체 체인이 자연스럽게 스트리밍됩니다.

### 개념 5: batch — 대량 처리의 효율성

여러 입력을 한꺼번에 처리해야 할 때 `batch`를 사용합니다. 내부적으로 **스레드 풀**을 사용하여 병렬 실행하므로 순차 `invoke`보다 훨씬 빠릅니다:

```python
# batch — 여러 입력을 병렬로 처리
countries = [
    {"country": "한국"},
    {"country": "미국"},
    {"country": "독일"},
    {"country": "브라질"},
]

results = chain.batch(countries)
for country, result in zip(countries, results):
    print(f"{country['country']}: {result}")
# 출력:
# 한국: 한국의 수도는 서울(Seoul)입니다.
# 미국: 미국의 수도는 워싱턴 D.C.(Washington, D.C.)입니다.
# 독일: 독일의 수도는 베를린(Berlin)입니다.
# 브라질: 브라질의 수도는 브라질리아(Brasília)입니다.
```

`max_concurrency` 파라미터로 동시 실행 수를 제한할 수도 있어요. API 호출 제한(rate limit)이 있을 때 유용합니다:

```python
# 동시에 최대 2개만 실행
results = chain.batch(countries, config={"max_concurrency": 2})
```

### 개념 6: 비동기 메서드 — ainvoke, astream, abatch

웹 서버나 비동기 환경에서는 `async/await` 패턴이 필수인데요, LCEL 체인은 별도 코드 변경 없이 비동기를 지원합니다:

```python
import asyncio

async def main():
    chain = prompt | model | parser
    
    # ainvoke — 비동기 단일 실행
    result = await chain.ainvoke({"country": "캐나다"})
    print(result)
    
    # astream — 비동기 스트리밍
    async for chunk in chain.astream({"country": "호주"}):
        print(chunk, end="", flush=True)
    print()
    
    # abatch — 비동기 배치
    results = await chain.abatch([
        {"country": "이탈리아"},
        {"country": "스페인"},
    ])
    for r in results:
        print(r)

asyncio.run(main())
```

> 🔥 **실무 팁**: FastAPI 같은 비동기 웹 프레임워크에서는 반드시 `ainvoke`, `astream`을 사용하세요. 동기 `invoke`를 쓰면 이벤트 루프가 블로킹되어 다른 요청을 처리할 수 없게 됩니다. 나중에 [Ch17: LangServe와 API 배포]에서 이 패턴을 자세히 다룹니다.

## 실습: 직접 해보기

지금까지 배운 내용을 종합하여, **다국어 번역 체인**을 만들어 봅시다. invoke, stream, batch를 모두 활용합니다.

```python
"""
LCEL 기초 실습: 다국어 번역 체인
- invoke, stream, batch 통합 인터페이스 활용
- 파이프 연산자로 체인 구성
"""
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 1단계: 컴포넌트 준비
# ============================================================

# 번역 프롬프트 — 원문 언어를 자동 감지하여 대상 언어로 번역
translate_prompt = ChatPromptTemplate.from_template(
    "다음 텍스트를 {target_language}로 번역해주세요. "
    "번역문만 출력하고 다른 설명은 하지 마세요.\n\n"
    "텍스트: {text}"
)

# 모델 — 번역 작업이므로 temperature를 낮게 설정
model = ChatOpenAI(model="gpt-4o", temperature=0.3)

# 파서 — 문자열 출력
parser = StrOutputParser()

# ============================================================
# 2단계: LCEL로 체인 구성 (단 한 줄!)
# ============================================================
translate_chain = translate_prompt | model | parser

# 체인이 RunnableSequence인지 확인
print(f"체인 타입: {type(translate_chain).__name__}")
# 출력: 체인 타입: RunnableSequence

print(f"체인 구성 요소: {len(translate_chain.steps)}개")
# 출력: 체인 구성 요소: 3개

# ============================================================
# 3단계: invoke — 단일 번역
# ============================================================
print("\n=== invoke: 단일 번역 ===")
result = translate_chain.invoke({
    "text": "인공지능은 우리의 삶을 변화시키고 있습니다.",
    "target_language": "영어"
})
print(f"번역 결과: {result}")
# 출력: 번역 결과: Artificial intelligence is changing our lives.

# ============================================================
# 4단계: stream — 스트리밍 번역 (긴 텍스트에 유용)
# ============================================================
print("\n=== stream: 스트리밍 번역 ===")
print("번역 중: ", end="")
for chunk in translate_chain.stream({
    "text": (
        "LangChain은 대규모 언어 모델을 활용한 애플리케이션 개발을 "
        "쉽게 만들어주는 오픈소스 프레임워크입니다. "
        "다양한 컴포넌트를 조합하여 복잡한 AI 파이프라인을 구축할 수 있습니다."
    ),
    "target_language": "일본어"
}):
    print(chunk, end="", flush=True)
print()  # 줄바꿈

# ============================================================
# 5단계: batch — 대량 번역 (병렬 처리)
# ============================================================
print("\n=== batch: 대량 번역 ===")
texts_to_translate = [
    {"text": "안녕하세요, 반갑습니다!", "target_language": "영어"},
    {"text": "오늘 날씨가 좋습니다.", "target_language": "스페인어"},
    {"text": "파이썬은 배우기 쉬운 언어입니다.", "target_language": "프랑스어"},
    {"text": "커피 한 잔 주세요.", "target_language": "중국어"},
]

# 순차 실행 시간 측정
start = time.time()
sequential_results = [translate_chain.invoke(t) for t in texts_to_translate]
sequential_time = time.time() - start
print(f"순차 실행 시간: {sequential_time:.2f}초")

# batch 실행 시간 측정
start = time.time()
batch_results = translate_chain.batch(texts_to_translate)
batch_time = time.time() - start
print(f"배치 실행 시간: {batch_time:.2f}초")
print(f"속도 향상: {sequential_time / batch_time:.1f}배\n")

# 번역 결과 출력
for text_info, result in zip(texts_to_translate, batch_results):
    print(f"  [{text_info['target_language']}] {result}")

# ============================================================
# 6단계: 입출력 스키마 확인 — 체인의 타입 정보
# ============================================================
print("\n=== 체인 스키마 확인 ===")
print(f"입력 스키마: {translate_chain.input_schema.model_json_schema()}")
# 출력: {'properties': {'text': {'title': 'Text', 'type': 'string'},
#         'target_language': {'title': 'Target Language', 'type': 'string'}},
#        'required': ['text', 'target_language'], ...}

print(f"출력 스키마: {translate_chain.output_schema.model_json_schema()}")
# 출력: {'title': 'StrOutputParserOutput', 'type': 'string'}
```

## 더 깊이 알아보기

### LCEL의 탄생 — "체인을 더 쉽게 만들 수는 없을까?"

LangChain의 창시자 **Harrison Chase**는 2022년 10월, 머신러닝 스타트업 Robust Intelligence에서 일하던 중 첫 코드를 오픈소스로 공개했습니다. 그의 첫 경험은 회사 내부의 Notion과 Slack 데이터를 LLM으로 질의하는 것이었는데, 이것이 나중에 **RAG(Retrieval-Augmented Generation)**라 불리게 될 패턴의 원형이었죠.

초기 LangChain은 `LLMChain`, `SequentialChain`, `TransformChain` 등 각기 다른 클래스를 사용하여 체인을 구성했습니다. 문제는 체인 종류마다 API가 달랐고, 스트리밍이나 배치 처리를 추가하려면 상당한 보일러플레이트 코드가 필요했다는 점이에요.

2023년 3분기, LangChain 팀은 **LCEL**을 도입하며 이 문제를 근본적으로 해결했습니다. 핵심 아이디어는 두 가지였습니다:

1. **Runnable 프로토콜**: 모든 컴포넌트가 동일한 인터페이스를 구현하게 하여, 어떤 조합이든 자동으로 동작하게 만들자
2. **파이프 연산자**: Unix 셸의 파이프에서 영감을 받아, 체인 구성을 코드가 아닌 **선언(declaration)**으로 표현하자

이 설계는 **함수형 프로그래밍**의 함수 합성(function composition)과 Unix 철학의 "하나의 일을 잘하는 작은 프로그램을 파이프로 연결한다"에서 영감을 받았습니다.

### Python의 `__or__` 마법

파이프 연산자가 동작하는 비밀은 Python의 **던더 메서드(dunder method)**에 있습니다. Python에서 `a | b`를 쓰면 인터프리터는 `a.__or__(b)`를 호출합니다. LangChain의 `Runnable` 기본 클래스에는 이 `__or__` 메서드가 다음과 같은 로직으로 구현되어 있습니다:

```python
# langchain_core/runnables/base.py 에서 (간소화)
class Runnable:
    def __or__(self, other):
        return RunnableSequence(self, other)
    
    def __ror__(self, other):
        return RunnableSequence(other, self)
```

`__ror__`는 왼쪽 피연산자가 `__or__`를 지원하지 않을 때 호출되는데, 이 덕분에 일반 딕셔너리나 함수도 파이프라인에 참여할 수 있습니다. 이 우아한 설계 덕분에 LCEL은 "미니멀한 코드로 최대한의 기능을"이라는 목표를 달성했죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "LCEL 체인은 `|`를 쓰면 바로 실행된다." — 아닙니다! `|`는 체인 **객체를 생성(구성)**할 뿐, 데이터가 흐르지는 않습니다. `invoke()`, `stream()`, `batch()` 중 하나를 호출해야 비로소 실행됩니다. 이는 SQLAlchemy의 쿼리 빌더가 `.all()`을 호출해야 실행되는 것과 같은 원리예요.

> 💡 **알고 계셨나요?**: LCEL의 `batch()` 메서드는 기본적으로 `ThreadPoolExecutor`를 사용하여 병렬 실행합니다. 즉, 4개의 입력을 `batch`하면 4개의 `invoke`가 동시에 실행되어, 순차 처리 대비 최대 4배 빠를 수 있습니다. 단, LLM API의 rate limit에 주의하세요!

> 🔥 **실무 팁**: 디버깅 시 체인의 중간 결과를 확인하고 싶다면, 체인을 분할하여 각 단계를 따로 `invoke`하세요. `prompt.invoke({"topic": "AI"})` → `model.invoke(prompt_result)` → `parser.invoke(model_result)` 순으로 호출하면 어느 단계에서 문제가 발생하는지 쉽게 찾을 수 있습니다.

> 🔥 **실무 팁**: `chain.input_schema`와 `chain.output_schema`를 활용하면 체인이 기대하는 입력/출력 형태를 코드로 확인할 수 있습니다. 특히 복잡한 체인에서 "어떤 키를 넘겨야 하지?" 고민될 때 유용해요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **Runnable** | LangChain의 모든 실행 가능 컴포넌트가 구현하는 프로토콜. `invoke`, `stream`, `batch` 등 통합 인터페이스 제공 |
| **파이프 연산자(`\|`)** | 두 Runnable을 `RunnableSequence`로 연결하는 연산자. Python의 `__or__` 메서드로 구현 |
| **RunnableSequence** | `\|` 연산자로 생성되는 순차 실행 체인. 첫 컴포넌트의 출력이 다음의 입력이 됨 |
| **invoke** | 단일 입력 → 단일 출력. 가장 기본적인 실행 방식 |
| **stream** | 출력을 청크 단위로 실시간 전달. 사용자 체감 대기 시간 감소 |
| **batch** | 여러 입력을 병렬로 처리. `max_concurrency`로 동시 실행 수 제한 가능 |
| **비동기 메서드** | `ainvoke`, `astream`, `abatch` — 비동기 환경(FastAPI 등)에서 필수 |
| **.pipe() 메서드** | `\|` 연산자와 동일한 기능. 메서드 체이닝 스타일을 선호할 때 사용 |

## 다음 섹션 미리보기

이번 섹션에서는 LCEL의 기초와 `RunnableSequence`를 통한 순차 파이프라인을 배웠습니다. 하지만 실제 애플리케이션에서는 "원본 입력을 유지하면서 중간 처리도 하고 싶다"거나, "일반 Python 함수를 체인에 끼워 넣고 싶다"는 요구가 생기는데요. 다음 섹션 **[5.2: RunnablePassthrough와 RunnableLambda]**에서는 입력을 그대로 통과시키는 `RunnablePassthrough`와 임의의 함수를 Runnable로 감싸는 `RunnableLambda`를 배웁니다. 이 두 도구를 활용하면 LCEL 체인의 표현력이 크게 확장됩니다.

## 참고 자료

- [LangChain Expression Language (LCEL) — 공식 개념 문서](https://python.langchain.com/docs/concepts/lcel/) - LCEL의 설계 철학과 핵심 개념을 다루는 공식 레퍼런스
- [Runnable API Reference — LangChain 공식 문서](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html) - Runnable 클래스의 전체 메서드와 파라미터를 확인할 수 있는 API 문서
- [LangChain Expression Language Explained — Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/) - LCEL을 실습 중심으로 설명하는 튜토리얼
- [LangChain Expression Language (LCEL) — Aurelio AI](https://www.aurelio.ai/learn/langchain-lcel) - LCEL의 기초부터 고급 패턴까지 다루는 가이드
- [Why the Pipe Character "|" Works in LangChain's LCEL — Medium](https://medium.com/@MichaelHashimoto/why-the-pipe-character-works-in-langchains-lcel-b4e8685855f5) - 파이프 연산자의 Python 내부 동작 원리를 깊이 분석한 글
- [Reflections on Three Years of Building LangChain — LangChain Blog](https://blog.langchain.com/three-years-langchain/) - Harrison Chase가 LangChain의 3년 역사를 되돌아보는 블로그 포스트

---
### 🔗 Related Sessions
- [langchain](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [prompt_template](../03-프롬프트-엔지니어링과-템플릿/01-chatprompttemplate-기초.md) (prerequisite)
- [chain](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [output_parser](../04-출력-파서와-구조화된-출력/01-출력-파서-기초.md) (prerequisite)
