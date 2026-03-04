# Few-shot 프롬프팅

> LLM에게 "이렇게 해줘"라고 보여주는 가장 강력한 방법 — FewShotChatMessagePromptTemplate과 동적 예제 선택 전략

## 개요

이 섹션에서는 LangChain의 `FewShotChatMessagePromptTemplate`을 사용해 Few-shot 프롬프트를 체계적으로 구성하는 방법을 배웁니다. 고정 예제부터 시작하여, `SemanticSimilarityExampleSelector`로 입력에 맞는 예제를 자동으로 골라주는 동적 선택까지 다룹니다.

**선수 지식**: [Session 3.1: ChatPromptTemplate 기초](./01-chatprompttemplate-기초.md)에서 배운 `ChatPromptTemplate.from_messages()`와 `invoke()`, [Session 3.2: 고급 프롬프트 패턴](./02-고급-프롬프트-패턴.md)에서 배운 `MessagesPlaceholder`

**학습 목표**:
- `FewShotChatMessagePromptTemplate`으로 Few-shot 프롬프트를 구성할 수 있다
- `SemanticSimilarityExampleSelector`로 입력에 따라 예제를 동적으로 선택할 수 있다
- 예제 수(`k`)를 상황에 맞게 조절하는 전략을 세울 수 있다

## 왜 알아야 할까?

여러분이 후배에게 코드 리뷰를 가르친다고 상상해 보세요. "코드 리뷰 잘 해"라고만 말하면, 후배는 뭘 어떻게 해야 할지 막막하겠죠? 하지만 실제 코드 리뷰 예시 3개를 보여주면 — "아, 이런 식으로 하면 되는구나!" — 금방 패턴을 파악합니다.

LLM도 마찬가지입니다. 단순히 "번역해줘", "요약해줘"라고 지시하는 것(Zero-shot)보다, **몇 개의 예시를 보여주는 것(Few-shot)**이 훨씬 정확하고 일관된 결과를 만들어내거든요. 특히 다음과 같은 상황에서 Few-shot이 빛을 발합니다:

- **일관된 출력 형식**이 필요할 때 (예: 특정 JSON 구조)
- **도메인 특화 용어**를 사용해야 할 때 (예: 법률, 의학 번역)
- **분류 기준**이 미묘할 때 (예: 감정 분석의 "중립" vs "약간 긍정")
- **스타일이나 톤**을 맞춰야 할 때 (예: 브랜드 카피라이팅)

문제는, 예제를 하드코딩하면 모든 입력에 같은 예제가 들어간다는 겁니다. "한국어→영어 번역"을 요청하는데 "감정 분석" 예제가 들어가면 의미가 없겠죠? LangChain은 이 문제를 **동적 예제 선택(Dynamic Example Selection)**으로 해결합니다.

## 핵심 개념

### 개념 1: FewShotChatMessagePromptTemplate — 예제를 체계적으로 관리하기

> 💡 **비유**: 요리 레시피북을 생각해 보세요. "파스타를 만들어줘"라고만 하면 어떤 파스타인지 모르지만, 완성된 요리 사진 3장과 레시피를 보여주면 정확히 원하는 파스타를 만들 수 있습니다. `FewShotChatMessagePromptTemplate`은 이 "레시피 모음집"을 LLM에게 체계적으로 전달하는 도구입니다.

`FewShotChatMessagePromptTemplate`은 Few-shot 예제들을 채팅 메시지 형태로 변환해주는 프롬프트 템플릿입니다. 핵심 구조는 이렇습니다:

1. **예제 데이터**: 딕셔너리 리스트로 된 입출력 쌍
2. **예제 프롬프트**: 각 예제를 어떤 메시지 형태로 변환할지 정의
3. **Few-shot 템플릿**: 예제 프롬프트 + 예제 데이터를 합쳐서 메시지 시퀀스 생성

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# 1. 예제 데이터 준비
examples = [
    {"input": "안녕하세요", "output": "Hello"},
    {"input": "감사합니다", "output": "Thank you"},
    {"input": "좋은 아침입니다", "output": "Good morning"},
]

# 2. 각 예제가 어떤 메시지로 변환될지 정의
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),   # 사용자 메시지
    ("ai", "{output}"),     # AI 응답
])

# 3. Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. 최종 프롬프트에 통합
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 한국어를 영어로 번역하는 전문 번역가입니다."),
    few_shot_prompt,                    # Few-shot 예제들이 여기에 삽입
    ("human", "{input}"),               # 실제 사용자 입력
])

# 결과 확인
messages = final_prompt.invoke({"input": "반갑습니다"})
for msg in messages.messages:
    print(f"[{msg.type}] {msg.content}")
```

```
# 출력:
# [system] 당신은 한국어를 영어로 번역하는 전문 번역가입니다.
# [human] 안녕하세요
# [ai] Hello
# [human] 감사합니다
# [ai] Thank you
# [human] 좋은 아침입니다
# [ai] Good morning
# [human] 반갑습니다
```

보이시나요? 3개의 예제가 `human`-`ai` 메시지 쌍으로 변환되어 실제 입력 앞에 자연스럽게 삽입됩니다. LLM은 이 패턴을 보고 "아, 한국어가 들어오면 영어로 번역하면 되는구나"라고 학습하는 거죠.

### 개념 2: SemanticSimilarityExampleSelector — 입력에 맞는 예제를 자동으로 골라주기

> 💡 **비유**: 도서관 사서를 떠올려 보세요. "양자역학에 대해 알고 싶어요"라고 하면 물리학 코너에서 관련 책을 골라주고, "셰익스피어 작품을 찾아요"라고 하면 문학 코너에서 책을 골라줍니다. `SemanticSimilarityExampleSelector`는 바로 이런 "AI 사서" 역할을 합니다 — 사용자의 질문과 **의미적으로 가장 가까운** 예제들을 자동으로 선택해주거든요.

고정 예제의 한계는 명확합니다. 예제가 100개인데 매번 100개를 다 넣으면 토큰(Token)도 낭비되고, 오히려 관련 없는 예제가 노이즈가 됩니다. `SemanticSimilarityExampleSelector`는 임베딩(Embedding) 기반으로 입력과 가장 유사한 예제 `k`개만 선별합니다.

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 다양한 도메인의 예제 준비
examples = [
    {"input": "주가가 급등했다", "output": "Stock prices surged"},
    {"input": "환율이 하락했다", "output": "Exchange rates declined"},
    {"input": "배가 아프다", "output": "I have a stomachache"},
    {"input": "두통이 심하다", "output": "I have a severe headache"},
    {"input": "소스 코드를 커밋하다", "output": "Commit the source code"},
    {"input": "버그를 수정하다", "output": "Fix the bug"},
]

# SemanticSimilarityExampleSelector 생성
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                    # 예제 리스트
    OpenAIEmbeddings(),          # 임베딩 모델
    FAISS,                       # 벡터 스토어 클래스
    k=2,                         # 선택할 예제 수
)

# 예제 프롬프트 정의
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Few-shot 프롬프트에 example_selector 연결
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,  # examples 대신 example_selector 사용!
    input_variables=["input"],          # 선택 기준이 되는 변수
)

# 테스트: 금융 관련 입력
selected = example_selector.select_examples({"input": "금리가 인상되었다"})
print("금융 입력 → 선택된 예제:")
for ex in selected:
    print(f"  {ex['input']} → {ex['output']}")

# 테스트: 의료 관련 입력
selected = example_selector.select_examples({"input": "열이 나고 기침이 난다"})
print("\n의료 입력 → 선택된 예제:")
for ex in selected:
    print(f"  {ex['input']} → {ex['output']}")
```

```
# 출력 (유사도 기반 선택):
# 금융 입력 → 선택된 예제:
#   주가가 급등했다 → Stock prices surged
#   환율이 하락했다 → Exchange rates declined
#
# 의료 입력 → 선택된 예제:
#   배가 아프다 → I have a stomachache
#   두통이 심하다 → I have a severe headache
```

놀랍지 않나요? "금리 인상"을 입력하면 금융 관련 예제가, "열과 기침"을 입력하면 의료 관련 예제가 자동으로 선택됩니다. 이것이 가능한 이유는 임베딩 공간에서 의미적으로 가까운 벡터를 코사인 유사도(Cosine Similarity)로 찾기 때문이에요.

### 개념 3: 동적 예제 수 조절 — k 파라미터 전략

> 💡 **비유**: 시험공부할 때 기출문제를 푸는 것과 비슷합니다. 너무 적으면(k=1) 패턴 파악이 어렵고, 너무 많으면(k=20) 시간만 낭비되죠. 적절한 수(k=3~5)를 풀어야 효율이 가장 좋습니다.

`k` 파라미터는 선택할 예제의 수를 결정합니다. 이 값은 상황에 따라 전략적으로 조절해야 합니다:

| 상황 | 권장 k | 이유 |
|------|--------|------|
| 단순 형식 변환 | 1~2 | 패턴이 명확해서 적은 예제로 충분 |
| 분류/감정 분석 | 3~5 | 클래스별 최소 1개씩 보여줘야 함 |
| 복잡한 추론 | 2~3 | 긴 예제가 많으면 토큰 낭비 |
| 스타일 모방 | 4~6 | 톤과 스타일을 충분히 보여줘야 함 |

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 감정 분석 예제
sentiment_examples = [
    {"text": "이 제품 정말 최고예요!", "sentiment": "긍정"},
    {"text": "배송이 빨라서 좋았습니다", "sentiment": "긍정"},
    {"text": "가격 대비 별로입니다", "sentiment": "부정"},
    {"text": "다시는 안 살 거예요", "sentiment": "부정"},
    {"text": "그냥 그래요, 보통이에요", "sentiment": "중립"},
    {"text": "나쁘지는 않은데 특별하지도 않아요", "sentiment": "중립"},
]

# k=3: 긍정/부정/중립 각 1개씩 선택될 가능성
selector = SemanticSimilarityExampleSelector.from_examples(
    sentiment_examples,
    OpenAIEmbeddings(),
    FAISS,
    k=3,                          # 클래스별 1개씩 보여줄 수 있는 최소치
)

# 새 예제를 동적으로 추가할 수도 있음
selector.add_example(
    {"text": "기대 이상이에요, 강추합니다!", "sentiment": "긍정"}
)
```

여기서 중요한 점이 있습니다. `k`를 높이면 정확도는 올라갈 수 있지만, **토큰 비용과 지연 시간도 함께 증가**합니다. 프로덕션 환경에서는 정확도와 비용 사이의 균형점을 실험적으로 찾아야 해요.

### 개념 4: 다양한 예제 선택 전략

`SemanticSimilarityExampleSelector` 외에도 LangChain은 상황별로 쓸 수 있는 선택 전략을 제공합니다:

**LengthBasedExampleSelector** — 토큰 제한이 걱정될 때:

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate

# 예제의 총 길이가 max_length를 넘지 않도록 자동 조절
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="입력: {input}\n출력: {output}",
)

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100,              # 단어 수 기준 최대 길이
)
```

**MaxMarginalRelevanceExampleSelector** — 관련성과 다양성을 동시에 원할 때:

```python
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# MMR: 유사하면서도 서로 다른 예제를 선택
mmr_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=3,
    # fetch_k=10,  # 후보를 먼저 10개 뽑고, 그 중 다양성 고려하여 3개 선택
)
```

앞서 [Session 3.2: 고급 프롬프트 패턴](./02-고급-프롬프트-패턴.md)에서 배운 `RunnableBranch`와 결합하면, 입력 유형에 따라 다른 예제 선택 전략을 적용하는 것도 가능합니다.

## 실습: 직접 해보기

고객 리뷰를 분석하는 Few-shot 시스템을 처음부터 끝까지 만들어 보겠습니다. 고정 예제 방식과 동적 선택 방식을 모두 구현합니다.

```python
"""
Few-shot 프롬프팅 실습: 고객 리뷰 분석 시스템
- 고정 예제 vs 동적 예제 선택 비교
- SemanticSimilarityExampleSelector를 활용한 도메인 적응
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── 상수 정의 ──
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.3

# ── 1단계: 예제 데이터 준비 ──
# 다양한 카테고리의 리뷰-분석 쌍
review_examples = [
    {
        "review": "화면이 선명하고 배터리도 오래 가요. 가격도 합리적이라 매우 만족합니다.",
        "analysis": "감정: 긍정 | 카테고리: 전자제품 | 핵심키워드: 화면, 배터리, 가격 | 요약: 디스플레이 품질과 배터리 수명, 가성비에 만족"
    },
    {
        "review": "배송은 빨랐는데 포장이 엉망이어서 제품이 찌그러져 왔어요.",
        "analysis": "감정: 부정 | 카테고리: 배송/포장 | 핵심키워드: 배송, 포장, 파손 | 요약: 빠른 배송에도 불구하고 포장 불량으로 제품 손상"
    },
    {
        "review": "맛은 괜찮은데 양이 좀 적어요. 가격을 생각하면 보통입니다.",
        "analysis": "감정: 중립 | 카테고리: 식품 | 핵심키워드: 맛, 양, 가격 | 요약: 맛은 무난하나 양 대비 가격이 아쉬움"
    },
    {
        "review": "원단이 부드럽고 핏도 좋아요. 세탁해도 줄어들지 않아서 재구매했습니다.",
        "analysis": "감정: 긍정 | 카테고리: 의류 | 핵심키워드: 원단, 핏, 세탁 | 요약: 소재 품질과 내구성에 만족하여 재구매"
    },
    {
        "review": "설치가 너무 복잡하고 설명서도 부실해요. 고객센터 전화도 안 받아요.",
        "analysis": "감정: 부정 | 카테고리: 서비스/설치 | 핵심키워드: 설치, 설명서, 고객센터 | 요약: 설치 난이도와 고객 지원 부재에 불만"
    },
    {
        "review": "디자인은 예쁜데 실용성이 떨어져요. 수납공간이 부족합니다.",
        "analysis": "감정: 중립 | 카테고리: 가구/인테리어 | 핵심키워드: 디자인, 실용성, 수납 | 요약: 외관은 좋으나 기능적 한계가 있음"
    },
]

# ── 2단계: 예제 프롬프트 정의 ──
# 각 예제가 human-ai 메시지 쌍으로 변환됨
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{review}"),
    ("ai", "{analysis}"),
])

# ── 3단계: 방법 A — 고정 예제 방식 ──
print("=" * 60)
print("방법 A: 고정 예제 (모든 입력에 동일한 예제 사용)")
print("=" * 60)

fixed_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=review_examples[:3],     # 처음 3개만 고정 사용
)

fixed_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 고객 리뷰를 분석하는 전문가입니다. "
               "주어진 예시의 형식에 맞춰 분석 결과를 출력하세요."),
    fixed_few_shot,
    ("human", "{review}"),
])

# 체인 구성 (LCEL 파이프 연산자 사용)
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
fixed_chain = fixed_prompt | llm | StrOutputParser()

# 테스트
test_review = "이 운동화 쿠션감이 정말 좋아요. 장시간 걸어도 발이 편합니다."
result_fixed = fixed_chain.invoke({"review": test_review})
print(f"\n입력: {test_review}")
print(f"결과: {result_fixed}")

# ── 4단계: 방법 B — 동적 예제 선택 방식 ──
print("\n" + "=" * 60)
print("방법 B: 동적 예제 선택 (입력과 유사한 예제 자동 선택)")
print("=" * 60)

# SemanticSimilarityExampleSelector 생성
example_selector = SemanticSimilarityExampleSelector.from_examples(
    review_examples,              # 전체 예제 풀
    OpenAIEmbeddings(),           # 임베딩 모델
    FAISS,                        # 벡터 스토어
    k=2,                          # 가장 유사한 2개 선택
    input_keys=["review"],        # 'review' 필드 기준으로 유사도 계산
)

dynamic_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["review"],
)

dynamic_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 고객 리뷰를 분석하는 전문가입니다. "
               "주어진 예시의 형식에 맞춰 분석 결과를 출력하세요."),
    dynamic_few_shot,
    ("human", "{review}"),
])

dynamic_chain = dynamic_prompt | llm | StrOutputParser()

# 테스트 1: 의류 관련 리뷰
test_review_1 = "이 운동화 쿠션감이 정말 좋아요. 장시간 걸어도 발이 편합니다."
result_1 = dynamic_chain.invoke({"review": test_review_1})
print(f"\n입력: {test_review_1}")
print(f"결과: {result_1}")

# 어떤 예제가 선택되었는지 확인
selected = example_selector.select_examples({"review": test_review_1})
print(f"\n선택된 예제:")
for ex in selected:
    print(f"  → {ex['review'][:30]}...")

# 테스트 2: 음식 관련 리뷰 (다른 예제가 선택됨)
test_review_2 = "라면 국물이 진하고 면발도 쫄깃해요. 가성비 최고!"
result_2 = dynamic_chain.invoke({"review": test_review_2})
print(f"\n입력: {test_review_2}")
print(f"결과: {result_2}")

selected = example_selector.select_examples({"review": test_review_2})
print(f"\n선택된 예제:")
for ex in selected:
    print(f"  → {ex['review'][:30]}...")

# ── 5단계: 예제 동적 추가 ──
print("\n" + "=" * 60)
print("새 예제 추가 후 재테스트")
print("=" * 60)

# 운동/스포츠 관련 예제 추가
example_selector.add_example({
    "review": "러닝할 때 착용감이 가볍고 통기성이 좋아요.",
    "analysis": "감정: 긍정 | 카테고리: 스포츠용품 | 핵심키워드: 착용감, 통기성 | 요약: 운동 시 착용감과 통기성에 만족"
})

# 다시 테스트 — 이제 스포츠 관련 예제도 후보에 포함됨
selected = example_selector.select_examples({"review": test_review_1})
print(f"\n'{test_review_1[:20]}...'에 대해 선택된 예제:")
for ex in selected:
    print(f"  → {ex['review'][:40]}...")
```

## 더 깊이 알아보기

### Few-shot 프롬프팅의 탄생 — GPT-3와 "인컨텍스트 학습"의 발견

Few-shot 프롬프팅이라는 개념이 세상을 놀라게 한 건 2020년, OpenAI가 GPT-3 논문 *"Language Models are Few-Shot Learners"*를 발표했을 때입니다. Tom Brown을 포함한 31명의 저자가 참여한 이 논문에서, 1,750억 개의 파라미터를 가진 GPT-3가 **파인튜닝 없이** 몇 개의 예시만으로도 번역, 질문 응답, 산술 연산 등을 수행할 수 있음을 보여줬습니다.

이전까지 머신러닝에서 "Few-shot Learning"은 주로 이미지 인식 분야에서 연구되던 개념이었어요. 적은 수의 이미지 샘플로 새로운 클래스를 인식하는 것이죠. 그런데 GPT-3 팀은 **텍스트 프롬프트 안에 예시를 넣는 것**만으로 모델이 새로운 작업을 학습할 수 있다는 것을 발견했고, 이를 **인컨텍스트 학습(In-Context Learning, ICL)**이라고 명명했습니다.

흥미로운 점은, 이것이 의도된 기능이라기보다는 **모델 규모를 키우다 보니 자연스럽게 발현된 능력(Emergent Ability)**이라는 것입니다. 작은 모델에서는 나타나지 않다가, 파라미터 수가 일정 임계점을 넘자 갑자기 나타났거든요. 왜 이런 능력이 생기는지는 아직 활발히 연구되고 있는 열린 질문입니다.

### 예제 순서가 결과를 바꾼다?

2021년 Zhao 등이 발표한 논문 *"Calibrate Before Use"*에서 충격적인 사실이 밝혀졌습니다. Few-shot 예제의 **순서**를 바꾸는 것만으로 모델의 정확도가 거의 0%에서 거의 100%까지 변할 수 있다는 것이었죠. 같은 예제를 사용하더라도 배치 순서에 따라 결과가 극적으로 달라진다니, 놀랍지 않나요? 이 연구는 예제 선택뿐 아니라 **예제 정렬(ordering)**까지 고려해야 한다는 중요한 교훈을 남겼습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Few-shot 예제는 많을수록 좋다"
> 반드시 그렇지 않습니다. 예제가 너무 많으면 (1) 토큰 비용이 급증하고, (2) 관련 없는 예제가 노이즈로 작용하며, (3) 실제 입력이 컨텍스트 윈도우 끝에 밀려나 성능이 오히려 떨어질 수 있습니다. 대부분의 작업에서 3~5개의 잘 선별된 예제가 20개의 무작위 예제보다 효과적입니다.

> 💡 **알고 계셨나요?**: `FewShotChatMessagePromptTemplate`에서 `examples`와 `example_selector`를 **동시에** 지정하면 에러가 발생합니다. 둘 중 하나만 사용해야 합니다. 고정 예제를 쓸 때는 `examples`, 동적 선택을 쓸 때는 `example_selector`를 지정하세요.

> 🔥 **실무 팁**: `SemanticSimilarityExampleSelector`를 사용할 때, `input_keys` 파라미터를 지정하면 예제의 특정 필드만 유사도 계산에 사용됩니다. 예를 들어 예제에 `input`, `output`, `context` 필드가 있을 때 `input_keys=["input"]`으로 설정하면 `input` 필드만으로 유사도를 계산합니다. 출력 필드가 유사도 계산에 영향을 주는 것을 방지할 수 있어요.

> 🔥 **실무 팁**: 프로덕션에서는 벡터 스토어를 영속적으로 저장해 두세요. `FAISS`를 사용한다면 `vectorstore.save_local("examples_index")`로 저장하고, 서버 시작 시 `FAISS.load_local("examples_index", embeddings)`로 불러오면 매번 임베딩을 다시 계산하는 비용을 절약할 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `FewShotChatMessagePromptTemplate` | Few-shot 예제를 채팅 메시지 시퀀스로 변환하는 프롬프트 템플릿 |
| `example_prompt` | 각 예제를 어떤 메시지 형태로 변환할지 정의하는 `ChatPromptTemplate` |
| `examples` | 고정된 예제 리스트 (딕셔너리 리스트) |
| `example_selector` | 입력에 따라 동적으로 예제를 선택하는 컴포넌트 |
| `SemanticSimilarityExampleSelector` | 임베딩 유사도 기반으로 가장 관련 높은 예제를 선택 |
| `MaxMarginalRelevanceExampleSelector` | 관련성과 다양성을 동시에 고려하여 예제를 선택 (MMR 알고리즘) |
| `LengthBasedExampleSelector` | 프롬프트 길이 제한을 고려하여 예제 수를 자동 조절 |
| `k` 파라미터 | 선택할 예제의 수 — 작업 복잡도와 토큰 비용의 균형점을 찾아야 함 |
| `input_keys` | 유사도 계산에 사용할 예제 필드를 지정하는 파라미터 |
| `add_example()` | 런타임에 예제 풀에 새로운 예제를 추가하는 메서드 |

## 다음 섹션 미리보기

이번 섹션에서 Few-shot 프롬프팅으로 LLM에게 "이렇게 해줘"라는 패턴을 보여주는 방법을 배웠습니다. 다음 섹션 **[Session 3.4: 프롬프트 합성과 파이프라인](./04-프롬프트-관리와-langchain-hub.md)**에서는 여러 프롬프트를 조합하고 단계별로 연결하는 프롬프트 파이프라인을 다룹니다. 프롬프트 재사용성을 극대화하고, 복잡한 작업을 작은 프롬프트 단위로 분해하는 전략을 배울 거예요.

## 참고 자료

- [How to use few shot examples in chat models — LangChain 공식 문서](https://python.langchain.com/docs/how_to/few_shot_examples_chat/) - FewShotChatMessagePromptTemplate의 사용법을 단계별로 설명하는 공식 가이드
- [How to select examples by similarity — LangChain 공식 문서](https://python.langchain.com/docs/how_to/example_selectors_similarity/) - SemanticSimilarityExampleSelector의 활용법과 벡터 스토어 연동 방법
- [FewShotChatMessagePromptTemplate API Reference](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate.html) - 클래스의 전체 파라미터와 메서드를 확인할 수 있는 API 레퍼런스
- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) - Few-shot 프롬프팅의 개념을 정립한 GPT-3 논문
- [Few-Shot Prompting — Prompt Engineering Guide](https://www.promptingguide.ai/techniques/fewshot) - Few-shot 프롬프팅의 기법과 활용 사례를 정리한 종합 가이드

---
### 🔗 Related Sessions
- [lcel](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [invoke](01-langchain-소개와-개발-환경-설정/04-첫-번째-langchain-애플리케이션.md) (prerequisite)
- [chatprompttemplate](01-langchain-소개와-개발-환경-설정/04-첫-번째-langchain-애플리케이션.md) (prerequisite)
- [messagesplaceholder](./02-고급-프롬프트-패턴.md) (prerequisite)
- [runnablebranch](./02-고급-프롬프트-패턴.md) (prerequisite)
- [embedding](07-임베딩과-벡터-스토어/01-텍스트-임베딩-이해.md) (prerequisite)
