# RAG 평가와 개선

> RAG 시스템의 성능을 정량적으로 측정하는 방법을 학습합니다. RAGAS 프레임워크의 핵심 메트릭을 이해하고, 평가 데이터셋을 구축하여 RAG 파이프라인을 체계적으로 평가합니다.

## 개요

이 섹션에서는 RAG 시스템을 "감"이 아닌 "숫자"로 평가하는 방법을 배웁니다. 먼저 수동 평가의 한계를 이해한 뒤, RAGAS 프레임워크의 핵심 메트릭(충실도, 관련성, 정밀도, 재현율)을 단계별로 학습합니다. 평가 데이터셋을 체계적으로 구축하는 방법을 익히고, 기본 RAG 체인과 개선된 RAG 체인의 성능 차이를 RAGAS로 비교하는 실습까지 진행합니다.

**선수 지식**: 앞서 [9.1 기본 RAG 체인 구축](./01-기본-rag-체인-구축.md)에서 배운 RAG 파이프라인 5단계, [9.4 고급 RAG 패턴](./04-고급-rag-패턴.md)에서 다룬 HyDE, 재랭킹, Multi-Query Retriever 등의 고급 검색 기법

**학습 목표**:
- RAG 평가가 왜 "검색"과 "생성"을 분리해서 측정해야 하는지 설명할 수 있다
- RAGAS 프레임워크의 4대 핵심 메트릭의 의미와 계산 원리를 이해할 수 있다
- 효과적인 평가 데이터셋을 설계하고 구축할 수 있다
- RAGAS를 사용하여 RAG 체인의 성능을 정량적으로 측정하고 비교할 수 있다

## 왜 알아야 할까?

"우리 RAG 시스템, 잘 동작하나요?"라는 질문에 "네, 꽤 잘 되는 것 같아요"라고 대답하고 계신가요? 프로덕션 환경에서 이런 답변은 통하지 않습니다.

실제로 RAG 시스템을 배포하면 다양한 문제가 발생합니다. 검색된 문서가 질문과 관련이 없거나, 관련 문서를 찾았는데 LLM이 엉뚱한 답변을 만들거나(할루시네이션), 답변은 정확한데 질문의 핵심을 벗어나는 경우도 있죠. 이런 문제들을 "느낌"으로 잡는 건 한계가 있습니다.

[9.4 고급 RAG 패턴](./04-고급-rag-패턴.md)에서 ParentDocumentRetriever, HyDE, Cross-Encoder 재랭킹 등 다양한 기법을 배웠는데요, 이 기법들이 **실제로 얼마나 효과가 있는지** 어떻게 알 수 있을까요? "A 방식이 B 방식보다 낫다"고 말하려면 **정량적 증거**가 필요합니다.

하지만 평가를 처음부터 복잡하게 시작할 필요는 없습니다. 이번 세션에서는 **RAGAS 프레임워크 하나에 집중**하여 RAG 평가의 기초를 탄탄히 다지겠습니다. 다음 [9.6 프로덕션 RAG 아키텍처](./06-프로덕션-rag-아키텍처.md)에서 LangSmith를 활용한 체계적 모니터링과 A/B 테스트 프레임워크를 다룰 예정이니, 이번 세션에서 배운 메트릭 개념이 그 기반이 됩니다.

## 핵심 개념

### 개념 1: RAG 평가의 두 축 — 검색 vs 생성

> 💡 **비유**: RAG 시스템을 평가하는 건 레스토랑을 평가하는 것과 비슷합니다. "재료 구매"(검색)와 "요리"(생성)를 따로 평가해야 전체 품질을 정확히 파악할 수 있죠. 신선한 재료를 사왔는데 요리를 망칠 수도 있고, 평범한 재료로 놀라운 요리를 만들 수도 있으니까요.

RAG 평가는 크게 두 영역으로 나뉩니다:

| 평가 영역 | 측정 대상 | 핵심 질문 |
|-----------|----------|----------|
| **검색 평가** (Retrieval) | 검색된 문서(Context) | "올바른 문서를 찾았는가?" |
| **생성 평가** (Generation) | 최종 답변(Response) | "답변이 정확하고 충실한가?" |

이 두 영역을 독립적으로 평가해야 하는 이유가 있습니다. 만약 최종 답변이 틀렸다면, 검색이 잘못된 건지 생성이 잘못된 건지 구분해야 올바른 수정이 가능하거든요. 검색이 문제라면 임베딩 모델이나 청킹 전략을 바꿔야 하고, 생성이 문제라면 프롬프트나 LLM을 개선해야 합니다.

```
[사용자 질문] → [검색기] → [검색된 문서들] → [LLM 생성] → [최종 답변]
                    ↑                              ↑
              검색 평가 지점                   생성 평가 지점
           (Context Precision,             (Faithfulness,
            Context Recall)               Answer Relevancy)
```

이 구조를 이해하면, 이어서 배울 RAGAS 메트릭 4가지가 왜 그렇게 설계되었는지 자연스럽게 이해할 수 있습니다.

### 개념 2: 수동 평가에서 자동 평가로 — 왜 RAGAS가 필요한가

RAGAS를 본격적으로 배우기 전에, "그냥 눈으로 보면 안 되나?"라는 질문에 먼저 답해보겠습니다.

가장 직관적인 평가 방법은 **수동 평가**입니다. 질문 몇 개를 던져보고, 답변이 그럴듯한지 직접 확인하는 거죠:

```python
# 가장 원시적인 RAG 평가 — 눈으로 직접 확인
test_questions = [
    "LangChain의 LCEL이란 무엇인가요?",
    "RAG 파이프라인의 단계를 설명해주세요.",
    "벡터 스토어에서 사용하는 거리 측정 방식은?",
]

for question in test_questions:
    result = rag_chain.invoke(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"검색된 문서: {[doc.page_content[:50] for doc in result['context']]}")
    print("-" * 50)
    # 여기서 사람이 직접 판단: "이 답변, 괜찮은가?"
```

이 방법의 문제점은 명확합니다:

| 수동 평가의 한계 | 설명 |
|----------------|------|
| **확장 불가** | 질문이 100개, 1000개가 되면 사람이 다 볼 수 없음 |
| **주관적** | "꽤 괜찮다"의 기준이 사람마다 다름 |
| **재현 불가** | 같은 답변을 다른 날 보면 다르게 평가할 수 있음 |
| **비교 불가** | 개선 전후를 정량적으로 비교할 수 없음 |

기존 NLP 평가 메트릭(BLEU, ROUGE)은 단어 겹침만 보기 때문에, "의미적으로 같지만 표현이 다른 답변"을 올바르게 평가하지 못합니다. 예를 들어 "Harrison Chase가 만들었습니다"와 "2022년 Harrison Chase에 의해 공개되었습니다"는 의미가 같지만, 단어 겹침 점수는 낮게 나오죠.

**RAGAS의 혁신**은 바로 여기에 있습니다. **LLM을 심판(Judge)으로 활용**하여 의미 수준에서 평가할 수 있게 한 것이에요. 사람처럼 맥락을 이해하되, 자동으로 대규모 평가를 수행할 수 있습니다.

### 개념 3: RAGAS 프레임워크와 4대 핵심 메트릭

> 💡 **비유**: RAGAS 메트릭은 학교 시험의 채점 기준표와 같습니다. "답이 맞았느냐"만 보는 게 아니라, "풀이 과정이 논리적이냐", "문제를 정확히 이해했느냐", "필요한 공식을 다 활용했느냐"를 각각 평가하는 거죠.

RAGAS(Retrieval Augmented Generation Assessment)는 RAG 시스템을 자동으로 평가하는 오픈소스 프레임워크입니다. **사람의 라벨링(Ground Truth) 없이도** LLM을 심판(Judge)으로 활용하여 평가할 수 있다는 것이 핵심 특징이에요.

RAGAS의 4대 핵심 메트릭을 하나씩 차근차근 살펴보겠습니다:

**1. Faithfulness (충실도)** — 생성 평가

답변이 검색된 컨텍스트에 **사실적으로 근거**하는지 측정합니다. 할루시네이션을 잡아내는 핵심 메트릭이에요.

$$\text{Faithfulness} = \frac{\text{컨텍스트에 의해 지지되는 주장 수}}{\text{답변의 총 주장 수}}$$

- **주장(Claim)**: 답변에서 추출된 개별 사실적 진술
- **지지(Supported)**: 해당 주장이 검색된 컨텍스트에서 확인 가능한지 여부
- 값 범위: 0.0 ~ 1.0 (높을수록 좋음)

구체적인 예시를 볼까요?

```
컨텍스트: "LCEL은 파이프 연산자(|)를 사용하여 Runnable 컴포넌트를 조합한다."

답변: "LCEL은 파이프 연산자를 사용하며, 자동으로 GPU 가속을 지원합니다."
  → 주장 1: "파이프 연산자를 사용한다" → ✅ 컨텍스트에서 확인됨
  → 주장 2: "GPU 가속을 지원한다"   → ❌ 컨텍스트에 근거 없음 (할루시네이션!)

  Faithfulness = 1/2 = 0.5
```

**2. Answer Relevancy (답변 관련성)** — 생성 평가

답변이 질문에 **얼마나 적절하게** 응답하는지 측정합니다.

평가 방식이 흥미로운데요, LLM이 답변으로부터 "이 답변에 해당하는 질문은 무엇일까?"를 역으로 생성한 뒤, 원래 질문과의 **의미적 유사도**(코사인 유사도)를 계산합니다.

```
원래 질문: "LCEL이란 무엇인가요?"
답변: "LCEL은 파이프 연산자로 컴포넌트를 조합하는 선언적 언어입니다."

LLM이 답변에서 역생성한 질문들:
  → "LCEL의 정의는 무엇인가요?" (원래 질문과 유사도 높음 ✅)
  → "파이프 연산자의 역할은?" (원래 질문과 유사도 중간)

Answer Relevancy = 역생성 질문들과 원래 질문의 평균 유사도
```

**3. Context Precision (컨텍스트 정밀도)** — 검색 평가

검색된 문서들 중 **실제로 관련 있는 문서의 비율**을 측정합니다. 신호 대 잡음비(Signal-to-Noise Ratio)라고 생각하면 됩니다.

```
질문: "LCEL이란 무엇인가요?"
검색된 문서 4개 중:
  → 문서 1: LCEL 설명 문서 ✅ 관련 있음
  → 문서 2: LangChain 개요 ✅ 관련 있음
  → 문서 3: 벡터 스토어 가이드 ❌ 관련 없음
  → 문서 4: LangGraph 소개 ❌ 관련 없음

Context Precision ≈ 2/4 = 0.5 (관련 문서가 상위에 올수록 점수 높음)
```

**4. Context Recall (컨텍스트 재현율)** — 검색 평가

정답을 만들기 위해 필요한 정보를 검색기가 **얼마나 빠짐없이** 찾아왔는지 측정합니다. 이 메트릭은 참조 답변(Reference)이 필요합니다.

```
참조 답변: "RAG는 문서 로드, 텍스트 분할, 임베딩, 벡터 저장, 검색, 생성의 6단계이다."
참조 답변의 핵심 정보 6개 중 검색된 문서에 포함된 정보: 5개

Context Recall = 5/6 ≈ 0.83
```

이 4가지 메트릭을 종합하면 RAG 시스템의 약점을 정확히 진단할 수 있습니다:

| 진단 패턴 | 가능한 원인 | 개선 방향 |
|-----------|------------|----------|
| Faithfulness 낮음 | LLM이 컨텍스트를 무시하고 지어냄 | 프롬프트 강화, 온도 낮추기 |
| Answer Relevancy 낮음 | 답변이 질문 의도에서 벗어남 | 프롬프트에 질문 재확인 추가 |
| Context Precision 낮음 | 관련 없는 문서가 많이 검색됨 | 임베딩 모델 변경, 필터링 추가 |
| Context Recall 낮음 | 필요한 문서를 못 찾음 | 검색 k값 증가, 청킹 전략 변경 |

### 개념 4: RAGAS 시작하기 — 첫 번째 평가 실행

이제 실제 코드로 RAGAS를 사용해 봅시다. 가장 단순한 형태부터 시작합니다.

```python
# RAGAS 설치
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# 평가 데이터 구성 — 각 필드가 어떤 역할을 하는지 주목하세요
sample = SingleTurnSample(
    user_input="LangChain의 LCEL이란 무엇인가요?",         # 사용자 질문
    retrieved_contexts=[                                     # 검색된 문서들
        "LCEL은 LangChain Expression Language의 약자로, "
        "파이프 연산자(|)를 사용하여 컴포넌트를 선언적으로 조합하는 언어입니다.",
        "LangChain은 LLM 기반 애플리케이션 개발 프레임워크입니다.",
    ],
    response=(                                               # RAG 시스템의 답변
        "LCEL(LangChain Expression Language)은 파이프 연산자를 "
        "사용하여 프롬프트, 모델, 파서 등을 선언적으로 조합하는 "
        "LangChain의 체인 구성 언어입니다."
    ),
    reference=(                                              # 참조 정답 (Context Recall용)
        "LCEL은 LangChain Expression Language로, 파이프 연산자(|)를 "
        "통해 Runnable 컴포넌트를 조합하는 선언적 언어입니다."
    ),
)

# EvaluationDataset 생성
eval_dataset = EvaluationDataset(samples=[sample])

# 4대 메트릭으로 평가 실행
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(),        # 답변이 컨텍스트에 충실한가?
        AnswerRelevancy(),     # 답변이 질문에 적절한가?
        ContextPrecision(),    # 검색된 문서가 관련 있는가?
        ContextRecall(),       # 필요한 정보를 다 찾았는가?
    ],
)

# 결과 출력
print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88,
#  'context_precision': 0.75, 'context_recall': 0.90}

# DataFrame으로 변환하여 상세 분석
df = results.to_pandas()
print(df)
```

> ⚠️ **흔한 오해**: "RAGAS 점수가 높으면 RAG 시스템이 완벽하다"고 생각하기 쉽지만, RAGAS는 **LLM이 심판**이기 때문에 심판 자체의 한계가 있습니다. 특히 Faithfulness 평가에서 LLM이 미묘한 사실 왜곡을 놓칠 수 있어요. 따라서 RAGAS 점수는 **상대적 비교**와 **추세 파악**에 활용하고, 절대적 품질 보장으로 맹신하면 안 됩니다.

### 개념 5: 좋은 평가 데이터셋 만들기

RAGAS 메트릭이 아무리 정교해도, 평가 데이터셋이 부실하면 의미 있는 결과를 얻을 수 없습니다. 좋은 평가 데이터셋을 구축하는 원칙을 알아보겠습니다.

**평가 데이터셋의 구성 요소:**

```python
# 하나의 평가 샘플에 필요한 4가지 요소
sample = SingleTurnSample(
    user_input="...",           # 1. 사용자 질문
    retrieved_contexts=["..."], # 2. 검색된 문서들 (RAG 체인 실행 시 자동 수집)
    response="...",             # 3. RAG 시스템의 답변 (RAG 체인 실행 시 자동 생성)
    reference="...",            # 4. 참조 정답 (사람이 미리 작성 — Context Recall용)
)
# → 직접 준비해야 하는 것: user_input + reference
# → RAG 체인이 만드는 것: retrieved_contexts + response
```

**좋은 데이터셋의 3가지 원칙:**

| 원칙 | 설명 | 예시 |
|------|------|------|
| **대표성** | 실제 사용 패턴을 반영 | 자주 묻는 질문 유형 포함 |
| **다양성** | 쉬운/어려운/엣지 케이스 혼합 | 단순 사실, 비교, 추론 질문 |
| **정확성** | 참조 답변이 정확하고 완전함 | 문서에서 직접 확인 가능한 정답 |

**질문 유형별 예시:**

```python
# 다양한 유형의 평가 질문을 포함해야 합니다
eval_questions = [
    # 유형 1: 단순 사실 질문 (쉬움)
    {
        "question": "LangChain은 누가 만들었나요?",
        "reference": "LangChain은 2022년 10월 Harrison Chase가 처음 공개했습니다.",
        "type": "factual",
    },
    # 유형 2: 개념 설명 질문 (중간)
    {
        "question": "LCEL에서 파이프 연산자의 역할은 무엇인가요?",
        "reference": "파이프 연산자(|)는 Runnable 컴포넌트를 선언적으로 조합하는 데 사용됩니다.",
        "type": "conceptual",
    },
    # 유형 3: 여러 문서 종합 질문 (어려움)
    {
        "question": "RAG 파이프라인에서 벡터 스토어는 어떤 단계에 해당하나요?",
        "reference": (
            "벡터 스토어는 RAG 파이프라인의 '벡터 저장' 및 '검색' 단계에서 사용되며, "
            "임베딩된 텍스트를 저장하고 유사도 기반 검색을 수행합니다."
        ),
        "type": "multi_doc",
    },
    # 유형 4: 문서에 답이 없는 질문 (엣지 케이스)
    {
        "question": "LangChain의 라이선스는 무엇인가요?",
        "reference": "제공된 문서에서 LangChain의 라이선스 정보를 찾을 수 없습니다.",
        "type": "unanswerable",
    },
]
```

> 🔥 **실무 팁**: 평가 데이터셋은 최소 **20~50개** 질문을 권장합니다. 처음에는 10개로 시작하여 평가 파이프라인을 검증한 뒤, 점진적으로 늘려가세요. 질문 유형 비율은 단순 사실 40%, 개념 설명 30%, 종합/추론 20%, 엣지 케이스 10% 정도가 적당합니다.

## 실습: 직접 해보기

아래 실습에서는 간단한 RAG 시스템을 구축하고, RAGAS 메트릭으로 평가한 뒤, 프롬프트와 검색 설정을 개선하여 성능 변화를 비교합니다.

```python
"""
RAG 평가 & 개선 실습 — RAGAS 기본 워크플로우
==============================================
사전 준비:
  pip install langchain langchain-openai langchain-community
  pip install ragas faiss-cpu python-dotenv
"""
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

load_dotenv()

# ─────────────────────────────────────────────
# 1단계: 샘플 문서와 벡터 스토어 준비
# ─────────────────────────────────────────────

# 학습용 샘플 문서 (실제로는 문서 로더로 가져옵니다)
documents = [
    Document(
        page_content=(
            "LangChain은 LLM 기반 애플리케이션 개발을 위한 오픈소스 프레임워크입니다. "
            "2022년 10월 Harrison Chase가 처음 공개했으며, "
            "프롬프트 관리, 체인 구성, 에이전트 구축, 메모리 관리 등의 기능을 제공합니다. "
            "Python과 JavaScript/TypeScript 버전이 있습니다."
        ),
        metadata={"source": "langchain_overview.md", "chapter": 1},
    ),
    Document(
        page_content=(
            "LCEL(LangChain Expression Language)은 파이프 연산자(|)를 사용하여 "
            "Runnable 컴포넌트를 선언적으로 조합하는 언어입니다. "
            "LCEL로 구성된 체인은 자동으로 스트리밍, 배치 처리, "
            "비동기 실행을 지원합니다. invoke(), stream(), batch() 메서드를 "
            "통합 인터페이스로 제공합니다."
        ),
        metadata={"source": "lcel_guide.md", "chapter": 5},
    ),
    Document(
        page_content=(
            "RAG(Retrieval-Augmented Generation)는 외부 지식 소스에서 관련 정보를 "
            "검색하여 LLM 응답의 정확성을 높이는 기법입니다. "
            "기본 파이프라인은 문서 로드 → 텍스트 분할 → 임베딩 → 벡터 저장 → "
            "검색 → 생성의 단계로 구성됩니다. "
            "2020년 Facebook AI Research의 Patrick Lewis 팀이 제안했습니다."
        ),
        metadata={"source": "rag_basics.md", "chapter": 9},
    ),
    Document(
        page_content=(
            "벡터 스토어는 텍스트 임베딩을 저장하고 유사도 기반 검색을 수행하는 "
            "데이터베이스입니다. FAISS, Chroma, Pinecone, Weaviate 등이 대표적입니다. "
            "코사인 유사도, 유클리드 거리, 내적 등의 거리 측정 방식을 지원합니다."
        ),
        metadata={"source": "vectorstore_guide.md", "chapter": 7},
    ),
    Document(
        page_content=(
            "LangGraph는 상태 기반 그래프를 사용하여 복잡한 에이전트 워크플로우를 "
            "구축하는 LangChain 확장 라이브러리입니다. "
            "StateGraph의 노드(처리 단계)와 엣지(전환 조건)로 상태 머신을 구성하며, "
            "체크포인트를 통한 중단 후 재개, Human-in-the-Loop 패턴을 지원합니다."
        ),
        metadata={"source": "langgraph_intro.md", "chapter": 13},
    ),
]

# 텍스트 분할 및 벡터 스토어 구축
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,     # 작은 청크로 분할 (실습용)
    chunk_overlap=30,
)
splits = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(splits, embeddings)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─────────────────────────────────────────────
# 2단계: 기본 RAG 체인 (Baseline)
# ─────────────────────────────────────────────

# 기본 프롬프트 — 최소한의 지시만 포함
basic_prompt = ChatPromptTemplate.from_template(
    "다음 컨텍스트를 참고하여 질문에 답하세요.\n\n"
    "컨텍스트: {context}\n\n"
    "질문: {question}"
)

# 기본 검색기 — top_k=2
basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    """검색된 문서를 문자열로 포맷합니다."""
    return "\n\n".join(doc.page_content for doc in docs)


# LCEL로 기본 RAG 체인 구축
basic_rag_chain = (
    RunnableParallel(
        context=basic_retriever | format_docs,
        question=RunnablePassthrough(),
        raw_docs=basic_retriever,  # 평가용으로 원본 문서도 보존
    )
    | RunnableParallel(
        answer=basic_prompt | llm | StrOutputParser(),
        context=lambda x: x["raw_docs"],
    )
)

# ─────────────────────────────────────────────
# 3단계: 개선된 RAG 체인 (Improved)
# ─────────────────────────────────────────────

# 개선 1: 더 정교한 프롬프트 (9.2에서 배운 기법 적용)
improved_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 LangChain 전문가입니다. 아래 규칙을 반드시 따르세요:\n"
     "1. 제공된 컨텍스트에 있는 정보만 사용하여 답변하세요.\n"
     "2. 컨텍스트에 없는 내용은 절대 추측하지 마세요.\n"
     "3. 답변할 수 없는 경우 '제공된 문서에서 해당 정보를 찾을 수 없습니다'라고 "
     "답하세요.\n"
     "4. 가능하면 출처 문서를 언급하세요."),
    ("human",
     "컨텍스트:\n{context}\n\n질문: {question}"),
])

# 개선 2: 더 많은 문서 검색 — top_k=4
improved_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 개선된 LCEL RAG 체인
improved_rag_chain = (
    RunnableParallel(
        context=improved_retriever | format_docs,
        question=RunnablePassthrough(),
        raw_docs=improved_retriever,
    )
    | RunnableParallel(
        answer=improved_prompt | llm | StrOutputParser(),
        context=lambda x: x["raw_docs"],
    )
)

# ─────────────────────────────────────────────
# 4단계: 평가 데이터셋 구성
# ─────────────────────────────────────────────

# 평가용 질문-정답 쌍
test_data = [
    {
        "question": "LangChain은 누가 만들었나요?",
        "reference": "LangChain은 2022년 10월 Harrison Chase가 처음 공개했습니다.",
    },
    {
        "question": "LCEL에서 파이프 연산자의 역할은 무엇인가요?",
        "reference": "파이프 연산자(|)는 Runnable 컴포넌트를 선언적으로 조합하는 데 사용됩니다.",
    },
    {
        "question": "RAG 파이프라인의 단계를 설명해주세요.",
        "reference": (
            "RAG의 기본 파이프라인은 문서 로드, 텍스트 분할, 임베딩, "
            "벡터 저장, 검색, 생성의 6단계로 구성됩니다."
        ),
    },
    {
        "question": "벡터 스토어에서 사용하는 거리 측정 방식은?",
        "reference": "코사인 유사도, 유클리드 거리, 내적 등의 거리 측정 방식을 지원합니다.",
    },
    {
        "question": "LangGraph에서 체크포인트의 역할은?",
        "reference": "체크포인트는 그래프 실행 중간 상태를 저장하여 중단 후 재개를 가능하게 합니다.",
    },
]

# ─────────────────────────────────────────────
# 5단계: RAGAS 평가 함수 정의
# ─────────────────────────────────────────────


def evaluate_rag_chain(chain, test_data: list[dict], chain_name: str) -> dict:
    """RAG 체인을 RAGAS 메트릭으로 평가합니다."""
    samples = []

    for item in test_data:
        # RAG 체인 실행
        result = chain.invoke(item["question"])

        # 결과에서 답변과 컨텍스트 추출
        answer = result.get("answer", "")
        contexts = [
            doc.page_content
            for doc in result.get("context", [])
        ]

        # RAGAS 샘플 구성
        sample = SingleTurnSample(
            user_input=item["question"],
            retrieved_contexts=contexts,
            response=answer,
            reference=item["reference"],
        )
        samples.append(sample)

        # 개별 결과 출력
        print(f"\nQ: {item['question']}")
        print(f"A: {answer[:80]}...")
        print(f"  검색된 문서 수: {len(contexts)}")

    # RAGAS 평가 실행
    eval_dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
    )

    print(f"\n{'='*50}")
    print(f"[{chain_name}] 평가 결과")
    print(f"{'='*50}")
    for metric_name, score in results.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric_name:<25} {bar} {score:.3f}")
    print(f"{'='*50}")

    return dict(results)


# ─────────────────────────────────────────────
# 6단계: Baseline vs Improved 비교
# ─────────────────────────────────────────────

print("RAGAS 평가 시작\n")

# Baseline 평가
print("[Baseline] 기본 RAG 체인 평가")
baseline_scores = evaluate_rag_chain(
    basic_rag_chain, test_data, "Baseline"
)

# Improved 평가
print("\n[Improved] 개선된 RAG 체인 평가")
improved_scores = evaluate_rag_chain(
    improved_rag_chain, test_data, "Improved"
)

# 결과 비교
print("\n\n결과 비교: Baseline vs Improved")
print(f"{'메트릭':<25} {'Baseline':>10} {'Improved':>10} {'변화':>10}")
print("-" * 55)
for metric in baseline_scores:
    base = baseline_scores[metric]
    impr = improved_scores[metric]
    diff = impr - base
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    print(f"{metric:<25} {base:>10.3f} {impr:>10.3f} {arrow}{abs(diff):>8.3f}")

# 출력 예시:
# 결과 비교: Baseline vs Improved
# 메트릭                     Baseline   Improved       변화
# -------------------------------------------------------
# faithfulness                  0.720      0.880     ↑   0.160
# answer_relevancy              0.810      0.850     ↑   0.040
# context_precision             0.650      0.780     ↑   0.130
# context_recall                0.700      0.820     ↑   0.120
```

> 🔥 **실무 팁**: 위 실습에서 Baseline과 Improved의 차이는 **프롬프트 강화**(할루시네이션 방지 지시 추가)와 **검색 수 증가**(k=2→4)뿐입니다. 이 간단한 변경만으로도 Faithfulness와 Context Recall이 크게 개선되는 것을 확인할 수 있어요. 복잡한 기법을 적용하기 전에, 이런 기본적인 개선부터 시도하고 RAGAS로 효과를 검증하는 습관을 들이세요.

## 더 깊이 알아보기

### RAGAS의 탄생 스토리

RAGAS는 인도 벵갈루루 출신의 두 친구, **Shahul ES**와 **Jithin James**가 만들었습니다. Shahul은 Kaggle Grandmaster이자 Open-Assistant AI 오픈소스 프로젝트의 핵심 기여자였고, Jithin은 BentoML 초기 멤버로 MLOps 도구를 만들던 엔지니어였죠.

2023년, RAG가 폭발적으로 성장하면서 두 사람은 공통된 문제를 발견했습니다. "모두가 RAG를 만들고 있는데, 아무도 RAG가 *얼마나 잘 동작하는지* 제대로 측정하지 못한다." 기존의 NLP 평가 메트릭(BLEU, ROUGE 등)은 RAG의 특성을 반영하지 못했고, 사람이 직접 평가하는 건 너무 비싸고 느렸습니다.

그래서 그들은 **LLM을 심판으로 활용하는** 혁신적인 접근법을 고안했습니다. 답변을 개별 주장(Claim)으로 분해한 뒤, 각 주장이 컨텍스트에 근거하는지 LLM으로 검증하는 Faithfulness 메트릭이 대표적이에요. 2023년 9월 arXiv에 논문을 발표했고, 2024년 유럽 컴퓨터 언어학 학회(EACL)에서 데모 논문으로 채택되었습니다.

놀랍게도 RAGAS는 발표 후 불과 1년 만에 Y Combinator의 투자를 받았고, AWS, Microsoft, Databricks, Moody's 같은 기업들이 사용하면서 월 500만 건 이상의 평가를 처리하는 플랫폼으로 성장했습니다. "평가 없이는 개선 없다(You can't improve what you can't measure)"는 격언이 RAG 세계에서도 증명된 셈이죠.

### 평가 메트릭의 수학적 배경

Faithfulness 계산의 내부 과정을 좀 더 자세히 살펴보면:

1. **주장 추출(Claim Extraction)**: LLM이 답변에서 개별 사실적 진술을 추출합니다.
   - 예: "LCEL은 파이프 연산자를 사용한다" → 주장 1
   - 예: "LCEL은 선언적 조합을 지원한다" → 주장 2

2. **근거 검증(Verification)**: 각 주장이 컨텍스트에서 확인 가능한지 LLM이 판단합니다.

3. **점수 계산**:

$$\text{Faithfulness} = \frac{|V|}{|S|}$$

- $|V|$: 컨텍스트에 의해 지지된(Verified) 주장의 수
- $|S|$: 답변에서 추출된 전체 주장(Statement)의 수

Answer Relevancy의 경우, 역질문 생성 후 코사인 유사도를 계산합니다:

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \text{sim}(q, q_i)$$

- $q$: 원래 질문
- $q_i$: 답변에서 역생성된 $i$번째 질문
- $N$: 역생성된 질문 수 (보통 3~5개)

이게 의미하는 바는, 좋은 답변은 원래 질문을 충실히 반영하므로, 답변에서 역으로 추론한 질문이 원래 질문과 유사해야 한다는 것입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "RAGAS 점수가 0.9 이상이면 프로덕션 준비 완료!"라고 생각하기 쉽지만, RAGAS 점수의 **절대값**보다 **변화 추세**가 더 중요합니다. 평가 데이터셋의 구성, LLM 심판의 모델, 프롬프트에 따라 점수가 크게 달라질 수 있거든요. 같은 평가 조건에서 **상대 비교**하는 것이 가장 신뢰할 수 있는 활용법입니다.

> 💡 **알고 계셨나요?**: RAGAS의 Faithfulness 메트릭은 사실 NLI(Natural Language Inference, 자연어 추론) 태스크에서 영감을 받았습니다. NLI는 "전제(Premise)"가 주어졌을 때 "가설(Hypothesis)"이 참인지 판단하는 태스크인데, Faithfulness는 "컨텍스트"를 전제로, "답변의 각 주장"을 가설로 놓고 검증하는 거죠. 2023년 이전에는 RoBERTa 같은 소형 모델로 NLI 점수를 계산했지만, GPT-4급 LLM이 등장하면서 LLM을 심판으로 활용하는 것이 더 정확해졌습니다.

> 🔥 **실무 팁**: 평가 데이터셋을 만들 때 **엣지 케이스**를 반드시 포함하세요. 정상적인 질문뿐 아니라, (1) 문서에 답이 없는 질문, (2) 여러 문서의 정보를 종합해야 하는 질문, (3) 모호하거나 다의적인 질문도 넣어야 합니다. 실무에서 RAG 시스템이 가장 자주 실패하는 건 이런 엣지 케이스인데, 평가 데이터에 없으면 문제를 발견할 수 없습니다.

> 🔥 **실무 팁**: RAGAS 결과에서 특정 메트릭만 낮게 나온다면, 앞서 배운 진단 표를 활용하세요. Faithfulness가 낮으면 프롬프트에 "컨텍스트에 없는 정보는 사용하지 마세요" 지시를 추가하고, Context Recall이 낮으면 검색 k값을 늘리거나 청킹 전략을 변경해 보세요. 문제의 원인이 **검색 단계**인지 **생성 단계**인지 정확히 분리해야 올바른 개선 방향을 잡을 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| RAG 평가의 두 축 | 검색 품질과 생성 품질을 분리하여 평가해야 정확한 개선이 가능 |
| RAGAS | LLM을 심판으로 활용하여 RAG 시스템을 자동 평가하는 오픈소스 프레임워크 |
| Faithfulness (충실도) | 답변이 검색된 컨텍스트에 사실적으로 근거하는지 측정 (할루시네이션 탐지) |
| Answer Relevancy (답변 관련성) | 답변이 원래 질문에 얼마나 적절하게 응답하는지 측정 |
| Context Precision (컨텍스트 정밀도) | 검색된 문서 중 실제 관련 문서의 비율 측정 |
| Context Recall (컨텍스트 재현율) | 필요한 정보를 빠짐없이 검색했는지 측정 (참조 답변 필요) |
| SingleTurnSample | RAGAS 평가의 기본 데이터 단위 (질문, 답변, 컨텍스트, 참조) |
| EvaluationDataset | SingleTurnSample들의 모음으로, evaluate()에 전달 |
| LLM-as-Judge | LLM을 평가자로 사용하여 사람 없이 자동 채점하는 패러다임 |
| 평가 데이터셋 설계 | 대표성, 다양성, 정확성을 갖춘 질문-정답 쌍 구축 원칙 |

## 다음 섹션 미리보기

이번 세션에서 RAGAS 메트릭을 이해하고 RAG 시스템을 정량적으로 평가하는 기초를 다졌으니, 다음 [9.6 프로덕션 RAG 아키텍처](./06-프로덕션-rag-아키텍처.md)에서는 이를 실제 서비스에 적용합니다. **LangSmith를 활용한 체계적 모니터링 파이프라인** 구축, **A/B 테스트 프레임워크**로 고급 RAG 패턴(HyDE, 재랭킹 등)의 효과를 정량 비교하는 방법, 그리고 캐싱 전략, 비용 최적화, 확장 가능한 아키텍처 패턴을 함께 다룹니다.

## 참고 자료

- [RAGAS 공식 문서 — Evaluate a simple RAG system](https://docs.ragas.io/en/stable/tutorials/rag/) - RAGAS 프레임워크의 공식 튜토리얼로, SingleTurnSample과 EvaluationDataset 사용법을 단계별로 설명합니다
- [RAGAS 논문 (arXiv)](https://arxiv.org/abs/2309.15217) - Shahul Es, Jithin James 등이 발표한 원본 논문으로, Faithfulness와 Answer Relevancy 메트릭의 이론적 배경을 이해할 수 있습니다
- [RAGAS 메트릭 목록](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) - Faithfulness, Context Precision, Answer Relevancy 등 모든 RAGAS 메트릭의 계산 방식과 사용법을 정리한 공식 레퍼런스입니다
- [LangSmith RAG 평가 튜토리얼](https://docs.langchain.com/langsmith/evaluate-rag-tutorial) - 다음 세션(9.6)에서 활용할 LangSmith 기반 RAG 평가 공식 가이드입니다
- [Evaluating RAG pipelines with Ragas + LangSmith (LangChain 블로그)](https://blog.langchain.com/evaluating-rag-pipelines-with-ragas-langsmith/) - RAGAS와 LangSmith를 통합하여 RAG 파이프라인을 평가하는 실전 가이드입니다

---
### 🔗 Related Sessions
- [rag_pipeline](./01-기본-rag-체인-구축.md) (prerequisite)
- [format_docs](./01-기본-rag-체인-구축.md) (prerequisite)
- [retrieval_chain](./01-기본-rag-체인-구축.md) (prerequisite)
- [stuff_documents_chain](./01-기본-rag-체인-구축.md) (prerequisite)
- [production_rag_prompt](./02-rag-프롬프트-최적화.md) (prerequisite)
- [hyde_embeddings](./04-고급-rag-패턴.md) (prerequisite)
- [cross_encoder_reranker](./04-고급-rag-패턴.md) (prerequisite)
- [multi_query_retriever](./04-고급-rag-패턴.md) (prerequisite)
