# 멀티쿼리와 RAG Fusion

> 하나의 질문을 여러 관점으로 확장하고, 검색 결과를 지능적으로 융합하여 검색 품질을 극적으로 향상시키는 고급 검색 전략

## 개요

이 섹션에서는 사용자의 단일 질문을 LLM이 여러 변형 쿼리로 자동 확장하는 **MultiQueryRetriever**와, 여기에 Reciprocal Rank Fusion(RRF) 재순위화를 결합한 **RAG Fusion** 패턴을 학습합니다. 앞서 [8.2 키워드와 앙상블 검색](ch08/session_8.2.md)에서 배운 EnsembleRetriever가 "서로 다른 *검색 방식*"을 결합했다면, 이번 섹션에서는 "서로 다른 *질문*"을 결합하는 전략을 다룹니다.

**선수 지식**: 이전 섹션에서 배운 내용 중 필요한 것
- [8.1 검색기 기초](ch08/session_8.1.md)의 VectorStoreRetriever와 `as_retriever()` 사용법
- [8.2 키워드와 앙상블 검색](ch08/session_8.2.md)의 EnsembleRetriever와 Reciprocal Rank Fusion 개념
- LCEL 파이프 연산자(`|`)를 사용한 체인 구성 (Chapter 5)

**학습 목표**:
- MultiQueryRetriever의 작동 원리를 이해하고 직접 구성할 수 있다
- 커스텀 프롬프트로 쿼리 생성 전략을 제어할 수 있다
- RAG Fusion 패턴을 LCEL로 직접 구현할 수 있다
- 쿼리 확장과 결과 융합이 검색 품질에 미치는 영향을 설명할 수 있다

## 왜 알아야 할까?

검색 시스템에서 가장 큰 병목은 의외로 검색 엔진이 아니라 **사용자의 질문 자체**인 경우가 많습니다. 같은 정보를 찾더라도 "LangChain 메모리 사용법"과 "대화 기록 유지 방법"은 전혀 다른 검색 결과를 가져오죠. 사용자가 완벽한 질문을 던질 거라고 가정하는 것은 비현실적입니다.

실제 프로덕션 RAG 시스템에서 흔히 발생하는 문제들을 보겠습니다:

- **어휘 불일치**: 사용자는 "에러 처리"라고 쓰지만 문서에는 "예외 핸들링"으로 적혀 있음
- **관점 부족**: 하나의 질문으로는 관련 문서의 일부만 검색됨
- **모호한 질문**: "성능 개선하려면?"처럼 구체성이 부족한 질문

MultiQueryRetriever와 RAG Fusion은 이 문제를 LLM의 언어 이해 능력으로 해결합니다. 하나의 질문을 다양한 관점의 여러 질문으로 자동 확장하고, 그 결과를 지능적으로 융합하는 것이죠. 실무에서 이 기법을 적용하면 검색 재현율(recall)이 20~40% 향상되는 경우가 흔합니다.

## 핵심 개념

### 개념 1: MultiQueryRetriever — 하나의 질문, 다양한 관점

> 💡 **비유**: 도서관에서 책을 찾는다고 생각해보세요. "인공지능 입문서"를 찾고 있다면, 한 가지 키워드로만 검색하는 것보다 "AI 기초", "머신러닝 개론", "딥러닝 시작하기"처럼 여러 각도로 검색하면 훨씬 다양한 좋은 책을 발견할 수 있겠죠? MultiQueryRetriever가 바로 이 역할을 합니다 — LLM이 사서(司書) 역할을 하면서 여러분의 질문을 다양한 검색어로 바꿔주는 거예요.

MultiQueryRetriever는 LangChain이 제공하는 고급 검색기로, 내부적으로 다음 3단계를 수행합니다:

1. **쿼리 생성**: LLM에게 원본 질문의 변형 버전 3~5개를 생성하게 함
2. **병렬 검색**: 각 변형 쿼리로 독립적으로 문서를 검색
3. **결과 합집합**: 모든 검색 결과의 중복을 제거하고 합침

가장 간단한 사용법부터 살펴보겠습니다:

```python
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# 로깅 설정 — 생성된 쿼리를 확인하기 위해 필수!
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

# 샘플 문서 준비
docs = [
    Document(page_content="LangChain의 메모리 시스템은 대화 기록을 유지합니다.", metadata={"topic": "memory"}),
    Document(page_content="ConversationBufferMemory는 전체 대화를 저장합니다.", metadata={"topic": "memory"}),
    Document(page_content="ConversationSummaryMemory는 대화를 요약하여 저장합니다.", metadata={"topic": "memory"}),
    Document(page_content="LCEL에서 RunnableWithMessageHistory를 사용하면 대화 이력을 관리할 수 있습니다.", metadata={"topic": "lcel"}),
    Document(page_content="Chat Model은 시스템, 사용자, AI 메시지를 구분합니다.", metadata={"topic": "model"}),
    Document(page_content="프롬프트 템플릿에 chat_history 변수를 포함하면 이전 대화를 참조할 수 있습니다.", metadata={"topic": "prompt"}),
]

# 벡터 스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# MultiQueryRetriever 생성
llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
    include_original=True,  # 원본 쿼리도 포함
)

# 검색 실행
results = retriever.invoke("LangChain에서 대화 기록을 어떻게 관리하나요?")

# 결과 출력
for i, doc in enumerate(results, 1):
    print(f"[{i}] {doc.page_content}")
```

```
# 로그 출력 예시 (DEBUG 레벨):
# Generated queries:
# 1. LangChain 메모리 시스템의 종류와 사용법은 무엇인가요?
# 2. LangChain에서 이전 대화 내용을 저장하고 참조하는 방법은?
# 3. LangChain 채팅 애플리케이션에서 대화 이력 관리 기능은 어떻게 구현하나요?

# [1] LangChain의 메모리 시스템은 대화 기록을 유지합니다.
# [2] ConversationBufferMemory는 전체 대화를 저장합니다.
# [3] ConversationSummaryMemory는 대화를 요약하여 저장합니다.
# [4] LCEL에서 RunnableWithMessageHistory를 사용하면 대화 이력을 관리할 수 있습니다.
# [5] 프롬프트 템플릿에 chat_history 변수를 포함하면 이전 대화를 참조할 수 있습니다.
```

핵심은 `include_original=True`입니다. 이 옵션을 켜면 LLM이 생성한 변형 쿼리들에 **원본 질문까지 추가**되어 총 4개의 쿼리로 검색합니다. 원본 질문이 이미 충분히 좋은 경우를 놓치지 않기 위해서죠.

> ⚠️ **흔한 오해**: MultiQueryRetriever가 검색 *정확도*를 높인다고 생각하기 쉽지만, 실제로 주로 높아지는 것은 **재현율(recall)**입니다. 단일 쿼리로는 놓칠 수 있던 관련 문서를 추가로 찾아주는 것이지, 무관한 문서를 더 잘 걸러내는 것은 아닙니다.

### 개념 2: 커스텀 쿼리 생성 프롬프트

기본 프롬프트도 잘 작동하지만, 도메인에 맞게 쿼리 생성 전략을 세밀하게 제어하고 싶을 때가 있습니다. 예를 들어, 법률 문서 검색에서는 법률 용어와 일상 용어를 모두 포함한 쿼리를 생성하고 싶을 수 있죠.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# 커스텀 출력 파서: 줄바꿈으로 구분된 쿼리를 리스트로 변환
class LineListOutputParser(BaseOutputParser[list[str]]):
    """줄바꿈으로 구분된 텍스트를 리스트로 파싱합니다."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

# 도메인 특화 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 검색 쿼리 전문가입니다. 사용자의 질문을 다양한 관점에서 재구성하세요."),
    ("human", """다음 질문에 대해 5개의 서로 다른 검색 쿼리를 생성하세요.
각 쿼리는 원래 질문과 같은 정보를 찾되, 다른 단어와 관점을 사용해야 합니다.

규칙:
1. 기술 용어와 일상 용어를 번갈아 사용하세요
2. 구체적인 쿼리와 포괄적인 쿼리를 섞으세요
3. 각 쿼리는 한 줄에 하나씩 작성하세요

원래 질문: {question}"""),
])

# 쿼리 생성 체인 구성
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)  # 다양성을 위해 temperature 높임
output_parser = LineListOutputParser()
llm_chain = prompt | llm | output_parser

# MultiQueryRetriever에 커스텀 체인 적용
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm_chain=llm_chain,       # 커스텀 체인 사용
    include_original=True,
)

results = retriever.invoke("벡터 데이터베이스 성능 최적화")
```

`temperature`를 0.7로 올린 것에 주목하세요. 쿼리 생성에서는 약간의 창의성이 필요하기 때문에, 일반적인 LLM 호출보다 높은 temperature를 사용하는 것이 효과적입니다.

### 개념 3: RAG Fusion — 쿼리 확장 + 지능적 재순위화

> 💡 **비유**: 여러 명의 심사위원이 각각 독립적으로 요리 대회 참가자들의 순위를 매긴다고 상상해보세요. 한 심사위원의 판단만 믿는 것보다, 모든 심사위원의 순위를 종합하면 더 공정한 결과를 얻을 수 있겠죠? RAG Fusion은 바로 이 "심사위원 투표 집계" 방식을 검색에 적용한 것입니다. 각 쿼리가 하나의 심사위원 역할을 하고, RRF 알고리즘이 투표를 집계합니다.

RAG Fusion은 2024년 Zackary Rackauckas가 제안한 기법으로, MultiQueryRetriever의 아이디어를 한 단계 발전시킨 것입니다. 핵심 차이점은 결과를 단순히 합치는 것이 아니라, **Reciprocal Rank Fusion(RRF)으로 재순위화**한다는 점입니다.

MultiQueryRetriever와 RAG Fusion의 차이를 비교하면:

| 단계 | MultiQueryRetriever | RAG Fusion |
|------|-------------------|------------|
| 쿼리 생성 | LLM이 변형 쿼리 생성 | LLM이 변형 쿼리 생성 |
| 검색 | 각 쿼리로 독립 검색 | 각 쿼리로 독립 검색 |
| 결과 결합 | **합집합** (중복 제거) | **RRF 점수로 재순위화** |
| 순위 정보 | 활용하지 않음 | 각 쿼리에서의 순위를 점수에 반영 |

[8.2 키워드와 앙상블 검색](ch08/session_8.2.md)에서 EnsembleRetriever가 RRF를 사용한 것을 기억하시나요? 거기서는 *서로 다른 검색기*(BM25 + 벡터)의 결과를 융합했지만, RAG Fusion에서는 *서로 다른 쿼리*의 검색 결과를 융합합니다.

RRF 점수 공식을 다시 떠올려봅시다:

$$RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}$$

- $d$: 문서
- $R$: 모든 검색 결과 리스트의 집합 (각 쿼리의 검색 결과)
- $rank_r(d)$: 검색 결과 $r$에서 문서 $d$의 순위 (1부터 시작)
- $k$: 스무딩 상수 (보통 60)

이 수식이 의미하는 바는 이렇습니다: 어떤 문서가 **여러 쿼리의 결과에서 상위에 자주 등장할수록** 높은 점수를 받습니다. 한 쿼리에서 1등을 한 것보다, 세 쿼리에서 모두 3등 안에 든 문서가 더 높은 점수를 받을 수 있다는 것이죠.

### 개념 4: LCEL로 RAG Fusion 직접 구현하기

LangChain은 RAG Fusion을 위한 별도의 내장 클래스를 제공하지 않습니다. 대신 LCEL의 조합 능력을 활용하여 직접 구현할 수 있는데요, 이것이 오히려 장점입니다 — 각 단계를 완전히 제어할 수 있거든요.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60) -> list[Document]:
    """여러 검색 결과 리스트를 RRF 알고리즘으로 융합합니다.

    Args:
        results: 각 쿼리별 검색 결과 리스트의 리스트
        k: 스무딩 상수 (기본값 60)

    Returns:
        RRF 점수로 재순위화된 문서 리스트
    """
    fused_scores: dict[str, float] = {}  # page_content -> score
    doc_map: dict[str, Document] = {}     # page_content -> Document

    for doc_list in results:
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content
            if key not in fused_scores:
                fused_scores[key] = 0.0
                doc_map[key] = doc
            # RRF 점수 누적
            fused_scores[key] += 1.0 / (k + rank)

    # 점수 기준 내림차순 정렬
    sorted_docs = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc_map[content] for content, score in sorted_docs]
```

이 함수가 RAG Fusion의 핵심입니다. 각 쿼리의 검색 결과에서 문서별로 RRF 점수를 누적하고, 최종적으로 점수가 높은 순서대로 정렬합니다.

## 실습: 직접 해보기

이제 전체 RAG Fusion 파이프라인을 처음부터 끝까지 구축해보겠습니다. 쿼리 생성부터 최종 답변 생성까지의 완전한 코드입니다.

```python
"""RAG Fusion 완전 구현 실습"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ── 1단계: 샘플 문서 준비 ──
documents = [
    Document(page_content="FAISS는 Facebook AI Research가 개발한 벡터 유사도 검색 라이브러리입니다. 대규모 벡터 검색에 최적화되어 있습니다.", metadata={"source": "vectordb"}),
    Document(page_content="Chroma는 오픈소스 임베딩 데이터베이스로, 메타데이터 필터링과 함께 시맨틱 검색을 지원합니다.", metadata={"source": "vectordb"}),
    Document(page_content="Pinecone은 완전 관리형 벡터 데이터베이스 서비스로, 서버리스 아키텍처를 제공합니다.", metadata={"source": "vectordb"}),
    Document(page_content="벡터 검색의 성능을 높이려면 적절한 chunk_size와 chunk_overlap 설정이 중요합니다.", metadata={"source": "optimization"}),
    Document(page_content="임베딩 모델의 선택이 검색 품질에 가장 큰 영향을 미칩니다. 도메인 특화 모델을 사용하면 성능이 향상됩니다.", metadata={"source": "optimization"}),
    Document(page_content="하이브리드 검색은 키워드 검색과 시맨틱 검색을 결합하여 검색 품질을 향상시킵니다.", metadata={"source": "retrieval"}),
    Document(page_content="MMR(Maximal Marginal Relevance)은 검색 결과의 다양성을 확보하면서 관련성을 유지하는 알고리즘입니다.", metadata={"source": "retrieval"}),
    Document(page_content="IVF(Inverted File Index)와 HNSW(Hierarchical Navigable Small World)는 대표적인 ANN 인덱싱 알고리즘입니다.", metadata={"source": "algorithm"}),
    Document(page_content="벡터 데이터베이스의 인덱스를 주기적으로 재구축하면 검색 성능 저하를 방지할 수 있습니다.", metadata={"source": "optimization"}),
    Document(page_content="PQ(Product Quantization)를 사용하면 메모리 사용량을 줄이면서 근사 검색을 수행할 수 있습니다.", metadata={"source": "algorithm"}),
]

# 벡터 스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ── 2단계: 쿼리 생성 체인 ──
class LineListOutputParser(BaseOutputParser[list[str]]):
    """줄바꿈으로 구분된 텍스트를 리스트로 파싱합니다."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]


query_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 검색 쿼리를 다양한 관점으로 확장하는 전문가입니다."),
    ("human", """다음 질문에 대해 4개의 서로 다른 검색 쿼리를 생성하세요.
각 쿼리는 같은 정보를 찾되, 다른 단어와 관점을 사용하세요.
한 줄에 하나씩 작성하세요. 번호나 접두사 없이 쿼리만 작성하세요.

원래 질문: {question}"""),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 쿼리 생성 체인: 프롬프트 → LLM → 라인 파서
generate_queries = query_prompt | llm | LineListOutputParser()


# ── 3단계: RRF 함수 ──
def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60) -> list[Document]:
    """여러 검색 결과를 RRF로 융합하여 재순위화합니다."""
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in results:
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content
            if key not in fused_scores:
                fused_scores[key] = 0.0
                doc_map[key] = doc
            fused_scores[key] += 1.0 / (k + rank)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[content] for content, score in sorted_docs]


# ── 4단계: RAG Fusion 체인 구성 (LCEL) ──
# retriever.map()은 쿼리 리스트를 받아 각각에 대해 검색 실행
rag_fusion_retrieval = generate_queries | base_retriever.map() | RunnableLambda(reciprocal_rank_fusion)


# ── 5단계: 최종 RAG 체인 ──
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "주어진 컨텍스트를 기반으로 질문에 답하세요. 컨텍스트에 없는 내용은 '정보가 부족합니다'라고 답하세요."),
    ("human", """컨텍스트:
{context}

질문: {question}"""),
])


def format_docs(docs: list[Document]) -> str:
    """문서 리스트를 하나의 문자열로 포맷팅합니다."""
    return "\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)
    )


# 전체 RAG Fusion 체인
rag_fusion_chain = (
    {
        "context": rag_fusion_retrieval | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)


# ── 실행 ──
question = "벡터 데이터베이스 검색 성능을 어떻게 높일 수 있나요?"

# 1) 생성된 쿼리 확인
print("=== 생성된 쿼리 ===")
queries = generate_queries.invoke({"question": question})
for i, q in enumerate(queries, 1):
    print(f"  {i}. {q}")

# 2) 각 쿼리별 검색 결과 확인
print("\n=== 쿼리별 검색 결과 ===")
all_results = base_retriever.map().invoke(queries)
for i, (query, docs) in enumerate(zip(queries, all_results), 1):
    print(f"\n쿼리 {i}: {query}")
    for doc in docs:
        print(f"  - {doc.page_content[:50]}...")

# 3) RRF 융합 결과 확인
print("\n=== RRF 융합 결과 (재순위화) ===")
fused = reciprocal_rank_fusion(all_results)
for i, doc in enumerate(fused, 1):
    print(f"  [{i}] {doc.page_content[:60]}...")

# 4) 최종 답변
print("\n=== 최종 답변 ===")
answer = rag_fusion_chain.invoke(question)
print(answer)
```

```
# 출력 예시:
# === 생성된 쿼리 ===
#   1. 벡터 DB 쿼리 최적화 방법
#   2. 시맨틱 검색 성능 향상 전략
#   3. 벡터 인덱스 검색 속도 개선 기법
#   4. 임베딩 기반 검색 시스템 튜닝 방법
#
# === RRF 융합 결과 (재순위화) ===
#   [1] 벡터 검색의 성능을 높이려면 적절한 chunk_size와 chunk_overlap 설정이...
#   [2] 임베딩 모델의 선택이 검색 품질에 가장 큰 영향을 미칩니다...
#   [3] 벡터 데이터베이스의 인덱스를 주기적으로 재구축하면 검색 성능 저하를...
#   [4] IVF(Inverted File Index)와 HNSW(Hierarchical Navigable Small World)는...
#   [5] 하이브리드 검색은 키워드 검색과 시맨틱 검색을 결합하여...
#   [6] PQ(Product Quantization)를 사용하면 메모리 사용량을 줄이면서...
#   ...
```

코드의 핵심 구조를 단계별로 짚어보겠습니다:

- **`generate_queries`**: 사용자 질문 → LLM → 4개의 변형 쿼리 리스트
- **`base_retriever.map()`**: LCEL의 `.map()` 메서드가 쿼리 리스트의 각 항목에 대해 검색기를 병렬로 실행합니다. 입력이 `["쿼리1", "쿼리2", ...]`이면 출력은 `[[결과1], [결과2], ...]`
- **`reciprocal_rank_fusion`**: 중첩 리스트를 받아 RRF 점수로 재순위화한 단일 리스트 반환
- **최종 체인**: `RunnablePassthrough`로 원본 질문을 보존하면서, 융합된 검색 결과를 컨텍스트로 사용

## 더 깊이 알아보기

### RRF의 탄생: 단순함이 이긴 순간

Reciprocal Rank Fusion은 2009년 워털루 대학교의 Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher가 ACM SIGIR 학회에서 발표한 논문 "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"에서 처음 제안되었습니다.

당시 정보 검색 분야에서는 여러 검색 시스템의 결과를 효과적으로 결합하는 "메타 검색(meta-search)" 문제가 활발히 연구되고 있었는데요. 복잡한 기계 학습 기반 방법이나 투표 이론(Condorcet method)을 적용한 정교한 알고리즘들이 경쟁하고 있었습니다.

놀랍게도, Cormack 등이 제안한 RRF는 공식 한 줄(`1/(k+rank)`)의 극히 단순한 알고리즘임에도, 이 모든 복잡한 방법들을 능가하는 성능을 보여주었습니다. 특히 학습이 필요 없고, 각 검색 시스템의 점수 척도를 정규화할 필요도 없다는 점이 큰 장점이었죠. 스무딩 상수 $k=60$은 논문에서 실험적으로 결정된 값으로, 이후 15년이 지난 지금까지도 사실상 표준으로 사용되고 있습니다.

### RAG-Fusion 논문의 등장

2024년 초, Zackary Rackauckas가 arXiv에 발표한 논문 ["RAG-Fusion: a New Take on Retrieval-Augmented Generation"](https://arxiv.org/abs/2402.03367)은 2009년의 RRF 알고리즘을 RAG 파이프라인에 창의적으로 적용했습니다. 핵심 아이디어는 이렇습니다: 서로 다른 검색 *시스템*을 융합하는 대신, LLM으로 생성한 서로 다른 *쿼리*의 결과를 융합하자. 이 접근법은 별도의 검색 인프라 추가 없이도 단일 벡터 스토어에서 검색 품질을 크게 향상시킬 수 있다는 점에서 실용적인 돌파구가 되었습니다.

### k=60은 왜 60일까?

RRF의 스무딩 상수 $k=60$은 논문에서 TREC 실험 데이터를 대상으로 경험적으로 도출된 값입니다. $k$가 너무 작으면(예: $k=1$) 1등과 2등 사이의 점수 차이가 극단적으로 커지고, $k$가 너무 크면(예: $k=1000$) 모든 순위 간 차이가 미미해져서 순위 정보가 사실상 무의미해집니다. $k=60$은 상위 결과에 적절한 가중치를 부여하면서도, 하위 순위의 문서에도 합리적인 기여도를 허용하는 균형점입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "쿼리를 많이 생성할수록 좋다"고 생각하기 쉽지만, 쿼리 수를 5개 이상으로 늘리면 LLM 비용과 지연 시간이 선형적으로 증가하면서도 검색 품질 향상은 미미해지는 수확 체감이 발생합니다. 실무에서는 3~5개가 비용 대비 효과의 최적 구간입니다.

> 💡 **알고 계셨나요?**: MultiQueryRetriever의 결과 결합 방식은 단순 합집합(union)입니다. 즉, 한 쿼리에서 1등을 한 문서와 4등을 한 문서가 동등하게 취급됩니다. 이것이 RAG Fusion이 필요한 근본적인 이유입니다 — RRF는 순위 정보를 점수로 변환하여 "얼마나 자주, 얼마나 높은 순위에 등장했는가"를 정량적으로 반영합니다.

> 🔥 **실무 팁**: RAG Fusion의 쿼리 생성 프롬프트를 설계할 때, 단순히 "다른 표현으로 바꿔달라"고 하기보다 구체적인 전략을 제시하세요. 예를 들어 "1) 전문 용어 버전, 2) 일상 용어 버전, 3) 유사 질문, 4) 반대 관점 질문"처럼 각 쿼리의 역할을 명시하면 검색 다양성이 크게 향상됩니다.

> 🔥 **실무 팁**: `retriever.map()`은 쿼리 리스트를 입력받아 각 쿼리에 대해 검색을 **병렬로** 실행합니다. 하지만 비동기 환경이 아니라면 실제로는 순차 실행될 수 있으므로, 프로덕션에서는 `await retriever.abatch(queries)`를 사용하는 것이 더 효율적일 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| MultiQueryRetriever | LLM이 원본 질문을 여러 변형 쿼리로 확장하고, 각 쿼리의 검색 결과를 합집합으로 결합하는 검색기 |
| `include_original` | 원본 쿼리를 변형 쿼리 리스트에 포함시키는 옵션 (기본값 `False`) |
| RAG Fusion | 멀티쿼리 검색 결과를 RRF 알고리즘으로 재순위화하여 결합하는 고급 검색 패턴 |
| Reciprocal Rank Fusion (RRF) | 여러 순위 리스트를 `1/(k+rank)` 점수로 융합하는 알고리즘 (k=60) |
| `retriever.map()` | LCEL에서 리스트의 각 항목에 대해 Runnable을 적용하는 메서드 |
| 쿼리 생성 프롬프트 | 커스텀 프롬프트로 쿼리 생성 전략을 제어할 수 있음 |
| 쿼리 확장 vs 검색기 앙상블 | RAG Fusion은 다른 *쿼리*의 결과를, EnsembleRetriever는 다른 *검색기*의 결과를 융합 |

## 다음 섹션 미리보기

이번 섹션에서 쿼리를 확장하여 검색 다양성을 높이는 방법을 배웠다면, 다음 [8.4 컨텍스트 압축과 재순위화](ch08/session_8.4.md)에서는 검색된 문서 자체를 가공하는 전략을 다룹니다. ContextualCompressionRetriever를 사용하여 긴 문서에서 질문과 관련된 핵심 부분만 추출하고, LLM 기반 재순위화(reranking)로 최종 결과의 정밀도를 극대화하는 방법을 학습합니다. "넓게 검색하고(멀티쿼리), 좁게 압축한다(컨텍스트 압축)"는 조합이 프로덕션 RAG의 핵심 패턴이 됩니다.

## 참고 자료

- [LangChain MultiQueryRetriever 공식 가이드](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) - MultiQueryRetriever의 사용법과 커스터마이징 방법을 다루는 공식 문서
- [LangChain MultiQueryRetriever API 레퍼런스](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html) - `from_llm()`, `include_original` 등 상세 API 명세
- [RAG-Fusion: a New Take on Retrieval-Augmented Generation (arXiv)](https://arxiv.org/abs/2402.03367) - Zackary Rackauckas의 RAG Fusion 원본 논문
- [Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods (ACM)](https://dl.acm.org/doi/10.1145/1571941.1572114) - Cormack et al.의 RRF 원본 논문 (2009 SIGIR)
- [LangChain RAG Fusion 템플릿](https://python.langchain.com/v0.1/docs/templates/rag-fusion/) - LangChain에서 제공하는 RAG Fusion 구현 템플릿

---
### 🔗 Related Sessions
- [lcel](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [faiss](../07-임베딩과-벡터-스토어/03-벡터-스토어-구축---faiss와-chroma.md) (prerequisite)
- [vectorstoreretriever](../08-검색기retriever-심화/01-검색기-기초.md) (prerequisite)
- [ensembleretriever](../07-임베딩과-벡터-스토어/05-벡터-검색-최적화.md) (prerequisite)
- [rrf](../08-검색기retriever-심화/02-키워드와-앙상블-검색.md) (prerequisite)
