# 고급 RAG 패턴

> 기본 RAG의 한계를 뛰어넘는 4가지 고급 검색 전략 — Parent Document Retriever, HyDE, 재랭킹, 쿼리 분해를 마스터합니다.

## 개요

이 섹션에서는 기본 RAG 파이프라인의 검색 품질을 극적으로 향상시키는 **4가지 고급 패턴**을 학습합니다. [기본 RAG 체인 구축](./01-기본-rag-체인-구축.md)에서 만든 단순 벡터 검색을 넘어, 검색 정밀도와 문맥 풍부함을 동시에 잡는 실전 기법들을 다룹니다.

**선수 지식**:
- 기본 RAG 파이프라인과 LCEL 체인 구성 (Session 9.1)
- 벡터 스토어와 임베딩의 동작 원리 (Chapter 7)
- 검색기(Retriever) 인터페이스와 유사도 검색 (Chapter 8)
- `ContextualCompressionRetriever` 개념 (Chapter 8)

**학습 목표**:
- `ParentDocumentRetriever`로 검색 정밀도와 문맥 풍부함을 동시에 확보할 수 있다
- HyDE(Hypothetical Document Embeddings)로 쿼리-문서 간 의미 격차를 해소할 수 있다
- Cross-encoder 재랭킹으로 검색 결과의 정확도를 높일 수 있다
- 복잡한 질문을 서브 쿼리로 분해하여 포괄적인 검색을 수행할 수 있다

## 왜 알아야 할까?

Session 9.1에서 만든 기본 RAG는 "질문 → 벡터 검색 → LLM 생성" 이라는 단순한 구조였습니다. 이걸 **Naive RAG**라고 부르는데요, 실무에서 이 구조만으로는 금방 벽에 부딪힙니다.

어떤 벽이냐고요? 세 가지입니다:

1. **청크 크기 딜레마**: 작은 청크는 임베딩이 정확하지만 LLM에 전달할 문맥이 부족하고, 큰 청크는 문맥은 풍부하지만 검색 정밀도가 떨어집니다.
2. **쿼리-문서 의미 격차**: 사용자의 짧은 질문과 긴 문서는 의미 공간에서 서로 멀리 떨어져 있어, 관련 문서를 놓치기 쉽습니다.
3. **단일 쿼리의 한계**: 복잡한 질문 하나로는 필요한 모든 정보를 검색할 수 없습니다.

2024-2025년 RAG 생태계는 이런 문제들을 해결하는 **Advanced RAG** 패턴이 폭발적으로 발전했습니다. 이 섹션에서 다루는 4가지 패턴은 그중 가장 널리 검증된, 프로덕션에서 바로 쓸 수 있는 기법들입니다.

## 핵심 개념

### 개념 1: Parent Document Retriever — 작은 눈으로 찾고, 큰 손으로 집는다

> 💡 **비유**: 도서관에서 책을 찾는 상황을 상상해보세요. 색인 카드(작은 청크)로는 원하는 내용이 어디에 있는지 정확히 찾을 수 있지만, 색인 카드만 읽어서는 맥락을 알 수 없죠. 실제로는 색인 카드로 위치를 찾은 뒤, 해당 **챕터 전체**(부모 문서)를 꺼내서 읽습니다. `ParentDocumentRetriever`가 바로 이 전략입니다 — **작은 청크로 찾고, 큰 청크를 반환합니다**.

기본 RAG에서 청크 크기를 정할 때 항상 트레이드오프가 있었습니다:

| 전략 | 장점 | 단점 |
|------|------|------|
| 작은 청크 (200-400자) | 임베딩 정밀도 높음 | LLM에 전달할 문맥 부족 |
| 큰 청크 (1500-2000자) | 풍부한 문맥 제공 | 검색 정밀도 저하, 노이즈 증가 |

`ParentDocumentRetriever`는 **두 마리 토끼를 동시에 잡습니다**. 핵심 아이디어는 간단합니다:

1. 문서를 **큰 부모 청크**(예: 2000자)로 먼저 분할
2. 각 부모 청크를 다시 **작은 자식 청크**(예: 400자)로 분할
3. 자식 청크만 벡터 스토어에 임베딩
4. 검색 시: 자식 청크로 유사도 검색 → 매칭된 자식의 **부모 청크**를 반환

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 부모 청크: 큰 단위 (LLM에 전달할 문맥)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# 자식 청크: 작은 단위 (임베딩 검색용)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 벡터 스토어 (자식 청크 검색용) + 문서 저장소 (부모 청크 보관용)
vectorstore = FAISS.from_texts(["init"], OpenAIEmbeddings())  # 초기화
docstore = InMemoryStore()  # 부모 청크를 ID로 저장

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,  # None이면 원본 문서 전체가 부모
)

# 문서 인덱싱 — 내부에서 부모/자식 분할 + 저장이 자동으로 진행
retriever.add_documents(documents)

# 검색 — 자식으로 찾고, 부모를 반환!
results = retriever.invoke("LangChain의 주요 특징은?")
# results에는 2000자짜리 부모 청크가 들어 있음
```

> 🔥 **실무 팁**: `parent_splitter`를 `None`으로 설정하면 원본 문서 전체가 부모가 됩니다. 짧은 문서(블로그 글, FAQ 등)에는 이 방식이 더 효과적일 수 있어요. 반대로 긴 논문이나 기술 문서라면 부모 청크 크기를 명시적으로 설정하는 게 좋습니다.

### 개념 2: HyDE — 가상의 답변으로 진짜 문서를 찾는다

> 💡 **비유**: 미술관에서 특정 그림을 찾고 싶은데 제목을 모른다고 해보세요. "노을이 지는 바다 풍경화"라고 설명하는 것보다, 직접 비슷한 그림을 **대충 스케치해서** "이것과 비슷한 그림 있나요?"라고 물어보는 게 훨씬 효과적이겠죠. HyDE가 바로 이 전략입니다 — LLM이 **가상의 답변 문서를 먼저 생성**하고, 그 문서의 임베딩으로 실제 문서를 검색합니다.

HyDE(Hypothetical Document Embeddings)는 쿼리와 문서 사이의 **의미 격차(Semantic Gap)** 문제를 해결합니다. 사용자 질문은 보통 짧고 추상적인 반면, 문서는 길고 구체적이거든요. 이 둘의 임베딩은 벡터 공간에서 상당히 떨어져 있을 수 있습니다.

HyDE의 동작 과정:
```
사용자 쿼리: "RAG에서 검색 품질을 높이는 방법은?"
     ↓ (1) LLM이 가상 답변 생성
가상 문서: "RAG 시스템의 검색 품질을 높이기 위해서는 먼저 청킹 전략을
최적화해야 합니다. 재랭킹(reranking)을 도입하면 초기 검색 결과의
정밀도를 크게 향상시킬 수 있으며, 쿼리 확장 기법도 효과적입니다..."
     ↓ (2) 가상 문서를 임베딩
가상 문서 벡터: [0.12, -0.34, 0.56, ...]
     ↓ (3) 이 벡터로 실제 문서 검색
실제 문서들: [관련 문서 1, 관련 문서 2, ...]
```

핵심 통찰은 이렇습니다: 가상 문서가 사실과 다를 수 있지만, **문서의 "느낌"과 "패턴"은 실제 답변 문서와 유사**하기 때문에 임베딩 공간에서 더 가까이 위치하게 됩니다.

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o", temperature=0)
base_embeddings = OpenAIEmbeddings()

# HyDE 임베딩 생성기 — 기존 임베딩을 "감싸는" 래퍼
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"  # 도메인에 맞는 프롬프트 선택
)

# 기존 임베딩 대신 hyde_embeddings를 사용 — 드롭인 교체!
vectorstore = FAISS.from_documents(documents, hyde_embeddings)

# 검색 시: 쿼리 → 가상 문서 생성 → 임베딩 → 유사도 검색
results = vectorstore.similarity_search("RAG에서 검색 품질을 높이는 방법은?")
```

> ⚠️ **흔한 오해**: "HyDE가 만든 가상 문서가 틀리면 검색 결과도 엉망 아닌가요?" — 놀랍게도 아닙니다! HyDE의 원 논문에서 밝혔듯이, 가상 문서의 **사실적 정확성은 중요하지 않습니다**. 중요한 건 문서의 의미적 패턴(relevance pattern)이고, 이건 LLM이 꽤 잘 잡아냅니다. 다만 HyDE는 **매 검색마다 LLM을 호출**하므로 지연(latency)과 비용이 증가한다는 점은 고려해야 합니다.

### 개념 3: 재랭킹(Reranking) — 1차 합격자를 다시 면접보다

> 💡 **비유**: 대학 입시를 떠올려보세요. 수능(bi-encoder)으로 수만 명 중 1차 합격자를 빠르게 선별하고, 면접(cross-encoder)으로 그중 최종 합격자를 정밀하게 가려냅니다. 수능만으로 최종 합격을 정하면 실력 있는 학생을 놓칠 수 있고, 모든 지원자를 면접보면 시간이 너무 오래 걸리죠. **2단계 전략**이 핵심입니다.

기본 벡터 검색은 **bi-encoder** 방식입니다. 쿼리와 문서를 각각 독립적으로 임베딩한 뒤 코사인 유사도를 계산하죠. 빠르지만 정밀도에 한계가 있습니다.

**Cross-encoder**는 쿼리와 문서를 **함께** 트랜스포머에 넣어서 관련성 점수를 매깁니다. 훨씬 정확하지만, 모든 문서에 대해 이걸 할 수는 없습니다(N개 문서면 N번의 forward pass가 필요하니까요).

그래서 **2단계 검색(Two-stage Retrieval)** 패턴을 씁니다:

```
1단계: bi-encoder로 상위 20-50개 후보 빠르게 검색 (밀리초)
2단계: cross-encoder로 후보들을 재랭킹하여 상위 3-5개 선별 (100-500ms)
```

LangChain에서는 `ContextualCompressionRetriever`로 이 패턴을 깔끔하게 구현합니다:

**방법 A: 로컬 Cross-Encoder (무료, HuggingFace 모델)**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 로컬에서 실행되는 cross-encoder 모델
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# 재랭킹 압축기: 상위 5개만 선별
compressor = CrossEncoderReranker(model=model, top_n=5)

# 기본 검색기를 감싸서 재랭킹 적용
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})  # 1단계: 20개 검색
)

# 20개 검색 → cross-encoder로 재랭킹 → 상위 5개 반환
results = reranking_retriever.invoke("트랜스포머 아키텍처의 핵심 원리")
```

**방법 B: Cohere Rerank API (클라우드, 다국어 지원)**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Cohere API 기반 재랭킹 (한국어도 지원!)
compressor = CohereRerank(
    model="rerank-multilingual-v3.0",  # 다국어 모델
    top_n=5
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)
```

| 방법 | 장점 | 단점 |
|------|------|------|
| HuggingFace Cross-Encoder | 무료, 오프라인 가능 | GPU 권장, 모델 다운로드 필요 |
| Cohere Rerank | 간편, 다국어 강력 | API 비용, 네트워크 의존 |

### 개념 4: 쿼리 분해 — 복잡한 질문을 잘게 쪼개기

> 💡 **비유**: 의사에게 "머리도 아프고, 소화도 안 되고, 무릎도 아파요"라고 한꺼번에 말하면 의사도 당황하겠죠. 하나씩 물어보면 각 증상에 맞는 정확한 진단을 받을 수 있습니다. 쿼리 분해도 마찬가지 — 복잡한 질문을 **개별 하위 질문으로 분리**해서 각각에 대해 정확한 문서를 검색합니다.

복잡한 질문 하나로 검색하면 모든 측면을 커버하는 문서를 찾기 어렵습니다. 쿼리 분해에는 두 가지 주요 전략이 있습니다:

**전략 1: Multi-Query Retriever — 같은 질문, 다른 표현**

LLM이 원본 질문을 여러 가지 다른 관점으로 **재표현(rephrase)** 합니다. 각 변형 쿼리로 검색한 결과를 합집합(union)으로 모읍니다.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Multi-Query: 하나의 질문 → 여러 변형 생성 → 각각 검색 → 합집합
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
    include_original=True  # 원본 쿼리도 포함
)

# 내부적으로 3-5개의 쿼리 변형을 자동 생성
results = multi_query_retriever.invoke("LangChain에서 RAG 성능을 최적화하려면?")
# "RAG 검색 품질 개선 방법", "LangChain RAG 파이프라인 튜닝", ... 등으로 변형
```

**전략 2: 쿼리 분해(Query Decomposition) — 복합 질문을 서브 질문으로**

복합 질문을 독립적인 하위 질문으로 분해합니다. 각 서브 쿼리에 대해 별도로 검색한 뒤 결과를 종합합니다.

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 서브 쿼리 스키마 정의
class SubQueries(BaseModel):
    """사용자 질문을 분해한 하위 질문 목록"""
    queries: list[str] = Field(
        description="각각 하나의 개념에 집중하는 독립적인 하위 질문 리스트"
    )

# 분해 프롬프트
decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "주어진 질문을 검색에 적합한 독립적인 하위 질문들로 분해하세요. "
     "각 하위 질문은 하나의 개념이나 사실에만 집중해야 합니다. "
     "2-4개의 하위 질문을 생성하세요."),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 구조화된 출력으로 서브 쿼리 생성
decomposition_chain = decomposition_prompt | llm.with_structured_output(SubQueries)

# 사용 예시
sub_queries = decomposition_chain.invoke({
    "question": "HyDE와 Cross-Encoder 재랭킹의 장단점을 비교하고, "
                "어떤 상황에서 각각을 쓰는 게 좋은지 알려줘"
})
# sub_queries.queries → [
#   "HyDE의 장점과 단점은?",
#   "Cross-Encoder 재랭킹의 장점과 단점은?",
#   "HyDE와 재랭킹을 각각 적합한 사용 사례는?"
# ]

# 각 서브 쿼리로 검색하고 결과 합치기
all_docs = []
seen_ids = set()
for query in sub_queries.queries:
    docs = retriever.invoke(query)
    for doc in docs:
        doc_id = doc.page_content[:100]  # 간단한 중복 제거
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            all_docs.append(doc)
```

## 실습: 직접 해보기

4가지 고급 패턴을 하나의 통합 시스템에 조합하는 실습입니다. Parent Document Retriever로 검색하고, 재랭킹으로 정밀도를 높이는 파이프라인을 구축해봅시다.

```python
"""
고급 RAG 패턴 통합 실습
- Parent Document Retriever로 검색 정밀도 + 문맥 풍부함 확보
- Cross-Encoder 재랭킹으로 최종 결과 정밀 선별
- Multi-Query로 검색 커버리지 확대
"""

import os
from dotenv import load_dotenv

load_dotenv()

# === 1. 필수 임포트 ===
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import (
    ParentDocumentRetriever,
    ContextualCompressionRetriever,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# === 2. 샘플 문서 준비 ===
documents = [
    Document(
        page_content=(
            "RAG(Retrieval-Augmented Generation)는 외부 지식을 활용하여 "
            "LLM의 응답 품질을 높이는 기술입니다. 기본 RAG 파이프라인은 "
            "문서 로드, 텍스트 분할, 임베딩, 검색, 생성의 5단계로 구성됩니다. "
            "검색 단계에서 벡터 유사도를 사용하며, 생성 단계에서 검색된 문서를 "
            "컨텍스트로 활용합니다. 청크 크기와 검색 전략이 전체 성능에 큰 영향을 "
            "미치며, 프로덕션 환경에서는 다양한 최적화 기법이 필요합니다."
        ),
        metadata={"source": "rag_overview.md", "section": "기본 개념"},
    ),
    Document(
        page_content=(
            "HyDE(Hypothetical Document Embeddings)는 검색 품질을 향상시키는 "
            "쿼리 변환 기법입니다. 사용자의 짧은 질문을 LLM으로 가상의 답변 문서로 "
            "변환한 뒤, 그 문서의 임베딩으로 실제 문서를 검색합니다. 2022년 Luyu Gao "
            "등이 발표한 논문에서 제안되었으며, zero-shot 환경에서도 fine-tuned "
            "리트리버 수준의 성능을 달성했습니다. 다만 매 검색마다 LLM 호출이 "
            "필요하므로 지연 시간과 비용이 증가하는 단점이 있습니다."
        ),
        metadata={"source": "advanced_rag.md", "section": "HyDE"},
    ),
    Document(
        page_content=(
            "Cross-encoder 재랭킹은 2단계 검색 전략의 핵심입니다. "
            "1단계에서 bi-encoder(임베딩 모델)로 후보 문서를 빠르게 검색하고, "
            "2단계에서 cross-encoder가 쿼리-문서 쌍을 함께 평가하여 정밀 점수를 매깁니다. "
            "BAAI/bge-reranker, Cohere Rerank 등 다양한 모델이 있으며, "
            "LangChain의 ContextualCompressionRetriever와 함께 사용합니다. "
            "재랭킹 추가로 100-500ms의 지연이 발생하지만, 검색 정밀도가 "
            "크게 향상되어 최종 답변 품질이 눈에 띄게 개선됩니다."
        ),
        metadata={"source": "advanced_rag.md", "section": "재랭킹"},
    ),
    Document(
        page_content=(
            "Multi-Query Retriever는 단일 질문을 여러 관점의 질문으로 변형하여 "
            "검색 커버리지를 높이는 기법입니다. LLM이 원본 질문의 대안적 표현을 "
            "3-5개 생성하고, 각각으로 검색한 결과를 합집합으로 반환합니다. "
            "거리 기반 유사도 검색의 한계를 극복하며, 다양한 관련 문서를 "
            "더 폭넓게 찾아낼 수 있습니다. LangChain의 MultiQueryRetriever로 "
            "쉽게 구현할 수 있으며, include_original 옵션으로 원본 쿼리도 "
            "포함할 수 있습니다."
        ),
        metadata={"source": "advanced_rag.md", "section": "쿼리 변환"},
    ),
    Document(
        page_content=(
            "Parent Document Retriever는 청크 크기의 딜레마를 해결합니다. "
            "작은 자식 청크를 임베딩하여 정밀한 검색을 수행하고, 검색된 자식의 "
            "부모 청크를 반환하여 풍부한 컨텍스트를 제공합니다. "
            "child_splitter와 parent_splitter를 별도로 설정할 수 있으며, "
            "InMemoryStore에 부모 문서를 저장합니다. parent_splitter를 None으로 "
            "설정하면 원본 문서 전체가 부모로 사용되어 FAQ 같은 짧은 문서에 적합합니다."
        ),
        metadata={"source": "advanced_rag.md", "section": "Parent Document"},
    ),
]

# === 3. Parent Document Retriever 구성 ===
print("=== Parent Document Retriever 구성 ===")

embeddings = OpenAIEmbeddings()

# 부모/자식 분할기
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,          # 부모: 큰 단위 (LLM에 전달할 문맥)
    chunk_overlap=200
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,           # 자식: 작은 단위 (정밀 검색용)
    chunk_overlap=50
)

# 벡터 스토어 + 문서 저장소
vectorstore = FAISS.from_texts(["placeholder"], embeddings)
docstore = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 문서 인덱싱
parent_retriever.add_documents(documents)
print(f"문서 {len(documents)}개 인덱싱 완료")

# === 4. 재랭킹 적용 ===
print("\n=== 재랭킹 적용 ===")

# Cross-Encoder 모델 로드 (로컬 실행)
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# Parent Document Retriever + 재랭킹 조합
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=parent_retriever  # 부모 문서 검색 → 재랭킹
)

# === 5. Multi-Query + 재랭킹 조합 ===
print("\n=== Multi-Query + 재랭킹 조합 ===")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=reranking_retriever,  # 재랭킹이 적용된 검색기를 래핑
    llm=llm,
    include_original=True
)

# === 6. 완전한 RAG 체인 구성 ===
print("\n=== 완전한 고급 RAG 체인 ===")

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.\n"
     "주어진 컨텍스트만을 기반으로 답변하세요.\n"
     "컨텍스트에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 답하세요.\n\n"
     "컨텍스트:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    """검색된 문서를 포맷팅"""
    return "\n\n---\n\n".join(
        f"[출처: {doc.metadata.get('source', '알 수 없음')}]\n{doc.page_content}"
        for doc in docs
    )

# LCEL로 고급 RAG 체인 구성
advanced_rag_chain = (
    {
        "context": multi_query_retriever | format_docs,  # 다중 쿼리 + 재랭킹
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# === 7. 테스트 ===
print("\n=== 테스트 실행 ===\n")

test_questions = [
    "RAG에서 검색 품질을 향상시키는 기법들을 설명해줘",
    "HyDE의 동작 원리와 장단점은?",
    "Parent Document Retriever는 어떤 문제를 해결하나요?",
]

for question in test_questions:
    print(f"Q: {question}")
    answer = advanced_rag_chain.invoke(question)
    print(f"A: {answer}\n")
    print("-" * 60 + "\n")
```

## 더 깊이 알아보기

### HyDE의 탄생 — "틀린 답이 좋은 검색을 만든다"

HyDE는 2022년 12월, Carnegie Mellon University의 **Luyu Gao** 등이 발표한 논문 *"Precise Zero-Shot Dense Retrieval without Relevance Labels"*에서 처음 제안되었습니다. 이 논문의 핵심 발견은 직관에 반하는 것이었는데요 — LLM이 생성한 가상의 답변이 **사실적으로 틀려도**, 그 임베딩은 실제 정답 문서와 가까운 위치에 놓인다는 것이었습니다.

이건 마치 범인의 몽타주와 비슷합니다. 몽타주가 범인과 100% 똑같지 않아도 "이런 느낌의 사람"이라는 패턴을 전달하면 실제 범인을 찾는 데 큰 도움이 되는 것처럼, 가상 문서도 "이런 느낌의 문서"라는 의미적 패턴을 임베딩에 담아내는 거죠.

놀랍게도 HyDE는 fine-tuning 없이도(zero-shot) 웹 검색, QA, 사실 검증 등 다양한 벤치마크에서 기존 fine-tuned 리트리버와 대등하거나 더 나은 성능을 보여주었고, 한국어와 일본어를 포함한 다국어에서도 효과가 있음이 검증되었습니다. 이 논문은 2023년 ACL(자연어 처리 분야 최고 학회)에 게재되었습니다.

### 2단계 검색의 기원 — 정보 검색(IR) 50년의 지혜

재랭킹은 사실 RAG보다 훨씬 오래된 개념입니다. 정보 검색(Information Retrieval) 분야에서는 1990년대부터 **cascade retrieval** 또는 **telescoping evaluation**이라는 이름으로 2단계 검색을 연구해왔습니다. 먼저 빠르고 대략적인 모델(BM25 같은 키워드 매칭)로 후보를 줄이고, 느리지만 정밀한 모델로 최종 순위를 정하는 전략이죠.

2019년 BERT가 등장하면서 cross-encoder 재랭킹이 비약적으로 발전했고, 2024-2025년에는 Cohere Rerank v3, BGE Reranker 같은 전문 모델들이 RAG 파이프라인의 필수 구성 요소로 자리잡았습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "고급 패턴을 모두 적용하면 항상 더 좋은 결과를 얻는다" — 아닙니다! 각 기법은 **특정 문제를 해결**하기 위한 것입니다. 청크 크기가 문제라면 Parent Document Retriever를, 쿼리-문서 의미 격차가 문제라면 HyDE를, 검색 결과의 정밀도가 문제라면 재랭킹을 적용하세요. 무분별한 중첩은 지연 시간만 늘리고 효과는 미미할 수 있습니다. **먼저 기본 RAG의 어디가 약한지 진단**하고, 맞는 약을 처방하세요.

> 💡 **알고 계셨나요?**: HyDE에서 사용하는 `prompt_key` 파라미터에는 `"web_search"`, `"sci_fact"`, `"fiqa"`, `"trec_covid"` 등 도메인별 프롬프트가 내장되어 있습니다. 도메인에 맞는 프롬프트를 선택하면 가상 문서의 품질이 크게 달라지며, 커스텀 프롬프트를 직접 만들어 쓸 수도 있습니다.

> 🔥 **실무 팁**: 재랭킹의 `k`(초기 검색 개수)와 `top_n`(최종 반환 개수)의 비율이 중요합니다. 경험적으로 **k=20, top_n=3~5** 정도가 좋은 출발점이에요. k가 너무 작으면 재랭킹할 후보 자체가 부실하고, 너무 크면 재랭킹 시간이 길어집니다. 그리고 한국어 문서를 다룬다면 Cohere의 `rerank-multilingual-v3.0` 모델이 로컬 cross-encoder보다 성능이 좋은 경우가 많으니 비교해보세요.

> 🔥 **실무 팁**: `ParentDocumentRetriever`의 `docstore`로 `InMemoryStore`를 쓰면 프로세스가 종료되면 부모 문서가 사라집니다. 프로덕션에서는 Redis, MongoDB 등 영속적 저장소를 사용하세요. LangChain은 `RedisStore`, `MongoDBStore` 등을 제공합니다.

## 핵심 정리

| 개념 | 해결하는 문제 | 핵심 원리 | LangChain 클래스 |
|------|-------------|----------|-----------------|
| Parent Document Retriever | 청크 크기 딜레마 | 작은 청크로 검색, 큰 부모 청크 반환 | `ParentDocumentRetriever` |
| HyDE | 쿼리-문서 의미 격차 | 가상 답변 문서를 생성하여 임베딩으로 검색 | `HypotheticalDocumentEmbedder` |
| Cross-Encoder 재랭킹 | 검색 결과 정밀도 부족 | 2단계 검색: bi-encoder → cross-encoder | `ContextualCompressionRetriever` + `CrossEncoderReranker` |
| Multi-Query Retriever | 단일 쿼리의 한계 | 질문을 다양하게 변형 후 합집합 검색 | `MultiQueryRetriever` |
| 쿼리 분해 | 복합 질문 처리 어려움 | 서브 질문으로 분해 후 개별 검색 | `with_structured_output` + 커스텀 체인 |

## 다음 섹션 미리보기

이번 섹션에서 개별 고급 패턴을 익혔다면, 다음 섹션 **RAG 성능 평가와 최적화**에서는 이 패턴들이 실제로 얼마나 효과적인지 **정량적으로 측정**하는 방법을 배웁니다. Faithfulness(충실도), Relevancy(관련성), Context Recall(문맥 재현율) 같은 RAG 전용 평가 지표를 학습하고, 어떤 패턴 조합이 최적인지 체계적으로 실험하는 프레임워크를 구축합니다.

## 참고 자료

- [LangChain ParentDocumentRetriever 공식 가이드](https://python.langchain.com/docs/how_to/parent_document_retriever/) - 부모 문서 검색기의 사용법과 예제를 공식 문서에서 확인
- [LangChain MultiQueryRetriever 공식 가이드](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) - 다중 쿼리 검색기의 동작 원리와 구현 방법
- [LangChain Cross Encoder Reranker 통합 가이드](https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/) - HuggingFace Cross-Encoder를 LangChain에서 재랭킹에 활용하는 방법
- [Cohere Rerank + LangChain 공식 문서](https://docs.cohere.com/docs/rerank-on-langchain) - Cohere Rerank API를 LangChain에 통합하는 공식 가이드
- [HyDE 원 논문 — Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)](https://arxiv.org/abs/2212.10496) - HyDE 기법의 학술적 기반을 다진 ACL 2023 논문
- [Pinecone — Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/) - 재랭킹의 개념과 실전 적용을 그림과 함께 알기 쉽게 설명

---
### 🔗 Related Sessions
- [lcel](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [embedding](07-임베딩과-벡터-스토어/01-텍스트-임베딩-이해.md) (prerequisite)
- [with_structured_output](03-프롬프트-엔지니어링과-템플릿/05-프롬프트-엔지니어링-실전-기법.md) (prerequisite)
- [contextualcompressionretriever](08-검색기retriever-심화/04-컨텍스트-압축.md) (prerequisite)
- [rag_pipeline](./01-기본-rag-체인-구축.md) (prerequisite)
