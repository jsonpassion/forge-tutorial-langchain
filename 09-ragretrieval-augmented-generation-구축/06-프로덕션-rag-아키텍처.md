# 프로덕션 RAG 아키텍처

> 실험실에서 벗어나 실전 서비스로 — 인덱싱 자동화, 증분 업데이트, 모니터링, 확장성까지 프로덕션 수준의 RAG 시스템을 설계합니다.

## 개요

이 섹션에서는 앞서 구축한 RAG 파이프라인을 **실제 서비스 환경**에 배포할 수 있는 수준으로 끌어올립니다. 프로토타입과 프로덕션 사이에는 인덱싱 자동화, 데이터 동기화, 성능 모니터링, 비용 최적화라는 거대한 간극이 존재하는데요, 이 세션이 바로 그 다리를 놓아줄 겁니다.

**선수 지식**: 
- [9.1 기본 RAG 체인 구축](ch09/session_9_1.md)에서 배운 RAG 파이프라인 5단계
- [9.4 고급 RAG 패턴](ch09/session_9_4.md)의 Parent Document Retriever, 재랭킹
- [9.5 RAG 평가와 개선](ch09/session_9_5.md)의 RAGAS 메트릭과 LangSmith 평가
- [7.5 임베딩 캐싱과 최적화](ch07/session_7_5.md)의 CacheBackedEmbeddings 기본 개념과 로컬 캐시 활용법

**학습 목표**:
- LangChain Indexing API와 RecordManager로 인덱싱 파이프라인을 자동화할 수 있다
- 증분 업데이트(Incremental Update)와 정리(Cleanup) 전략을 적절히 선택할 수 있다
- LangSmith와 커스텀 콜백으로 RAG 시스템을 모니터링할 수 있다
- 캐싱, 비동기 처리, 수평 확장 등 프로덕션 확장성 패턴을 설계할 수 있다

## 왜 알아야 할까?

노트북에서 `retriever.invoke("질문")` 한 줄이면 RAG가 동작합니다. 그런데 이걸 실제 서비스로 올리는 순간, 전혀 다른 세계가 펼쳐지죠.

**"문서가 매일 업데이트되는데, 벡터 스토어는 어떻게 동기화하지?"** — 전체 재인덱싱은 비용과 시간이 어마어마합니다. 100만 건의 문서를 매번 다시 임베딩하면 API 비용만 수십만 원이 들 수 있거든요.

**"답변 품질이 갑자기 떨어졌는데, 어디가 문제인지 모르겠다"** — 임베딩 모델, 청킹 전략, 프롬프트, LLM 중 어디서 병목이 생겼는지 추적할 수 없다면 디버깅은 불가능합니다.

**"사용자가 늘어나면 응답 시간이 급격히 느려진다"** — 캐싱 없이 매 요청마다 임베딩+검색+LLM 호출을 반복하면, 동시 사용자 100명만 돼도 시스템이 허덕입니다.

프로덕션 RAG는 이 세 가지 질문에 대한 답을 갖추고 있어야 합니다. 이번 세션에서 하나씩 해결해 보겠습니다.

## 핵심 개념

### 개념 1: 인덱싱 파이프라인 자동화 — RecordManager와 Indexing API

> 💡 **비유**: 도서관의 사서를 떠올려 보세요. 새로운 책이 들어오면 사서는 기존 장서 목록을 확인하고, 이미 있는 책은 건너뛰고, 개정판이 나왔으면 이전 판을 빼고 새 판을 꽂습니다. LangChain의 `RecordManager`가 바로 이 사서 역할을 합니다 — 어떤 문서가 이미 벡터 스토어에 있는지, 언제 추가되었는지, 내용이 바뀌었는지를 추적하죠.

LangChain의 **Indexing API**는 `index()` 함수와 `RecordManager`의 조합으로 동작합니다. RecordManager는 각 문서의 **해시값**, **기록 시간**, **소스 ID**를 저장하여, 다음 인덱싱 실행 시 변경된 문서만 처리할 수 있게 합니다.

핵심은 **cleanup 모드** 선택입니다:

| 모드 | 동작 | 적합한 상황 |
|------|------|-------------|
| `None` | 중복 제거만, 삭제 없음 | 데이터가 추가만 되는 경우 |
| `"incremental"` | 소스 ID 기준으로 변경/삭제 감지, 즉시 정리 | 실시간 업데이트가 필요한 서비스 |
| `"full"` | 이번 실행에 포함되지 않은 모든 문서 삭제 | 배치 재인덱싱, 전체 동기화 |
| `"scoped_full"` | full과 유사하나 배치 단위로 동작 | 대용량 데이터셋의 배치 처리 |

```python
from langchain_core.indexing import RecordManager, index
from langchain.indexes import SQLRecordManager
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 스토어 생성
vectorstore = Chroma(
    collection_name="my_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# RecordManager 생성 — SQLite 기반 (프로덕션에서는 PostgreSQL 권장)
record_manager = SQLRecordManager(
    namespace="my_documents",          # 네임스페이스로 컬렉션 구분
    db_url="sqlite:///record_manager.db",
)

# 스키마 생성 (최초 1회)
record_manager.create_schema()
```

### 개념 2: 증분 업데이트 전략

> 💡 **비유**: 뉴스 사이트를 운영한다고 생각해 보세요. 매시간 새 기사가 올라오고, 기존 기사가 수정되고, 오래된 기사는 삭제됩니다. 매번 모든 기사를 처음부터 다시 인덱싱하는 건 마치 도서관 전체를 불태우고 다시 짓는 것과 같죠. **증분 업데이트**는 "변경된 부분만 골라서 처리"하는 현명한 접근입니다.

`index()` 함수는 문서의 **콘텐츠 해시**를 계산하여 변경 여부를 판단합니다. 같은 내용이면 건너뛰고, 내용이 바뀌었으면 이전 버전을 삭제하고 새 버전을 추가합니다.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 문서 로더 + 텍스트 분할기
loader = DirectoryLoader(
    "./docs/",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# 문서 로드 및 분할
docs = loader.load()
split_docs = text_splitter.split_documents(docs)

# 증분 인덱싱 실행
# source_id_key: 각 Document의 metadata에서 소스를 식별할 키
indexing_result = index(
    docs_source=split_docs,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="incremental",       # 증분 모드
    source_id_key="source",      # metadata["source"]로 소스 문서 식별
)

print(indexing_result)
# {'num_added': 42, 'num_updated': 3, 'num_skipped': 150, 'num_deleted': 1}
```

**`incremental` vs `full`의 핵심 차이점**은 **삭제 처리**에 있습니다:

```python
# incremental 모드: 소스 파일이 삭제되어도 벡터 스토어에서 자동 삭제하지 않음
# → 로더가 반환한 소스 ID에 대해서만 변경/삭제 감지
indexing_result = index(
    docs_source=split_docs,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

# full 모드: 이번 실행에 포함되지 않은 모든 문서를 벡터 스토어에서 삭제
# → 완전한 동기화 보장, 단 인덱싱 중 일시적 중복 가능
indexing_result = index(
    docs_source=split_docs,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="full",
    source_id_key="source",
)
```

> ⚠️ **흔한 오해**: "incremental 모드면 삭제된 문서도 자동으로 처리되겠지?" — 아닙니다! `incremental` 모드는 로더가 반환한 소스 ID에 대해서만 변경을 감지합니다. 원본 파일이 삭제되어 로더가 아예 해당 문서를 반환하지 않으면, 벡터 스토어에 **유령 문서(ghost document)**로 남습니다. 완전한 동기화가 필요하다면 주기적으로 `full` 모드를 실행하세요.

### 개념 3: 프로덕션 임베딩 캐시 — Redis 기반 분산 캐싱과 네임스페이스 전략

[7.5 임베딩 캐싱과 최적화](ch07/session_7_5.md)에서 `CacheBackedEmbeddings`의 기본 개념과 `InMemoryByteStore`, `LocalFileStore`를 활용한 로컬 캐싱을 배웠습니다. 여기서는 **프로덕션 환경에서 실제로 부딪히는 문제들** — 다중 서버 간 캐시 공유, 멀티테넌트 격리, 모델 마이그레이션 — 을 해결하는 패턴에 집중합니다.

**왜 로컬 캐시로는 부족한가?**

단일 서버에서는 `LocalFileStore`로 충분하지만, 프로덕션에서는 여러 워커/서버가 동시에 인덱싱과 쿼리를 처리합니다. 각 서버가 자체 로컬 캐시를 유지하면 **캐시 히트율이 분산**되어 비용 절감 효과가 크게 줄어들죠. Redis 기반 중앙 캐시는 모든 서버가 하나의 캐시를 공유하므로 히트율을 극대화합니다.

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import RedisStore
from langchain_openai import OpenAIEmbeddings

underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── 프로덕션 Redis 캐시 구성 ────────────────────────────
# 네임스페이스 전략: "{모델명}:{테넌트/컬렉션}" 형태로 계층 구분
redis_store = RedisStore(
    redis_url="redis://localhost:6379",
    namespace="text-embedding-3-small:production_docs",
)

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_embedding_cache=redis_store,
    namespace=underlying_embeddings.model,  # 모델별 캐시 분리
)
```

**네임스페이스 전략 — 캐시 충돌 없이 다중 환경 운영하기**

프로덕션에서는 다양한 이유로 캐시를 격리해야 합니다. `namespace` 매개변수가 이 역할을 하는데, 체계적인 네이밍 규칙이 핵심입니다:

```python
# 패턴 1: 모델 버전별 격리 — 모델 교체 시 캐시 충돌 방지
# text-embedding-3-small → text-embedding-3-large 마이그레이션 시
# 이전 캐시를 보존하면서 새 모델의 캐시를 독립적으로 구축
old_store = RedisStore(
    redis_url="redis://localhost:6379",
    namespace="emb:v1:text-embedding-3-small",
)
new_store = RedisStore(
    redis_url="redis://localhost:6379",
    namespace="emb:v1:text-embedding-3-large",
)

# 패턴 2: 멀티테넌트 격리 — SaaS 환경에서 고객사별 캐시 분리
def get_tenant_embeddings(tenant_id: str) -> CacheBackedEmbeddings:
    """테넌트별 격리된 임베딩 캐시를 반환합니다."""
    store = RedisStore(
        redis_url="redis://localhost:6379",
        namespace=f"emb:{underlying_embeddings.model}:tenant:{tenant_id}",
    )
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embeddings,
        document_embedding_cache=store,
        namespace=underlying_embeddings.model,
    )

# 패턴 3: 환경별 격리 — staging과 production 캐시 분리
import os
env = os.getenv("DEPLOY_ENV", "dev")  # dev, staging, production
store = RedisStore(
    redis_url=os.getenv("REDIS_URL"),
    namespace=f"emb:{env}:{underlying_embeddings.model}",
)
```

**캐시 워밍과 무중단 모델 마이그레이션**

임베딩 모델을 교체할 때 가장 위험한 순간은 **새 모델의 캐시가 아직 비어 있는 때**입니다. 갑자기 모든 요청이 API 호출로 폭주하면 비용과 지연이 급증하죠. 이를 방지하는 패턴은 다음과 같습니다:

```python
async def warm_cache_for_new_model(
    documents: list[str],
    new_embeddings: CacheBackedEmbeddings,
    batch_size: int = 100,
) -> int:
    """새 모델의 캐시를 사전에 채워 무중단 전환을 준비합니다."""
    total_warmed = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        # embed_documents 호출 시 캐시에 자동 저장
        await new_embeddings.aembed_documents(batch)
        total_warmed += len(batch)
        logger.info(f"캐시 워밍 진행: {total_warmed}/{len(documents)}")
    return total_warmed

# 마이그레이션 절차:
# 1단계: 새 모델 캐시를 백그라운드로 워밍 (기존 모델로 서비스 유지)
# 2단계: 캐시 히트율이 충분히 올라오면 트래픽을 새 모델로 전환
# 3단계: 이전 모델의 캐시 네임스페이스를 일정 기간 후 삭제
```

> ⚠️ **주의**: RedisStore의 기본 TTL은 무제한입니다. 임베딩 캐시는 모델이 바뀌지 않는 한 영구 보관이 합리적이지만, 네임스페이스 전략 없이 무한 축적하면 Redis 메모리가 고갈됩니다. `maxmemory-policy allkeys-lru`를 Redis에 설정하거나, 사용하지 않는 네임스페이스를 주기적으로 정리하세요.

### 개념 4: 모니터링과 로깅 — LangSmith 통합

> 💡 **비유**: 자동차 계기판을 생각해 보세요. 속도, RPM, 연료량, 엔진 온도를 한눈에 볼 수 있어야 안전하게 운전할 수 있죠. RAG 시스템도 마찬가지입니다. 검색 지연 시간, 답변 품질, 토큰 소비량, 에러율을 실시간으로 볼 수 있어야 장애가 나기 전에 대응할 수 있습니다.

LangSmith는 LangChain 애플리케이션의 **관찰 가능성(Observability)** 플랫폼으로, 환경 변수 하나만 설정하면 모든 LangChain 호출을 자동으로 트레이싱합니다.

```python
import os

# LangSmith 트레이싱 활성화 — 이것만으로 모든 호출이 추적됩니다
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "ls_..."  # .env로 관리
os.environ["LANGSMITH_PROJECT"] = "production-rag"
```

하지만 프로덕션에서는 **커스텀 메트릭**이 필요합니다. LangChain의 콜백(Callback) 시스템으로 검색 품질, 응답 시간, 비용을 세밀하게 추적할 수 있죠.

```python
import time
import logging
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

logger = logging.getLogger("rag_monitor")

class RAGMonitorCallback(BaseCallbackHandler):
    """RAG 파이프라인의 핵심 메트릭을 수집하는 커스텀 콜백."""

    def __init__(self):
        self.retrieval_start: float = 0
        self.llm_start: float = 0
        self.total_tokens: int = 0
        self.retrieved_docs: int = 0

    def on_retriever_start(self, query: str, **kwargs: Any) -> None:
        """검색 시작 시간 기록."""
        self.retrieval_start = time.time()
        logger.info(f"[검색 시작] 쿼리: {query[:100]}...")

    def on_retriever_end(self, documents: list[Document], **kwargs: Any) -> None:
        """검색 완료 — 지연 시간과 문서 수 기록."""
        elapsed = time.time() - self.retrieval_start
        self.retrieved_docs = len(documents)
        logger.info(
            f"[검색 완료] {elapsed:.3f}초, {self.retrieved_docs}개 문서 반환"
        )
        # 프로덕션에서는 Prometheus, DataDog 등으로 메트릭 전송
        if elapsed > 2.0:
            logger.warning(f"[성능 경고] 검색 지연 {elapsed:.3f}초 — 임계값 초과")

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """LLM 호출 시작."""
        self.llm_start = time.time()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLM 호출 완료 — 지연 시간과 토큰 사용량 기록."""
        elapsed = time.time() - self.llm_start
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.total_tokens = token_usage.get("total_tokens", 0)
            logger.info(
                f"[LLM 완료] {elapsed:.3f}초, "
                f"토큰: {self.total_tokens}"
            )
```

### 개념 5: 확장성 패턴 — 캐싱, 비동기, 수평 확장

> 💡 **비유**: 인기 식당을 떠올려 보세요. 손님이 10명일 때는 셰프 한 명이면 충분하지만, 100명이 되면? 주방을 키우고(수평 확장), 자주 나가는 메뉴는 미리 만들어 두고(캐싱), 주문 접수와 요리와 서빙을 동시에 진행해야(비동기 처리) 합니다.

프로덕션 RAG의 확장성은 크게 세 가지 축으로 나뉩니다:

**1. 다계층 캐싱 (Multi-Level Caching)**

```python
from langchain_community.cache import RedisCache, RedisSemanticCache
from langchain_core.globals import set_llm_cache

# 1단계: 정확히 같은 쿼리에 대한 LLM 응답 캐시
set_llm_cache(RedisCache(redis_url="redis://localhost:6379"))

# 2단계: 의미적으로 유사한 쿼리에 대한 시맨틱 캐시
semantic_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    score_threshold=0.95,  # 유사도 95% 이상이면 캐시 히트
)
set_llm_cache(semantic_cache)
```

**2. 비동기 처리 (Async Processing)**

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 비동기 RAG 체인
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "다음 컨텍스트를 바탕으로 질문에 답하세요.\n\n"
    "컨텍스트: {context}\n질문: {question}"
)

async def async_rag_query(question: str, retriever) -> str:
    """비동기 RAG 쿼리 — 검색과 다른 전처리를 동시에 실행."""
    # 검색과 쿼리 분석을 동시에 실행
    docs_task = retriever.ainvoke(question)
    # 다른 비동기 작업(예: 사용자 이력 조회)도 동시 실행 가능
    docs = await docs_task

    context = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"context": context, "question": question})
    return result

# 여러 쿼리를 동시에 처리
async def batch_queries(questions: list[str], retriever):
    """여러 질문을 동시에 처리하여 처리량 극대화."""
    tasks = [async_rag_query(q, retriever) for q in questions]
    return await asyncio.gather(*tasks)
```

**3. 인덱싱과 쿼리 경로 분리**

프로덕션에서 가장 중요한 아키텍처 결정은 **인덱싱 파이프라인**과 **쿼리 파이프라인**을 분리하는 것입니다. 인덱싱은 지연 시간에 관대하지만, 쿼리는 서브초(sub-second) 응답이 필요하거든요.

```
┌─────────────────────────────────────────────────┐
│              인덱싱 파이프라인 (비동기/배치)          │
│                                                 │
│  문서 소스 → 로더 → 분할 → 임베딩 → 벡터 스토어     │
│      ↑                          ↓               │
│  [스케줄러/이벤트]         [RecordManager]         │
│  (Cron, Webhook)          (변경 추적)             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              쿼리 파이프라인 (실시간)               │
│                                                 │
│  사용자 질문 → 캐시 확인 → 임베딩 → 벡터 검색       │
│      ↓                              ↓           │
│  [응답 캐시]                    [LLM 생성]        │
│  (Redis)                      (스트리밍)          │
└─────────────────────────────────────────────────┘
```

## 실습: 직접 해보기

아래는 인덱싱 자동화, 임베딩 캐싱, 모니터링을 통합한 **프로덕션 수준 RAG 시스템**의 완전한 구현입니다. 복사-붙여넣기로 실행할 수 있습니다.

```python
"""
프로덕션 RAG 시스템 — 인덱싱 자동화 + 캐싱 + 모니터링 통합 예제
필요 패키지: pip install langchain langchain-openai langchain-chroma python-dotenv redis
"""
import os
import time
import logging
import hashlib
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager
from langchain_core.indexing import index
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore, RedisStore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("production_rag")

# LangSmith 트레이싱 (선택사항 — API 키가 있으면 자동 활성화)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "production-rag-demo")

# ── 상수 ──────────────────────────────────────────────────
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "production_docs"
CHROMA_DIR = "./chroma_production"
RECORD_MANAGER_DB = "sqlite:///record_manager_production.db"

# 프로덕션 캐시 설정 — 환경에 따라 Redis 또는 로컬 파일 선택
REDIS_URL = os.getenv("REDIS_URL")  # redis://localhost:6379
CACHE_DIR = "./embedding_cache"     # Redis 미사용 시 폴백


# ── 1. 모니터링 콜백 ─────────────────────────────────────
class ProductionRAGCallback(BaseCallbackHandler):
    """프로덕션 RAG 메트릭 수집 콜백."""

    def __init__(self):
        self.metrics: dict[str, Any] = {}
        self._retrieval_start: float = 0
        self._llm_start: float = 0

    def on_retriever_start(self, query: str, **kwargs) -> None:
        self._retrieval_start = time.time()

    def on_retriever_end(self, documents: list[Document], **kwargs) -> None:
        elapsed = time.time() - self._retrieval_start
        self.metrics["retrieval_latency_ms"] = round(elapsed * 1000, 1)
        self.metrics["num_retrieved_docs"] = len(documents)
        logger.info(
            f"검색 완료: {elapsed:.3f}초, {len(documents)}개 문서"
        )

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        self._llm_start = time.time()

    def on_llm_end(self, response: Any, **kwargs) -> None:
        elapsed = time.time() - self._llm_start
        self.metrics["llm_latency_ms"] = round(elapsed * 1000, 1)
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.metrics["total_tokens"] = usage.get("total_tokens", 0)
            self.metrics["prompt_tokens"] = usage.get("prompt_tokens", 0)
            self.metrics["completion_tokens"] = usage.get("completion_tokens", 0)
        logger.info(f"LLM 완료: {elapsed:.3f}초")

    def get_summary(self) -> dict[str, Any]:
        """수집된 메트릭 요약 반환."""
        total = self.metrics.get("retrieval_latency_ms", 0) + \
                self.metrics.get("llm_latency_ms", 0)
        self.metrics["total_latency_ms"] = round(total, 1)
        return self.metrics


# ── 2. 인덱싱 파이프라인 ──────────────────────────────────
class ProductionIndexer:
    """RecordManager 기반 프로덕션 인덱싱 파이프라인."""

    def __init__(self):
        # 임베딩 — Redis 캐시 우선, 미사용 시 로컬 파일 폴백
        base_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        if REDIS_URL:
            # 프로덕션: Redis 분산 캐시 — 다중 서버 간 캐시 공유
            cache_store = RedisStore(
                redis_url=REDIS_URL,
                namespace=f"emb:{COLLECTION_NAME}",
            )
            logger.info(f"Redis 임베딩 캐시 활성화: {REDIS_URL}")
        else:
            # 개발/로컬: 파일 캐시 (Ch7.5에서 배운 방식)
            cache_store = LocalFileStore(CACHE_DIR)
            logger.info(f"로컬 파일 임베딩 캐시 활성화: {CACHE_DIR}")

        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=base_embeddings,
            document_embedding_cache=cache_store,
            namespace=EMBEDDING_MODEL,  # 모델별 캐시 분리
        )

        # 벡터 스토어
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

        # RecordManager
        self.record_manager = SQLRecordManager(
            namespace=COLLECTION_NAME,
            db_url=RECORD_MANAGER_DB,
        )
        self.record_manager.create_schema()

        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def index_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*.md",
        cleanup: str = "incremental",
    ) -> dict:
        """디렉토리의 문서를 인덱싱합니다.

        Args:
            directory: 문서가 있는 디렉토리 경로
            glob_pattern: 파일 매칭 패턴
            cleanup: 정리 모드 ("incremental", "full", None)

        Returns:
            인덱싱 결과 통계
        """
        start = time.time()
        logger.info(f"인덱싱 시작: {directory} (cleanup={cleanup})")

        # 문서 로드
        loader = DirectoryLoader(
            directory,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs = loader.load()
        logger.info(f"로드 완료: {len(docs)}개 문서")

        # 텍스트 분할
        split_docs = self.text_splitter.split_documents(docs)
        logger.info(f"분할 완료: {len(split_docs)}개 청크")

        # 인덱싱 (RecordManager가 변경 추적)
        result = index(
            docs_source=split_docs,
            record_manager=self.record_manager,
            vector_store=self.vectorstore,
            cleanup=cleanup,
            source_id_key="source",
        )

        elapsed = time.time() - start
        result["elapsed_seconds"] = round(elapsed, 2)
        logger.info(
            f"인덱싱 완료: "
            f"추가={result['num_added']}, "
            f"업데이트={result['num_updated']}, "
            f"건너뜀={result['num_skipped']}, "
            f"삭제={result['num_deleted']}, "
            f"소요={elapsed:.2f}초"
        )
        return result

    def get_retriever(self, search_kwargs: dict | None = None):
        """검색기 반환."""
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 4},
        )


# ── 3. 쿼리 파이프라인 ──────────────────────────────────
class ProductionRAGChain:
    """모니터링이 통합된 프로덕션 RAG 체인."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 정확하고 도움이 되는 AI 어시스턴트입니다.\n"
             "주어진 컨텍스트만을 사용하여 질문에 답하세요.\n"
             "컨텍스트에 답이 없으면 '제공된 문서에서 해당 정보를 찾을 수 없습니다'라고 답하세요.\n"
             "답변 끝에 참고한 출처를 표시하세요."),
            ("human",
             "컨텍스트:\n{context}\n\n질문: {question}"),
        ])

        # LCEL 체인 구성
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """문서를 메타데이터 포함 문자열로 변환."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "알 수 없음")
            formatted.append(
                f"[문서 {i}] (출처: {source})\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def query(self, question: str) -> dict[str, Any]:
        """질문에 대한 답변과 메트릭을 반환합니다."""
        callback = ProductionRAGCallback()

        answer = self.chain.invoke(
            question,
            config={"callbacks": [callback]},
        )

        return {
            "answer": answer,
            "metrics": callback.get_summary(),
        }


# ── 4. 실행 ──────────────────────────────────────────────
def main():
    # 테스트용 샘플 문서 생성
    docs_dir = Path("./sample_docs")
    docs_dir.mkdir(exist_ok=True)

    sample_files = {
        "langchain_intro.md": (
            "# LangChain 소개\n\n"
            "LangChain은 LLM 기반 애플리케이션 개발을 위한 프레임워크입니다.\n"
            "LCEL을 사용하면 체인을 선언적으로 구성할 수 있습니다.\n"
            "Runnable 인터페이스는 invoke, stream, batch 메서드를 제공합니다."
        ),
        "rag_basics.md": (
            "# RAG 기초\n\n"
            "RAG는 검색 증강 생성의 약자입니다.\n"
            "외부 지식을 검색하여 LLM 응답의 정확성을 높입니다.\n"
            "인덱싱, 검색, 생성의 3단계로 구성됩니다."
        ),
        "vector_store.md": (
            "# 벡터 스토어\n\n"
            "벡터 스토어는 임베딩 벡터를 저장하고 유사도 검색을 수행합니다.\n"
            "Chroma, FAISS, Pinecone 등 다양한 벡터 DB를 지원합니다.\n"
            "MMR 검색은 관련성과 다양성을 동시에 최적화합니다."
        ),
    }

    for filename, content in sample_files.items():
        (docs_dir / filename).write_text(content, encoding="utf-8")

    # 인덱싱 파이프라인 실행
    indexer = ProductionIndexer()
    result = indexer.index_directory("./sample_docs", glob_pattern="**/*.md")
    print(f"\n📊 인덱싱 결과: {result}")

    # 두 번째 실행 — 변경 없으면 건너뜀 확인
    print("\n--- 두 번째 인덱싱 (변경 없음) ---")
    result2 = indexer.index_directory("./sample_docs", glob_pattern="**/*.md")
    print(f"📊 결과: {result2}")
    # num_skipped만 증가하고 num_added는 0이어야 합니다

    # 문서 수정 후 증분 업데이트
    (docs_dir / "langchain_intro.md").write_text(
        "# LangChain 소개 (업데이트됨)\n\n"
        "LangChain v0.3은 안정적인 프로덕션 배포를 지원합니다.\n"
        "LCEL 체인은 스트리밍과 배치 처리를 기본 지원합니다.\n"
        "Runnable 인터페이스는 모든 컴포넌트의 통합 API입니다.",
        encoding="utf-8",
    )

    print("\n--- 세 번째 인덱싱 (문서 1개 수정) ---")
    result3 = indexer.index_directory("./sample_docs", glob_pattern="**/*.md")
    print(f"📊 결과: {result3}")
    # num_updated가 증가해야 합니다

    # RAG 쿼리
    retriever = indexer.get_retriever(search_kwargs={"k": 3})
    rag = ProductionRAGChain(retriever)

    print("\n--- RAG 쿼리 ---")
    response = rag.query("LangChain의 LCEL이란 무엇인가요?")
    print(f"\n💬 답변: {response['answer']}")
    print(f"📈 메트릭: {response['metrics']}")


if __name__ == "__main__":
    main()
```

## 더 깊이 알아보기

### 인덱싱 자동화의 역사 — "동기화 문제"의 해결

벡터 스토어와 원본 데이터 소스의 **동기화 문제**는 RAG가 등장한 초기부터 골칫거리였습니다. 2023년 초, LangChain 팀은 이 문제를 "Syncing Data Sources to Vector Stores"라는 블로그 포스트에서 공식적으로 다뤘는데요. 당시 대부분의 RAG 시스템은 인덱싱할 때마다 벡터 스토어를 **통째로 비우고 다시 채우는** 방식을 사용했습니다. 수천 건이면 괜찮았지만, 수십만 건이 되자 임베딩 API 비용이 하루에 수백 달러씩 나가는 사태가 벌어졌죠.

LangChain 팀의 해결책은 데이터베이스 세계에서 오랫동안 사용해온 **변경 데이터 캡처(CDC, Change Data Capture)** 패턴에서 영감을 받았습니다. 각 문서의 콘텐츠 해시를 계산하고, 이전 해시와 비교하여 변경된 것만 처리하는 방식이죠. 이것이 바로 `RecordManager`와 `index()` 함수의 탄생 배경입니다.

재미있는 것은, 이 패턴이 사실 Git의 동작 방식과 거의 동일하다는 점입니다. Git도 파일의 SHA-1 해시를 계산하여 변경 여부를 판단하잖아요? RecordManager는 말하자면 "벡터 스토어의 Git"인 셈입니다.

### LLM 관찰 가능성의 부상

2024~2025년 사이, LLM 애플리케이션의 **관찰 가능성(Observability)**은 하나의 독립적인 분야로 성장했습니다. LangSmith 외에도 Langfuse(오픈소스), Weights & Biases Weave, Arize Phoenix 등 다양한 도구가 등장했죠. 이는 전통적인 소프트웨어의 APM(Application Performance Monitoring) — Datadog, New Relic 등 — 이 LLM의 비결정적(non-deterministic) 특성을 제대로 다루지 못했기 때문입니다. "같은 입력에 다른 출력"이 나오는 시스템을 어떻게 모니터링할 것인가라는 새로운 도전이 이 도구들의 탄생을 이끌었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "캐싱을 하면 답변이 outdated될 수 있으니 프로덕션에서는 쓰면 안 된다" — 캐싱의 핵심은 **적절한 TTL(Time To Live) 설정**입니다. 임베딩 캐시는 모델이 바뀌지 않는 한 영구 캐시가 가능하고, LLM 응답 캐시는 15분~1시간 정도의 TTL을 설정하면 최신성과 성능 사이의 균형을 잡을 수 있습니다. 시맨틱 캐시는 유사도 임계값(`score_threshold`)을 높게 설정하여 오히트(false hit)를 방지하세요.

> 💡 **알고 계셨나요?**: LangSmith의 프로덕션 트레이스 분석에 따르면(2025 Q4, 150개 기업), RAG 파이프라인에서 **검색 단계가 전체 지연 시간의 15~30%**를 차지하는 반면, **LLM 생성 단계가 60~75%**를 차지합니다. 즉, 응답 속도를 개선하려면 검색 최적화보다 **LLM 호출 최적화(스트리밍, 캐싱, 작은 모델 사용)**가 더 효과적인 경우가 많습니다. 물론 검색 품질은 답변 정확도에 직결되므로, [9.4 고급 RAG 패턴](ch09/session_9_4.md)에서 배운 재랭킹이나 HyDE도 함께 고려하세요.

> 🔥 **실무 팁**: 프로덕션 인덱싱 파이프라인에서 가장 흔한 장애 원인은 **소스 ID 누락**입니다. `index()` 함수에 `source_id_key="source"`를 지정했는데, 문서의 `metadata`에 `"source"` 키가 없으면 에러가 발생합니다. 커스텀 로더를 사용한다면, 반드시 모든 문서에 소스 식별자를 metadata로 추가하세요:
> ```python
> doc.metadata["source"] = f"notion://{page_id}"  # 소스 추적 가능한 고유 ID
> ```

> 🔥 **실무 팁**: 프로덕션 배포 전 반드시 **임베딩 모델 버전을 고정**하세요. `text-embedding-3-small`에서 `text-embedding-3-large`로 바꾸면 기존 벡터와 새 벡터의 차원과 공간이 달라져 검색 품질이 급격히 떨어집니다. 모델을 변경할 때는 전체 재인덱싱이 필요합니다. `CacheBackedEmbeddings`의 `namespace`를 모델명으로 설정하면, 모델 변경 시 캐시 충돌을 자동으로 방지할 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **RecordManager** | 문서의 해시, 기록 시간, 소스 ID를 추적하여 중복 인덱싱을 방지하는 관리자 |
| **Indexing API (`index()`)** | RecordManager와 벡터 스토어를 연동하여 변경된 문서만 처리하는 함수 |
| **Cleanup 모드** | `None`(중복 제거만), `incremental`(소스 단위 즉시 정리), `full`(전체 동기화), `scoped_full`(배치 단위 전체 동기화) |
| **프로덕션 임베딩 캐시** | Redis 기반 분산 캐시로 다중 서버 간 임베딩 공유, 네임스페이스로 모델·테넌트·환경 격리 |
| **캐시 워밍** | 모델 마이그레이션 시 새 캐시를 사전 구축하여 무중단 전환을 보장하는 패턴 |
| **LangSmith 트레이싱** | 환경 변수 설정만으로 모든 LangChain 호출을 자동 추적하는 관찰 가능성 플랫폼 |
| **커스텀 콜백** | `BaseCallbackHandler`를 상속하여 검색 지연, 토큰 사용량 등 커스텀 메트릭 수집 |
| **다계층 캐싱** | 임베딩 캐시 + LLM 응답 캐시 + 시맨틱 캐시를 조합하여 비용과 지연 시간 절감 |
| **인덱싱/쿼리 분리** | 인덱싱(배치, 비동기)과 쿼리(실시간, 저지연)를 독립적으로 확장할 수 있는 아키텍처 |

## 다음 섹션 미리보기

축하합니다! Chapter 9를 마쳤습니다. 기본 RAG 체인 구축부터 프롬프트 최적화, 대화형 RAG, 고급 검색 패턴, 평가, 그리고 프로덕션 아키텍처까지 — RAG의 전체 여정을 완주했습니다.

다음 [Chapter 10: 메모리와 대화 관리](ch10/session_10_1.md)에서는 RAG를 넘어 LLM 애플리케이션의 또 다른 핵심 축인 **대화 메모리**를 깊이 다룹니다. [9.3 대화형 RAG](ch09/session_9_3.md)에서 `RunnableWithMessageHistory`를 사용해 본 경험이 큰 도움이 될 거예요. 단순한 대화 기록 저장을 넘어, 장기 메모리, 요약 메모리, 엔티티 메모리 등 다양한 메모리 전략을 배우게 됩니다.

## 참고 자료

- [How to use the LangChain indexing API](https://python.langchain.com/docs/how_to/indexing/) — RecordManager와 index() 함수의 공식 가이드, cleanup 모드별 동작을 코드로 확인할 수 있습니다
- [LangSmith Observability Quickstart](https://docs.langchain.com/langsmith/observability-quickstart) — LangSmith 트레이싱 설정과 프로덕션 모니터링 가이드
- [CacheBackedEmbeddings API Reference](https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.cache.CacheBackedEmbeddings.html) — 임베딩 캐싱의 공식 API 문서와 사용 예제
- [Syncing Data Sources to Vector Stores](https://blog.langchain.com/syncing-data-sources-to-vector-stores/) — LangChain 팀이 인덱싱 API를 설계한 배경과 동기를 설명한 블로그
- [Production RAG Architecture: Scaling Considerations](https://apxml.com/courses/optimizing-rag-for-production/chapter-1-production-rag-foundations/production-rag-architecture-scaling) — 프로덕션 RAG의 확장성 패턴(캐싱, 비동기, 수평 확장)을 체계적으로 정리한 가이드

---
### 🔗 Related Sessions
- [embedding](../07-임베딩과-벡터-스토어/01-텍스트-임베딩-이해.md) (prerequisite)
- [rag_pipeline](../09-ragretrieval-augmented-generation-구축/01-기본-rag-체인-구축.md) (prerequisite)
- [format_docs](../09-ragretrieval-augmented-generation-구축/01-기본-rag-체인-구축.md) (prerequisite)
- [retrieval_chain](../09-ragretrieval-augmented-generation-구축/01-기본-rag-체인-구축.md) (prerequisite)
- [langsmith](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
