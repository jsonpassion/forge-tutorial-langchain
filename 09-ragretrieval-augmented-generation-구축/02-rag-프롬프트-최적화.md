# RAG 프롬프트 최적화

> RAG 시스템의 두뇌를 설계합니다 — 시스템 프롬프트, 컨텍스트 포맷팅, 소스 인용, 답변 불가 처리까지

## 개요

이 섹션에서는 RAG 파이프라인의 "생성(Generation)" 단계에 집중합니다. 앞서 [9.1: 기본 RAG 체인 구축](ch09/session_01.md)에서 검색된 문서를 LLM에 전달하는 기본 구조를 만들었다면, 이번에는 **어떻게 전달하느냐**에 따라 답변 품질이 극적으로 달라지는 것을 체험합니다.

**선수 지식**: Session 9.1의 기본 RAG 체인(RunnableParallel, format_docs), Chapter 4의 출력 파서와 구조화된 출력, Chapter 3의 프롬프트 템플릿
**학습 목표**:
- RAG 전용 시스템 프롬프트를 설계할 수 있다
- 검색된 문서를 효과적으로 포맷팅하는 전략을 구현할 수 있다
- LLM 응답에 소스 인용(Citation)을 자동으로 포함시킬 수 있다
- 컨텍스트에 답이 없을 때 안전하게 처리하는 "답변 불가" 로직을 구현할 수 있다

## 왜 알아야 할까?

여러분이 도서관에서 리포트를 쓴다고 상상해 보세요. 사서가 관련 책 5권을 가져다 줬습니다. 같은 책을 받더라도, **"이 책들의 내용만으로 답하세요"**라고 지시받은 사람과 **"자유롭게 답하세요"**라고 지시받은 사람의 리포트는 완전히 다를 겁니다.

RAG에서도 마찬가지입니다. 검색 품질이 아무리 좋아도, 프롬프트가 엉성하면 LLM은 컨텍스트를 무시하고 자기 지식으로 답하거나, 출처를 날조하거나, 틀린 정보를 자신 있게 말합니다. 실제로 LangSmith의 프로덕션 트레이스 분석에 따르면, **RAG 프롬프트 최적화만으로 답변의 사실 정확도(Faithfulness)가 40~70% 향상**될 수 있다고 합니다.

프로덕션 RAG 시스템에서 "환각(Hallucination)을 얼마나 줄이느냐"는 곧 시스템의 신뢰도이고, 그 신뢰도는 프롬프트 설계에서 시작됩니다.

## 핵심 개념

### 개념 1: RAG 시스템 프롬프트 설계

> 💡 **비유**: 시스템 프롬프트는 신입사원의 "업무 매뉴얼"과 같습니다. 매뉴얼에 "모르면 모른다고 말해라", "반드시 근거를 밝혀라"라고 써 있으면, 그 직원은 추측 대신 정확한 답변을 내놓겠죠. LLM에게 주는 시스템 프롬프트가 바로 이 매뉴얼입니다.

RAG 시스템 프롬프트에는 일반 챗봇과 다른 **4가지 핵심 요소**가 필요합니다:

1. **역할 정의**: LLM이 무엇을 하는 존재인지 명확히
2. **컨텍스트 사용 규칙**: 제공된 문서만 사용할 것
3. **답변 불가 처리**: 모를 때의 행동 지침
4. **출력 형식 지정**: 답변의 구조와 톤

LangChain Hub에 공유된 대표적인 RAG 프롬프트를 살펴보겠습니다. `rlm/rag-prompt`는 가장 널리 쓰이는 기본 템플릿입니다:

```python
from langchain_core.prompts import ChatPromptTemplate

# 기본 RAG 프롬프트 (rlm/rag-prompt 스타일)
basic_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise."""),
    ("human", """Question: {question}

Context: {context}"""),
])
```

하지만 프로덕션에서는 더 정교한 프롬프트가 필요합니다. 한국어 RAG 시스템에 최적화된 프롬프트를 설계해 보겠습니다:

```python
from langchain_core.prompts import ChatPromptTemplate

# 프로덕션 수준 한국어 RAG 프롬프트
production_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 제공된 문서를 기반으로 질문에 답변하는 전문 AI 어시스턴트입니다.

## 답변 규칙
1. 반드시 아래 제공된 컨텍스트의 정보만을 사용하여 답변하세요.
2. 컨텍스트에 답변할 수 있는 정보가 없으면, "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
3. 답변 시 참고한 문서의 출처를 [출처: 파일명] 형태로 표기하세요.
4. 추측이나 외부 지식을 사용하지 마세요.

## 답변 형식
- 핵심 답변을 먼저 제시하세요.
- 필요시 부연 설명을 추가하세요.
- 답변은 간결하되 정확해야 합니다."""),
    ("human", """질문: {question}

참고 문서:
{context}"""),
])
```

> 🔥 **실무 팁**: 시스템 프롬프트에서 "하지 마세요" 형태의 금지 규칙은 LLM이 종종 무시합니다. **"반드시 ~하세요"** 형태의 긍정적 지시가 더 효과적입니다. "외부 지식을 사용하지 마세요" 대신 "오직 제공된 컨텍스트만 사용하세요"가 낫습니다.

### 개념 2: 컨텍스트 포맷팅 전략

> 💡 **비유**: 같은 재료로 요리하더라도, 재료를 어떻게 손질하느냐에 따라 맛이 달라지죠. 당근을 통째로 넣는 것과 잘게 다져 넣는 것이 다르듯, 검색된 문서를 LLM에 **어떤 형식으로 전달하느냐**가 답변 품질을 좌우합니다.

Session 9.1에서 사용한 기본 `format_docs` 함수를 기억하시나요? 단순히 `page_content`를 이어 붙이는 방식이었습니다. 이제 이것을 더 정교하게 만들어 보겠습니다.

**전략 1: 메타데이터 포함 포맷팅**

```python
from langchain_core.documents import Document

def format_docs_with_metadata(docs: list[Document]) -> str:
    """문서를 메타데이터와 함께 구조화된 형식으로 포맷팅합니다."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page", "")
        # 출처 정보를 포함한 포맷
        header = f"[문서 {i}] 출처: {source}"
        if page:
            header += f", 페이지: {page}"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# 출력 예시:
# [문서 1] 출처: manual.pdf, 페이지: 42
# LangChain은 LLM 기반 애플리케이션 개발을 위한...
#
# ---
#
# [문서 2] 출처: guide.pdf, 페이지: 15
# LCEL은 LangChain의 선언적 체인 구성...
```

**전략 2: XML 태그 기반 포맷팅**

모델이 구조를 더 잘 파악할 수 있도록 XML 태그를 활용하는 방법도 있습니다. 특히 Claude 계열 모델에서 효과적입니다:

```python
def format_docs_with_xml(docs: list[Document]) -> str:
    """XML 태그로 문서를 구조화하여 LLM이 경계를 명확히 인식하게 합니다."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(
            f"<document index=\"{i}\" source=\"{source}\">\n"
            f"{doc.page_content}\n"
            f"</document>"
        )
    return "\n\n".join(formatted)

# 출력 예시:
# <document index="1" source="manual.pdf">
# LangChain은 LLM 기반 애플리케이션 개발을 위한...
# </document>
#
# <document index="2" source="guide.pdf">
# LCEL은 LangChain의 선언적 체인 구성...
# </document>
```

**전략 3: `document_prompt`를 활용한 LangChain 내장 포맷팅**

`create_stuff_documents_chain`의 `document_prompt` 파라미터를 활용하면 별도 함수 없이도 포맷팅이 가능합니다:

```python
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 각 문서의 포맷을 지정하는 프롬프트
doc_prompt = PromptTemplate.from_template(
    "[출처: {source}]\n{page_content}"
)

# 메인 QA 프롬프트
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "제공된 문서를 기반으로 질문에 답변하세요.\n\n{context}"),
    ("human", "{input}"),
])

# document_prompt로 포맷 지정
chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
    document_prompt=doc_prompt,      # 각 문서의 포맷
    document_separator="\n---\n",     # 문서 간 구분자
)
```

### 개념 3: 소스 인용(Citation) 구현

> 💡 **비유**: 학술 논문을 쓸 때 "~에 따르면[1]"처럼 출처를 밝히는 것과 같습니다. RAG 시스템도 "어느 문서에서 이 정보를 가져왔는지" 명시해야 사용자가 답변을 검증할 수 있습니다.

소스 인용을 구현하는 3가지 접근법이 있습니다:

**방법 1: 프롬프트 기반 인용 (가장 간단)**

프롬프트에서 인용 형식을 지시하는 방법입니다:

```python
citation_prompt = ChatPromptTemplate.from_messages([
    ("system", """제공된 문서를 기반으로 질문에 답변하세요.

답변 시 반드시 참고한 문서 번호를 인라인으로 표기하세요.
예시: "LangChain은 LLM 애플리케이션 프레임워크입니다[1][3]."

답변 마지막에 참고한 문서 목록을 정리하세요.
예시:
---
참고 문서:
[1] manual.pdf, p.42
[3] guide.pdf, p.15"""),
    ("human", """질문: {question}

참고 문서:
{context}"""),
])
```

**방법 2: 구조화된 출력(Structured Output)을 활용한 인용**

`with_structured_output`을 사용하면 인용 정보를 프로그래밍적으로 처리할 수 있습니다:

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Citation(BaseModel):
    """답변에서 참조한 문서의 인용 정보"""
    source: str = Field(description="문서의 출처(파일명 또는 URL)")
    page: str = Field(default="", description="참조한 페이지 번호")
    quote: str = Field(description="답변 근거가 된 원문 인용구")

class AnswerWithCitations(BaseModel):
    """소스 인용이 포함된 RAG 답변"""
    answer: str = Field(description="질문에 대한 답변")
    citations: list[Citation] = Field(description="답변에 사용된 출처 목록")
    confidence: str = Field(
        description="답변 신뢰도: high, medium, low"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 구조화된 출력을 반환하는 LLM
structured_llm = llm.with_structured_output(AnswerWithCitations)

# 결과 예시:
# AnswerWithCitations(
#     answer="LangChain은 LLM 기반 애플리케이션 개발 프레임워크입니다.",
#     citations=[
#         Citation(source="manual.pdf", page="42", quote="LangChain is a framework for..."),
#     ],
#     confidence="high"
# )
```

**방법 3: LCEL을 활용한 완전한 인용 체인**

이제 이 모든 것을 하나의 LCEL 체인으로 결합합니다:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class CitedAnswer(BaseModel):
    """인용이 포함된 답변"""
    answer: str = Field(description="질문에 대한 답변")
    sources: list[str] = Field(description="참조한 문서 출처 목록")

def format_docs_numbered(docs):
    """문서에 번호를 매겨 포맷팅합니다."""
    result = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        result.append(
            f"[{i}] (출처: {source})\n{doc.page_content}"
        )
    return "\n\n".join(result)

prompt = ChatPromptTemplate.from_messages([
    ("system", """제공된 문서를 기반으로 질문에 답변하세요.
답변에 사용한 문서의 출처를 sources 필드에 포함하세요."""),
    ("human", "질문: {question}\n\n참고 문서:\n{context}"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(CitedAnswer)

# LCEL 체인 구성
citation_chain = (
    RunnableParallel(
        context=retriever | format_docs_numbered,
        question=RunnablePassthrough(),
    )
    | prompt
    | structured_llm
)

# 실행
result = citation_chain.invoke("LangChain의 핵심 개념은 무엇인가요?")
print(f"답변: {result.answer}")
print(f"출처: {result.sources}")
```

### 개념 4: 답변 불가(Fallback) 처리

> 💡 **비유**: 좋은 의사는 확실하지 않은 진단을 내리지 않습니다. "더 검사가 필요합니다"라고 솔직히 말하죠. RAG 시스템도 마찬가지입니다. 컨텍스트에 답이 없을 때 **모른다고 정직하게 말하는 것**이 잘못된 답변보다 100배 낫습니다.

답변 불가 처리에는 두 가지 수준이 있습니다:

**수준 1: 프롬프트 수준 처리**

```python
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 제공된 문서 기반으로 답변하는 AI 어시스턴트입니다.

## 핵심 규칙
- 컨텍스트에 답이 있으면: 정확히 답변하고 출처를 밝히세요.
- 컨텍스트에 부분적 정보만 있으면: 알 수 있는 부분만 답변하고, 
  부족한 부분을 명시하세요.
- 컨텍스트에 답이 전혀 없으면: 다음과 같이 응답하세요:
  "제공된 문서에서 해당 질문에 대한 답변을 찾을 수 없습니다.
   다음을 시도해 보세요:
   - 질문을 더 구체적으로 바꿔보세요
   - 관련 키워드를 포함해 다시 질문해 보세요"

절대로 컨텍스트에 없는 내용을 지어내지 마세요."""),
    ("human", "질문: {question}\n\n참고 문서:\n{context}"),
])
```

**수준 2: 구조화된 출력으로 프로그래밍적 처리**

```python
from pydantic import BaseModel, Field
from enum import Enum

class AnswerStatus(str, Enum):
    ANSWERED = "answered"          # 답변 완료
    PARTIAL = "partial"            # 부분 답변
    NOT_FOUND = "not_found"        # 답변 불가

class RAGResponse(BaseModel):
    """RAG 시스템의 구조화된 응답"""
    status: AnswerStatus = Field(
        description="답변 상태: answered, partial, not_found"
    )
    answer: str = Field(description="질문에 대한 답변")
    sources: list[str] = Field(
        default_factory=list,
        description="참조한 문서 출처"
    )
    missing_info: str = Field(
        default="",
        description="답변에 부족한 정보 설명 (partial/not_found인 경우)"
    )

structured_llm = llm.with_structured_output(RAGResponse)

# 응답 상태에 따른 후처리
def handle_response(response: RAGResponse) -> str:
    """답변 상태에 따라 적절한 응답을 반환합니다."""
    if response.status == AnswerStatus.ANSWERED:
        sources_text = ", ".join(response.sources)
        return f"{response.answer}\n\n📚 출처: {sources_text}"
    
    elif response.status == AnswerStatus.PARTIAL:
        return (
            f"{response.answer}\n\n"
            f"⚠️ 참고: {response.missing_info}"
        )
    
    else:  # NOT_FOUND
        return (
            "제공된 문서에서 해당 정보를 찾을 수 없습니다.\n"
            f"💡 도움말: {response.missing_info}"
        )
```

## 실습: 직접 해보기

이제 위에서 배운 모든 개념을 하나의 프로덕션 수준 RAG 체인으로 결합해 보겠습니다. 이 실습에서는 시스템 프롬프트 설계, 메타데이터 포맷팅, 구조화된 인용, 답변 불가 처리를 모두 적용합니다.

```python
"""
RAG 프롬프트 최적화 실습 — 프로덕션 수준의 인용 기반 RAG 체인
"""

import os
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

# ── 1. 설정 ──────────────────────────────────────────────
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# ── 2. 샘플 문서 준비 ──────────────────────────────────────
# 실제 프로젝트에서는 문서 로더 + 텍스트 분할기를 사용합니다
docs = [
    Document(
        page_content="LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션을 쉽게 만들 수 있도록 도와주는 "
                     "오픈소스 프레임워크입니다. Python과 JavaScript를 지원하며, "
                     "2022년 10월 Harrison Chase가 처음 공개했습니다.",
        metadata={"source": "langchain_intro.pdf", "page": "1"}
    ),
    Document(
        page_content="LCEL(LangChain Expression Language)은 체인을 선언적으로 구성하는 방법입니다. "
                     "파이프 연산자(|)로 컴포넌트를 연결하며, 스트리밍, 배치 처리, "
                     "비동기 실행을 기본 지원합니다.",
        metadata={"source": "lcel_guide.pdf", "page": "5"}
    ),
    Document(
        page_content="RAG(Retrieval-Augmented Generation)는 외부 지식 소스에서 관련 정보를 검색하여 "
                     "LLM의 응답 품질을 높이는 기법입니다. 2020년 Meta AI 연구팀의 "
                     "Patrick Lewis 등이 발표한 논문에서 처음 제안되었습니다.",
        metadata={"source": "rag_paper.pdf", "page": "1"}
    ),
    Document(
        page_content="벡터 스토어는 텍스트 임베딩을 저장하고 유사도 검색을 수행하는 데이터베이스입니다. "
                     "FAISS, Chroma, Pinecone, Weaviate 등이 대표적이며, "
                     "각각 인메모리, 로컬 영속, 클라우드 호스팅 등 다른 특성을 가집니다.",
        metadata={"source": "vectorstore_guide.pdf", "page": "3"}
    ),
]

# ── 3. 벡터 스토어 생성 ──────────────────────────────────────
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 4. 컨텍스트 포맷팅 함수 ──────────────────────────────────
def format_docs_with_metadata(docs: list[Document]) -> str:
    """메타데이터를 포함한 구조화된 포맷으로 문서를 변환합니다."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page", "")
        header = f"[문서 {i}] 출처: {source}"
        if page:
            header += f" (p.{page})"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# ── 5. 구조화된 출력 스키마 ──────────────────────────────────
class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"

class SourceCitation(BaseModel):
    """개별 인용 정보"""
    document_index: int = Field(description="참조한 문서 번호 (1부터 시작)")
    source_file: str = Field(description="출처 파일명")
    relevant_quote: str = Field(description="근거가 된 핵심 인용구 (1~2문장)")

class RAGAnswer(BaseModel):
    """인용과 상태 정보가 포함된 RAG 응답"""
    status: AnswerStatus = Field(
        description="답변 상태 — answered: 완전한 답변, partial: 부분 답변, not_found: 답변 불가"
    )
    answer: str = Field(description="질문에 대한 답변")
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description="답변에 사용된 소스 인용 목록"
    )
    follow_up_suggestion: str = Field(
        default="",
        description="추가 질문 제안 (partial 또는 not_found인 경우)"
    )

# ── 6. 프로덕션 RAG 프롬프트 ────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 제공된 문서를 기반으로 정확하게 답변하는 전문 AI 어시스턴트입니다.

## 답변 규칙
1. **반드시** 아래 제공된 참고 문서의 정보만 사용하여 답변하세요.
2. 각 문서는 [문서 N] 형태로 번호가 매겨져 있습니다. 답변에 사용한 문서의 번호와 출처를 citations에 포함하세요.
3. 답변의 근거가 되는 핵심 문장을 relevant_quote에 인용하세요.
4. 컨텍스트에 답이 충분하면 status를 "answered"로, 부분적이면 "partial"로, 없으면 "not_found"로 설정하세요.
5. "not_found" 또는 "partial"인 경우, 사용자가 다음에 시도할 수 있는 질문을 follow_up_suggestion에 제안하세요.

## 주의사항
- 컨텍스트에 없는 내용을 추측하거나 지어내지 마세요.
- 모든 사실적 주장에는 반드시 인용을 붙이세요."""),
    ("human", """질문: {question}

참고 문서:
{context}"""),
])

# ── 7. 체인 구성 ────────────────────────────────────────────
structured_llm = llm.with_structured_output(RAGAnswer)

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs_with_metadata,  # 검색 + 포맷팅
        question=RunnablePassthrough(),                  # 질문 통과
    )
    | rag_prompt           # 프롬프트 적용
    | structured_llm       # 구조화된 출력 생성
)

# ── 8. 응답 포맷터 ──────────────────────────────────────────
def format_response(response: RAGAnswer) -> str:
    """RAGAnswer를 사용자 친화적인 텍스트로 변환합니다."""
    lines = []
    
    # 상태 아이콘
    status_icon = {
        AnswerStatus.ANSWERED: "✅",
        AnswerStatus.PARTIAL: "⚠️",
        AnswerStatus.NOT_FOUND: "❌",
    }
    lines.append(f"{status_icon[response.status]} [{response.status.value}]\n")
    lines.append(response.answer)
    
    # 인용 정보
    if response.citations:
        lines.append("\n📚 출처:")
        for cite in response.citations:
            lines.append(
                f"  [{cite.document_index}] {cite.source_file}"
                f" — \"{cite.relevant_quote}\""
            )
    
    # 후속 질문 제안
    if response.follow_up_suggestion:
        lines.append(f"\n💡 추천 질문: {response.follow_up_suggestion}")
    
    return "\n".join(lines)

# ── 9. 테스트 실행 ──────────────────────────────────────────
if __name__ == "__main__":
    # 테스트 1: 답변 가능한 질문
    print("=" * 60)
    print("테스트 1: 일반 질문")
    print("=" * 60)
    result = rag_chain.invoke("LangChain은 무엇이고 누가 만들었나요?")
    print(format_response(result))
    
    # 테스트 2: 부분 답변 가능한 질문
    print("\n" + "=" * 60)
    print("테스트 2: 부분 답변 질문")
    print("=" * 60)
    result = rag_chain.invoke("LangChain과 LlamaIndex의 차이점은 무엇인가요?")
    print(format_response(result))
    
    # 테스트 3: 답변 불가능한 질문
    print("\n" + "=" * 60)
    print("테스트 3: 답변 불가 질문")
    print("=" * 60)
    result = rag_chain.invoke("파이썬의 GIL은 무엇인가요?")
    print(format_response(result))
```

예상 출력:

```
============================================================
테스트 1: 일반 질문
============================================================
✅ [answered]

LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션을 쉽게 만들 수 있도록 
도와주는 오픈소스 프레임워크입니다. Python과 JavaScript를 지원하며, 
2022년 10월 Harrison Chase가 처음 공개했습니다.

📚 출처:
  [1] langchain_intro.pdf — "LangChain은 대규모 언어 모델(LLM)을 활용한 
  애플리케이션을 쉽게 만들 수 있도록 도와주는 오픈소스 프레임워크입니다."

============================================================
테스트 2: 부분 답변 질문
============================================================
⚠️ [partial]

LangChain은 LLM 기반 애플리케이션 개발을 위한 오픈소스 프레임워크입니다. 
그러나 제공된 문서에는 LlamaIndex에 대한 정보가 포함되어 있지 않아 
직접적인 비교가 어렵습니다.

📚 출처:
  [1] langchain_intro.pdf — "LangChain은 대규모 언어 모델(LLM)을 활용한..."

💡 추천 질문: "LangChain의 주요 기능과 특징은 무엇인가요?"

============================================================
테스트 3: 답변 불가 질문
============================================================
❌ [not_found]

제공된 문서에서 파이썬 GIL에 대한 정보를 찾을 수 없습니다.

💡 추천 질문: "LangChain이나 LCEL에 대해 질문해 보세요."
```

## 더 깊이 알아보기

### RAG의 탄생과 프롬프트 전략의 진화

RAG라는 개념은 2020년 Meta AI(당시 Facebook AI Research)의 Patrick Lewis, Ethan Perez 등이 발표한 논문 *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*에서 처음 제안되었습니다. 흥미로운 점은, 초기 RAG 논문에서는 프롬프트 설계보다 **검색 모듈의 학습**에 초점을 맞추었다는 것입니다. 검색기와 생성기를 end-to-end로 학습시키는 것이 핵심이었죠.

그런데 GPT-3.5와 GPT-4가 등장하면서 상황이 바뀌었습니다. 모델이 충분히 강력해지자, 학습 없이도 **프롬프트만 잘 설계하면** 뛰어난 RAG 성능을 얻을 수 있게 된 겁니다. 이것이 바로 "프롬프트 엔지니어링 시대의 RAG"입니다.

LangChain Hub의 `rlm/rag-prompt`는 이 변화를 상징합니다. LangChain 공동 창업자 Harrison Chase의 팀원인 Lance Martin이 만든 이 프롬프트는, 단 4줄의 지시문으로 RAG의 핵심 원칙을 담았습니다: *"제공된 컨텍스트로 답하라, 모르면 모른다고 하라, 3문장 이내로 간결하게."* 이 심플한 프롬프트가 LangChain Hub에서 가장 많이 사용된 프롬프트가 되었다는 것은, 좋은 프롬프트가 복잡할 필요 없다는 교훈을 줍니다.

최근에는 **"Context Engineering"**이라는 용어가 등장했습니다. LangChain 블로그에서 사용한 이 표현은, 단순히 프롬프트를 잘 쓰는 것을 넘어서 **컨텍스트 윈도우에 적절한 정보를 적절한 형식으로 채우는 기술**을 의미합니다. 선택(select), 압축(compress), 격리(isolate), 기록(write) — 이 4가지 전략이 차세대 RAG 프롬프트 엔지니어링의 핵심입니다.

### 인용(Citation)의 진화

소스 인용은 RAG의 "신뢰성 레이어"입니다. 초기에는 단순히 검색된 문서 목록을 응답 끝에 붙이는 수준이었지만, 현재는 인라인 인용(in-text citation), 하이라이트 기반 인용, 그리고 LLM의 tool-calling을 활용한 구조화된 인용으로 발전했습니다. 2024~2025년의 연구에 따르면, 적절한 인용 구현은 환각(hallucination)을 45~72% 감소시킵니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "프롬프트에 '모르면 모른다고 해'라고 쓰면 LLM이 항상 따를 것이다." — 실제로는 LLM이 이 지시를 종종 무시합니다. 특히 사용자 질문이 LLM의 기본 지식에 있는 내용이라면, 컨텍스트에 없어도 자기 지식으로 답하려 합니다. 이를 방지하려면 **구조화된 출력**으로 `status` 필드를 강제하고, 후처리에서 `not_found` 상태를 체크하는 이중 방어가 필요합니다.

> 💡 **알고 계셨나요?**: LangChain Hub의 `rlm/rag-prompt`는 영어로 작성되어 있지만, 한국어 RAG 시스템에 그대로 사용해도 GPT-4 계열에서는 한국어 질문에 한국어로 답변합니다. 하지만 프롬프트 자체를 한국어로 작성하면 **답변의 한국어 품질이 약 15% 향상**된다는 실험 결과가 있습니다. 한국어 서비스라면 시스템 프롬프트도 한국어로 작성하세요.

> 🔥 **실무 팁**: 검색된 문서를 LLM에 넣을 때, **문서 순서가 답변에 영향을 줍니다**. LLM은 컨텍스트의 처음과 끝에 있는 정보를 더 잘 활용하는 경향이 있습니다(이를 "Lost in the Middle" 현상이라 합니다). 가장 관련성 높은 문서를 **맨 처음**에 배치하세요. LangChain의 검색기는 기본적으로 유사도 순으로 정렬하므로, 별도 재정렬 없이도 어느 정도 이 원칙이 지켜집니다.

> 🔥 **실무 팁**: `document_separator`로 문서 사이에 `"\n---\n"` 같은 구분자를 넣으면, LLM이 문서 경계를 더 잘 인식합니다. 구분자 없이 문서를 연결하면, LLM이 서로 다른 문서의 내용을 혼동하여 잘못된 인용을 생성할 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| RAG 시스템 프롬프트 | 역할 정의, 컨텍스트 사용 규칙, 답변 불가 처리, 출력 형식을 포함하는 RAG 전용 프롬프트 |
| 컨텍스트 포맷팅 | 메타데이터 포함, XML 태그, `document_prompt` 등으로 검색 문서를 구조화하여 LLM에 전달 |
| 소스 인용 (Citation) | 프롬프트 지시, `with_structured_output`, LCEL 체인 등으로 답변에 출처를 명시 |
| 답변 불가 처리 (Fallback) | `AnswerStatus` enum으로 answered/partial/not_found를 구분하고, 후처리로 안전하게 대응 |
| `with_structured_output` | Pydantic 모델을 사용해 LLM 응답을 프로그래밍적으로 처리 가능한 구조로 강제 |
| `document_prompt` | `create_stuff_documents_chain`에서 각 문서의 포맷을 지정하는 파라미터 |
| Lost in the Middle | LLM이 컨텍스트의 중간 부분을 잘 활용하지 못하는 현상 — 중요 문서를 앞에 배치 |

## 다음 섹션 미리보기

이번 섹션에서 프롬프트를 최적화해 단일 턴 RAG의 품질을 높였습니다. 하지만 실제 사용자는 한 번의 질문으로 끝나지 않죠. "방금 말한 그것 더 자세히 설명해줘"처럼 **이전 대화를 참조하는 후속 질문**을 합니다. 다음 섹션에서는 **대화형 RAG(Conversational RAG)**를 구축합니다 — 대화 히스토리 관리, 질문 재작성(Query Reformulation), 그리고 `create_history_aware_retriever`를 활용하여 멀티턴 대화가 자연스럽게 이어지는 RAG 시스템을 만들어 보겠습니다.

## 참고 자료

- [LangChain 공식 문서 — How to return citations](https://python.langchain.com/docs/how_to/qa_citations/) - RAG에서 소스 인용을 구현하는 3가지 공식 가이드 (프롬프트 기반, 구조화된 출력, 후처리 방식)
- [LangChain 공식 문서 — Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output) - `with_structured_output` 메서드와 Pydantic 모델을 활용한 구조화된 출력 완전 가이드
- [LangChain Hub — rlm/rag-prompt](https://smith.langchain.com/hub/rlm/rag-prompt) - LangChain에서 가장 널리 사용되는 기본 RAG 프롬프트 템플릿
- [LangChain Hub — teddynote/rag-prompt](https://smith.langchain.com/hub/teddynote/rag-prompt) - QA 특화 정제 RAG 프롬프트, 한국어 RAG 시스템에 유용
- [LangChain 블로그 — Context Engineering](https://blog.langchain.com/context-engineering-for-agents/) - 프롬프트 엔지니어링을 넘어선 "컨텍스트 엔지니어링" 개념과 4가지 전략 소개
- [create_stuff_documents_chain API 레퍼런스](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html) - document_prompt, document_separator 등 파라미터 상세 문서

---
### 🔗 Related Sessions
- [lcel](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [chatprompttemplate](../01-langchain-소개와-개발-환경-설정/04-첫-번째-langchain-애플리케이션.md) (prerequisite)
- [with_structured_output](../03-프롬프트-엔지니어링과-템플릿/05-프롬프트-엔지니어링-실전-기법.md) (prerequisite)
- [rag_pipeline](../09-ragretrieval-augmented-generation-구축/01-기본-rag-체인-구축.md) (prerequisite)
- [format_docs](../09-ragretrieval-augmented-generation-구축/01-기본-rag-체인-구축.md) (prerequisite)
- [retrieval_chain](../09-ragretrieval-augmented-generation-구축/01-기본-rag-체인-구축.md) (prerequisite)
- [stuff_documents_chain](../09-ragretrieval-augmented-generation-구축/01-기본-rag-체인-구축.md) (prerequisite)
