# Streamlit UI 구축

> 18.4에서 완성한 ConversationalQA 백엔드를 Streamlit으로 감싸, 채팅·문서 업로드·소스 인용·대화 내보내기를 갖춘 웹 앱을 만듭니다.

## 개요

이 섹션에서는 앞서 구축한 문서 QA 백엔드를 실제 사용자가 조작할 수 있는 웹 인터페이스로 완성합니다. Streamlit의 채팅 전용 위젯(`st.chat_message`, `st.chat_input`)과 `st.session_state` 기반 상태 관리를 결합하여, 백엔드 로직 한 줄 수정 없이 프레젠테이션 레이어만 얹는 과정을 다룹니다.

**선수 지식**:
- [18.1 프로젝트 설계와 아키텍처](ch18/session1.md)의 4-계층 아키텍처 개념
- [18.2 문서 수집과 인덱싱 파이프라인](ch18/session2.md)의 `IngestionPipeline`
- [18.3 검색과 생성 파이프라인](ch18/session3.md)의 `QAPipeline`과 `CitedAnswer`
- [18.4 대화 관리와 메모리](ch18/session4.md)의 `ConversationalQA`와 `SessionManager`

**학습 목표**:
- `st.chat_message`와 `st.chat_input`으로 자연스러운 채팅 인터페이스를 구현할 수 있다
- `st.file_uploader`를 활용해 문서를 업로드하고 실시간으로 인덱싱할 수 있다
- 사이드바에 소스 인용과 메타데이터를 시각적으로 표시할 수 있다
- 대화 내역을 JSON/CSV로 내보내는 기능을 구현할 수 있다

---

## 왜 알아야 할까?

아무리 정교한 RAG 파이프라인을 만들어도, 사용자가 터미널에서 Python 스크립트를 실행해야 한다면 실무에서 쓰이기 어렵습니다. "좋은 백엔드 + 나쁜 UI = 아무도 안 쓰는 도구"라는 말이 있을 정도인데요. Streamlit은 **Python만으로** 인터랙티브 웹 앱을 만들 수 있어, 프론트엔드 경험이 없는 ML/데이터 엔지니어도 몇 시간 안에 프로토타입을 배포할 수 있습니다.

실제로 기업 내부 도구, PoC 데모, 해커톤 프로젝트에서 Streamlit + LangChain 조합은 가장 빠른 경로입니다. 이번 섹션을 마치면, 18.1~18.4에서 쌓아 올린 모든 백엔드 컴포넌트가 하나의 완성된 웹 앱으로 통합됩니다.

---

## 핵심 개념

### 개념 1: Streamlit의 실행 모델 — "매번 처음부터 다시 그리기"

> 💡 **비유**: Streamlit의 실행 모델은 **칠판 수업**과 비슷합니다. 학생이 질문(사용자 인터랙션)할 때마다 선생님이 칠판을 깨끗이 지우고 처음부터 다시 그리되, **수첩(`st.session_state`)에 적어둔 메모**는 유지합니다. 칠판은 매번 새로 그리지만, 수첩 덕분에 이전 대화 맥락을 잃지 않는 거죠.

Streamlit은 사용자가 버튼을 누르거나 입력을 보낼 때마다 **스크립트 전체를 위에서 아래로 다시 실행**합니다. 일반적인 웹 프레임워크(Flask, Django)와 달리 라우팅이나 콜백 함수를 등록하는 방식이 아닙니다. 이 "재실행 모델" 덕분에 코드가 선언적이고 단순하지만, **상태를 보존하려면 반드시 `st.session_state`를 사용**해야 합니다.

```python
import streamlit as st

# 앱이 재실행될 때마다 이 블록이 실행됨
# session_state에 없는 키만 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 히스토리 보존용

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"  # 현재 세션 ID
```

> ⚠️ **흔한 오해**: "Streamlit은 느리다"라고 생각하기 쉽지만, 재실행 시 **변경된 위젯만 다시 렌더링**합니다. `@st.cache_data`나 `@st.cache_resource`를 적절히 사용하면 무거운 연산(모델 로딩, 인덱스 빌드)은 캐싱되어 실질적인 체감 속도는 빠릅니다.

---

### 개념 2: 채팅 인터페이스 — `st.chat_message`와 `st.chat_input`

> 💡 **비유**: `st.chat_message`는 **말풍선**, `st.chat_input`은 **입력창**입니다. 카카오톡을 떠올려 보세요 — 화면 아래에 고정된 입력창에 메시지를 치면 위에 말풍선이 생기는 구조, 그게 Streamlit 채팅 UI의 전부입니다.

Streamlit 1.26부터 도입된 채팅 전용 위젯은 LLM 앱에 최적화되어 있습니다.

```python
import streamlit as st

st.title("📚 지능형 문서 QA")

# 저장된 대화 히스토리를 화면에 다시 그리기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # "user" 또는 "assistant"
        st.markdown(msg["content"])

# 하단에 고정된 채팅 입력 위젯
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    # 사용자 메시지 저장 & 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답 생성 & 표시
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            # 18.4의 ConversationalQA 호출
            result = st.session_state.qa.ask(
                question=prompt,
                session_id=st.session_state.session_id,
            )
        st.markdown(result["answer"])

    # 어시스턴트 메시지 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),  # 소스 인용 정보 함께 저장
    })
```

핵심 패턴은 세 단계입니다:

1. **히스토리 재생**: `for msg in st.session_state.messages` — 재실행 시 이전 대화를 다시 그립니다
2. **사용자 입력 처리**: `st.chat_input` — 왈러스 연산자(`:=`)로 입력 존재 여부를 체크합니다
3. **응답 생성 & 저장**: 백엔드 호출 후 결과를 `st.session_state`에 추가합니다

---

### 개념 3: 문서 업로드와 실시간 인덱싱

> 💡 **비유**: 도서관에 새 책을 기증하는 과정을 떠올려 보세요. 책을 접수 창구(`st.file_uploader`)에 맡기면, 사서가 분류·라벨링(청킹+메타데이터)을 거쳐 서가에 꽂아 놓습니다(벡터 인덱스). 그 다음부터 바로 검색이 가능해지죠.

`st.file_uploader`에 `accept_multiple_files=True`를 설정하면 여러 파일을 한 번에 받을 수 있습니다. 핵심은 **업로드된 파일을 임시 디렉토리에 저장한 뒤 `IngestionPipeline`에 넘기는 것**입니다.

```python
import tempfile
from pathlib import Path
from ingestion import IngestionPipeline  # 18.2에서 구현한 파이프라인

def process_uploaded_files(
    uploaded_files: list,
    pipeline: IngestionPipeline,
) -> dict:
    """업로드된 파일을 임시 저장 후 인덱싱 파이프라인에 전달합니다."""
    stats = {"success": 0, "failed": 0, "filenames": []}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            # 임시 디렉토리에 파일 저장
            tmp_path = Path(tmp_dir) / uploaded_file.name
            tmp_path.write_bytes(uploaded_file.getvalue())
            stats["filenames"].append(uploaded_file.name)

        try:
            # 18.2의 IngestionPipeline으로 문서 로드 → 청킹 → 인덱싱
            result = pipeline.ingest_directory(tmp_dir)
            stats["success"] = result.doc_count
        except Exception as e:
            stats["failed"] = len(uploaded_files)
            stats["error"] = str(e)

    return stats
```

사이드바에 업로더를 배치하면 채팅 영역을 방해하지 않습니다:

```python
with st.sidebar:
    st.header("📁 문서 관리")

    uploaded_files = st.file_uploader(
        "문서를 업로드하세요",
        type=["pdf", "docx", "txt", "html"],  # 지원 포맷
        accept_multiple_files=True,
        help="PDF, DOCX, TXT, HTML 파일을 지원합니다",
    )

    if uploaded_files and st.button("🔄 인덱싱 시작"):
        with st.spinner("문서를 처리하고 있습니다..."):
            stats = process_uploaded_files(
                uploaded_files,
                st.session_state.pipeline,
            )
        if stats["success"] > 0:
            st.success(f"✅ {stats['success']}개 문서 인덱싱 완료!")
        if stats["failed"] > 0:
            st.error(f"❌ {stats['failed']}개 문서 처리 실패")
```

> 🔥 **실무 팁**: `st.file_uploader`는 기본 최대 200MB입니다. 대용량 문서를 다루려면 `.streamlit/config.toml`에서 `server.maxUploadSize`를 조정하세요. 또한, 업로드 직후 바로 인덱싱하지 말고 **별도 버튼을 두는 것**이 좋습니다 — 사용자가 여러 파일을 선택한 뒤 한 번에 처리할 수 있으니까요.

---

### 개념 4: 소스 인용 표시 — 사이드바와 Expander 활용

> 💡 **비유**: 논문을 읽을 때 본문에는 핵심 내용이, 각주에는 출처가 있죠. 우리 앱에서도 채팅 영역은 답변(본문), 사이드바는 출처(각주) 역할을 합니다.

18.3에서 구현한 `CitedAnswer`는 `answer`, `citations` 필드를 갖고 있었습니다. 이 구조화된 출력을 사이드바에 시각적으로 표시합니다.

```python
def display_sources(sources: list[dict]) -> None:
    """사이드바에 소스 인용 정보를 표시합니다."""
    with st.sidebar:
        st.header("📖 참조 소스")

        if not sources:
            st.info("이 답변에 사용된 소스가 없습니다.")
            return

        for i, source in enumerate(sources, 1):
            with st.expander(
                f"📄 소스 {i}: {source.get('title', '제목 없음')}",
                expanded=(i == 1),  # 첫 번째 소스만 펼침
            ):
                # 메타데이터 표시
                st.caption(f"📁 파일: {source.get('file_name', 'N/A')}")
                st.caption(f"📃 페이지: {source.get('page', 'N/A')}")

                # 신뢰도 점수를 프로그레스 바로 시각화
                score = source.get("relevance_score", 0)
                st.progress(
                    min(score, 1.0),
                    text=f"관련도: {score:.1%}",
                )

                # 원문 발췌
                st.markdown("**발췌 내용:**")
                st.markdown(
                    f"> {source.get('content', '내용 없음')[:300]}..."
                )
```

이 함수를 응답 생성 직후에 호출하면, 답변이 나올 때마다 사이드바에 근거 자료가 함께 갱신됩니다.

```python
# 어시스턴트 응답 생성 후
if result.get("sources"):
    display_sources(result["sources"])
```

---

### 개념 5: 대화 내보내기 — `st.download_button`

> 💡 **비유**: 회의가 끝나면 회의록을 PDF로 내보내듯, QA 대화도 내보내기가 필요합니다. 분석·감사·공유 등 다양한 목적이 있거든요.

Streamlit의 `st.download_button`은 데이터를 바로 파일로 내려받을 수 있는 위젯입니다. 대화 내역을 JSON과 CSV 두 가지 형식으로 내보내는 기능을 만들어 보겠습니다.

```python
import json
import csv
import io
from datetime import datetime


def export_as_json(messages: list[dict]) -> str:
    """대화 내역을 JSON 문자열로 변환합니다."""
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": [
            {
                "role": msg["role"],
                "content": msg["content"],
                "sources": msg.get("sources", []),
            }
            for msg in messages
        ],
    }
    return json.dumps(export_data, ensure_ascii=False, indent=2)


def export_as_csv(messages: list[dict]) -> str:
    """대화 내역을 CSV 문자열로 변환합니다."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["순서", "역할", "내용", "소스 수"])  # 헤더

    for i, msg in enumerate(messages, 1):
        writer.writerow([
            i,
            msg["role"],
            msg["content"],
            len(msg.get("sources", [])),
        ])

    return output.getvalue()
```

사이드바에 내보내기 버튼을 배치합니다:

```python
with st.sidebar:
    st.header("💾 대화 내보내기")

    if st.session_state.messages:
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 JSON",
                data=export_as_json(st.session_state.messages),
                file_name=f"qa_export_{datetime.now():%Y%m%d_%H%M}.json",
                mime="application/json",
            )

        with col2:
            st.download_button(
                label="📥 CSV",
                data=export_as_csv(st.session_state.messages),
                file_name=f"qa_export_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
    else:
        st.info("내보낼 대화가 없습니다.")
```

---

### 개념 6: 무거운 리소스 캐싱 — `@st.cache_resource`

> 💡 **비유**: 매번 손님이 올 때마다 피자 반죽부터 만드는 대신, **미리 반죽을 대량으로 만들어 냉장고에 넣어두는** 것이 `@st.cache_resource`입니다. LLM 모델 객체, 벡터 인덱스, DB 연결처럼 **한 번 만들면 계속 재사용하는 무거운 객체**에 적합합니다.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


@st.cache_resource
def load_qa_system():
    """QA 시스템 컴포넌트를 한 번만 초기화합니다.

    Returns:
        초기화된 ConversationalQA 인스턴스
    """
    from conversational_qa import ConversationalQA  # 18.4
    from ingestion import IngestionPipeline          # 18.2

    # LLM과 임베딩 모델 — 앱 전체에서 공유
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 기존 인덱스 로드 또는 빈 인덱스 생성
    try:
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        vectorstore = None

    pipeline = IngestionPipeline(embeddings=embeddings)
    qa = ConversationalQA(llm=llm, vectorstore=vectorstore)

    return qa, pipeline


# 앱 시작 시 한 번만 실행
qa_system, ingestion_pipeline = load_qa_system()

# session_state에 등록하여 다른 함수에서 접근 가능
if "qa" not in st.session_state:
    st.session_state.qa = qa_system
if "pipeline" not in st.session_state:
    st.session_state.pipeline = ingestion_pipeline
```

`@st.cache_resource`와 `@st.cache_data`의 차이를 기억하세요:

| 데코레이터 | 용도 | 예시 |
|-----------|------|------|
| `@st.cache_resource` | 전역 공유 객체 (해싱 불가) | LLM, DB 연결, 벡터 스토어 |
| `@st.cache_data` | 직렬화 가능한 데이터 | DataFrame, JSON, 계산 결과 |

---

## 실습: 직접 해보기

이제 모든 개념을 하나의 완전한 앱으로 통합합니다. 이 코드는 18.1~18.4의 백엔드와 결합하여 실행할 수 있는 완전한 Streamlit 앱입니다.

```python
"""
지능형 문서 QA 시스템 — Streamlit UI
=====================================
실행: streamlit run app.py

필요 패키지:
    pip install streamlit langchain langchain-openai faiss-cpu python-dotenv
"""

import json
import csv
import io
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일에 OPENAI_API_KEY 필요)
load_dotenv()

# ──────────────────────────────────────────────
# 페이지 설정 (반드시 다른 st 호출보다 먼저)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="📚 문서 QA 시스템",
    page_icon="📚",
    layout="wide",              # 넓은 레이아웃으로 사이드바 활용
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# 1. 무거운 리소스 캐싱
# ──────────────────────────────────────────────
@st.cache_resource
def init_qa_system():
    """QA 시스템을 초기화합니다 (앱 전체에서 한 번만 실행)."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.history_aware_retriever import (
        create_history_aware_retriever,
    )
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_community.chat_message_histories import (
        ChatMessageHistory,
    )
    from langchain_core.runnables.history import RunnableWithMessageHistory

    # --- 모델 초기화 ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- 텍스트 분할기 ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # --- 기존 인덱스 로드 시도 ---
    index_path = Path("faiss_index")
    vectorstore = None
    if index_path.exists():
        try:
            vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            vectorstore = None

    # --- 세션 저장소 ---
    session_store: dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    # --- 프롬프트 ---
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "대화 기록과 최신 사용자 질문을 바탕으로, "
         "대화 기록 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요. "
         "질문에 답하지 말고, 필요하면 재구성만 하세요."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 문서 기반 QA 어시스턴트입니다. "
         "아래 검색된 컨텍스트를 활용하여 질문에 답변하세요. "
         "답을 모르면 모른다고 솔직히 말하세요. "
         "답변 끝에 참조한 문서의 출처를 명시하세요.\n\n"
         "컨텍스트:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    return {
        "llm": llm,
        "embeddings": embeddings,
        "text_splitter": text_splitter,
        "vectorstore": vectorstore,
        "get_session_history": get_session_history,
        "contextualize_prompt": contextualize_prompt,
        "qa_prompt": qa_prompt,
    }


def build_chain(system: dict):
    """벡터 스토어가 준비되면 RAG 체인을 구성합니다."""
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.history_aware_retriever import (
        create_history_aware_retriever,
    )
    from langchain_core.runnables.history import RunnableWithMessageHistory

    vs = system["vectorstore"]
    if vs is None:
        return None

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )

    # 히스토리 인식 리트리버
    history_aware_retriever = create_history_aware_retriever(
        system["llm"],
        retriever,
        system["contextualize_prompt"],
    )

    # 문서 결합 체인
    qa_chain = create_stuff_documents_chain(
        system["llm"],
        system["qa_prompt"],
    )

    # 검색 + 생성 통합
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # 메시지 히스토리 래핑
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        system["get_session_history"],
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain


# ──────────────────────────────────────────────
# 2. 유틸리티 함수
# ──────────────────────────────────────────────
def ingest_files(uploaded_files: list, system: dict) -> dict:
    """업로드된 파일을 처리하여 벡터 스토어에 추가합니다."""
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
        UnstructuredHTMLLoader,
    )
    from langchain_community.vectorstores import FAISS

    # 파일 확장자 → 로더 매핑
    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".html": UnstructuredHTMLLoader,
    }

    all_docs = []
    errors = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uf in uploaded_files:
            tmp_path = Path(tmp_dir) / uf.name
            tmp_path.write_bytes(uf.getvalue())

            ext = tmp_path.suffix.lower()
            loader_cls = LOADER_MAP.get(ext)
            if loader_cls is None:
                errors.append(f"{uf.name}: 지원하지 않는 형식 ({ext})")
                continue

            try:
                loader = loader_cls(str(tmp_path))
                docs = loader.load()
                # 메타데이터에 원본 파일명 추가
                for doc in docs:
                    doc.metadata["source_file"] = uf.name
                all_docs.extend(docs)
            except Exception as e:
                errors.append(f"{uf.name}: {e}")

    if not all_docs:
        return {"success": 0, "errors": errors}

    # 텍스트 분할
    splits = system["text_splitter"].split_documents(all_docs)

    # 벡터 스토어 업데이트 또는 생성
    if system["vectorstore"] is None:
        system["vectorstore"] = FAISS.from_documents(
            splits, system["embeddings"]
        )
    else:
        system["vectorstore"].add_documents(splits)

    # 인덱스 영속화
    system["vectorstore"].save_local("faiss_index")

    return {
        "success": len(all_docs),
        "chunks": len(splits),
        "errors": errors,
    }


def export_json(messages: list[dict]) -> str:
    """대화를 JSON으로 내보냅니다."""
    data = {
        "exported_at": datetime.now().isoformat(),
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_csv(messages: list[dict]) -> str:
    """대화를 CSV로 내보냅니다."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["순서", "역할", "내용"])
    for i, m in enumerate(messages, 1):
        writer.writerow([i, m["role"], m["content"]])
    return buf.getvalue()


# ──────────────────────────────────────────────
# 3. 세션 상태 초기화
# ──────────────────────────────────────────────
system = init_qa_system()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now():%Y%m%d_%H%M%S}"
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ──────────────────────────────────────────────
# 4. 사이드바 — 문서 관리 / 세션 / 내보내기
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📁 문서 업로드")
    uploaded = st.file_uploader(
        "분석할 문서를 올려주세요",
        type=["pdf", "docx", "txt", "html"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("🔄 인덱싱 시작", type="primary"):
        with st.spinner("문서를 처리 중입니다..."):
            result = ingest_files(uploaded, system)
        if result["success"] > 0:
            st.success(
                f"✅ {result['success']}개 문서 → "
                f"{result['chunks']}개 청크 인덱싱 완료"
            )
        for err in result.get("errors", []):
            st.error(err)

    st.divider()

    # --- 세션 관리 ---
    st.header("💬 세션 관리")
    new_session = st.button("🆕 새 대화 시작")
    if new_session:
        st.session_state.messages = []
        st.session_state.session_id = (
            f"session_{datetime.now():%Y%m%d_%H%M%S}"
        )
        st.session_state.last_sources = []
        st.rerun()

    st.caption(f"현재 세션: `{st.session_state.session_id}`")

    st.divider()

    # --- 소스 인용 표시 ---
    st.header("📖 참조 소스")
    if st.session_state.last_sources:
        for i, doc in enumerate(st.session_state.last_sources, 1):
            meta = doc.metadata if hasattr(doc, "metadata") else doc
            with st.expander(
                f"📄 소스 {i}: "
                f"{meta.get('source_file', meta.get('source', 'N/A'))}",
                expanded=(i == 1),
            ):
                st.caption(f"페이지: {meta.get('page', 'N/A')}")
                content = (
                    doc.page_content
                    if hasattr(doc, "page_content")
                    else meta.get("content", "")
                )
                st.markdown(f"> {content[:500]}...")
    else:
        st.info("질문을 하면 참조 소스가 여기에 표시됩니다.")

    st.divider()

    # --- 대화 내보내기 ---
    st.header("💾 내보내기")
    if st.session_state.messages:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 JSON",
                data=export_json(st.session_state.messages),
                file_name=f"qa_{datetime.now():%Y%m%d_%H%M}.json",
                mime="application/json",
            )
        with c2:
            st.download_button(
                "📥 CSV",
                data=export_csv(st.session_state.messages),
                file_name=f"qa_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
    else:
        st.caption("대화를 시작하면 내보내기가 활성화됩니다.")


# ──────────────────────────────────────────────
# 5. 메인 영역 — 채팅 인터페이스
# ──────────────────────────────────────────────
st.title("📚 지능형 문서 QA 시스템")
st.caption("문서를 업로드하고, 내용에 대해 자유롭게 질문하세요.")

# 대화 히스토리 재생
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 채팅 입력
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    # 벡터 스토어 존재 여부 확인
    if system["vectorstore"] is None:
        st.warning("⚠️ 먼저 사이드바에서 문서를 업로드하고 인덱싱하세요.")
    else:
        # 사용자 메시지 표시 & 저장
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG 체인 빌드 & 호출
        chain = build_chain(system)
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있습니다..."):
                response = chain.invoke(
                    {"input": prompt},
                    config={
                        "configurable": {
                            "session_id": st.session_state.session_id
                        }
                    },
                )

            answer = response["answer"]
            st.markdown(answer)

        # 어시스턴트 메시지 저장
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # 소스 문서 저장 → 사이드바에 표시
        st.session_state.last_sources = response.get("context", [])

        # 사이드바 소스를 갱신하기 위해 재실행
        st.rerun()
```

**실행 방법**:

```bash
# 1. 의존성 설치
pip install streamlit langchain langchain-openai langchain-community \
    faiss-cpu python-dotenv pypdf docx2txt unstructured

# 2. API 키 설정
echo "OPENAI_API_KEY=sk-..." > .env

# 3. 앱 실행
streamlit run app.py
```

앱을 실행하면 다음과 같은 화면이 나타납니다:

- **왼쪽 사이드바**: 문서 업로드, 세션 관리, 소스 인용, 내보내기 버튼
- **가운데 메인 영역**: 채팅 인터페이스 (상단에 대화 히스토리, 하단에 입력창)

---

## 더 깊이 알아보기

### Streamlit의 탄생 — "데이터 과학자의 좌절에서 태어난 프레임워크"

2018년, Carnegie Mellon 대학교의 컴퓨터 과학 교수였던 **Adrien Treuille**는 Google X 프로젝트와 자율주행 스타트업 Zoox의 VP를 거치며 한 가지 좌절을 반복적으로 경험했습니다 — "데이터 과학자들이 만든 훌륭한 모델이, 공유할 수 있는 인터페이스가 없어서 사장되는 것"이었습니다.

Treuille는 동료인 Thiago Teixeira, Amanda Kelly와 함께 "Python 스크립트를 그대로 웹 앱으로 바꿀 수는 없을까?"라는 질문에서 출발하여 Streamlit을 만들기 시작했습니다. 2019년 가을 오픈소스로 공개된 Streamlit은 "import streamlit as st" 한 줄로 시작할 수 있는 극도의 단순함으로 데이터 커뮤니티를 사로잡았고, 2022년 Snowflake에 인수되면서 엔터프라이즈 지원까지 갖추게 됩니다.

채팅 전용 위젯(`st.chat_message`, `st.chat_input`)은 2023년 LLM 붐과 함께 추가되었는데, 이전에는 `streamlit-chat`이라는 서드파티 컴포넌트를 별도 설치해야 했습니다. Streamlit 팀이 LLM 앱의 폭발적 수요를 감지하고 네이티브 위젯으로 빠르게 편입한 사례로, 오픈소스 커뮤니티의 피드백이 프레임워크 발전을 이끄는 좋은 예입니다.

### 왜 Gradio가 아닌 Streamlit인가?

ML 데모 앱 프레임워크로 Gradio도 인기가 높습니다. 두 프레임워크의 선택 기준을 간단히 정리하면:

| 기준 | Streamlit | Gradio |
|------|-----------|--------|
| **강점** | 범용 앱, 복잡한 레이아웃, 상태 관리 | ML 데모, 빠른 Input-Output 인터페이스 |
| **레이아웃 자유도** | 높음 (sidebar, columns, tabs, expander) | 보통 (Blocks API로 가능하나 제한적) |
| **LLM 통합** | 네이티브 채팅 위젯 | ChatInterface 컴포넌트 |
| **배포** | Streamlit Cloud, Docker, 클라우드 | HuggingFace Spaces, Docker |

우리 프로젝트처럼 사이드바 소스 인용, 세션 관리, 파일 업로드 등 **복합 레이아웃이 필요한 앱**에는 Streamlit이 더 적합합니다.

---

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "`st.session_state`에 뭐든 넣으면 된다"고 생각하기 쉽지만, LLM 객체나 FAISS 인덱스 같은 **직렬화 불가능한 대규모 객체**는 `@st.cache_resource`로 관리해야 합니다. `session_state`에 넣으면 위젯 상호작용 시마다 불필요하게 복사될 수 있어 메모리 문제가 발생합니다.

> 💡 **알고 계셨나요?**: Streamlit의 `st.chat_input`은 메인 영역에 배치하면 **화면 하단에 자동 고정**됩니다. 이 동작은 `st.container()` 안에 넣으면 해제되는데, 의도적으로 인라인 채팅 입력이 필요할 때 활용할 수 있습니다. 또한 Streamlit 최신 버전에서는 `st.chat_input`이 오디오 입력과 파일 첨부도 지원합니다.

> 🔥 **실무 팁**: 프로덕션 배포 시 `.streamlit/config.toml`에 아래 설정을 추가하세요:
> ```toml
> [server]
> maxUploadSize = 500          # MB 단위, 기본 200
> enableXsrfProtection = true  # CSRF 보호
> 
> [browser]
> gatherUsageStats = false     # 텔레메트리 비활성화
> ```
> 또한 `st.secrets`를 사용하면 `.env` 대신 Streamlit 네이티브 방식으로 API 키를 관리할 수 있습니다 (`.streamlit/secrets.toml`).

> 🔥 **실무 팁**: `st.rerun()`은 강력하지만 남용하면 무한 루프에 빠질 수 있습니다. 반드시 **상태 변경 후 한 번만** 호출하고, 조건부로 실행되도록 하세요. 특히 `st.rerun()`이 `if` 블록 안에 있는지 항상 확인하세요.

---

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `st.session_state` | 재실행 간 상태(대화 히스토리, 세션 ID 등)를 보존하는 딕셔너리 |
| `st.chat_message` | 사용자/어시스턴트 역할별 말풍선 UI를 렌더링하는 위젯 |
| `st.chat_input` | 메인 영역 하단에 고정되는 채팅 입력 위젯 |
| `st.file_uploader` | 다중 파일 업로드 지원, `accept_multiple_files=True` |
| `st.expander` | 접기/펼치기 가능한 컨테이너, 소스 인용 표시에 활용 |
| `st.download_button` | 데이터를 파일로 즉시 다운로드하는 버튼 위젯 |
| `@st.cache_resource` | LLM, 벡터 스토어 등 무거운 객체를 한 번만 초기화하고 캐싱 |
| `st.rerun()` | 상태 변경 후 UI를 즉시 갱신하기 위해 스크립트를 재실행 |

---

## 다음 섹션 미리보기

다음 [18.6 테스트, 배포, 운영](ch18/session6.md)에서는 지금까지 완성한 문서 QA 시스템을 **프로덕션 환경에 배포**합니다. pytest를 활용한 RAG 파이프라인 테스트, Docker 컨테이너화, Streamlit Cloud 및 클라우드 환경(AWS/GCP) 배포, 그리고 LangSmith를 활용한 운영 모니터링까지 다룹니다. UI가 완성된 지금, 실제 사용자에게 서비스하기 위한 마지막 단계입니다.

---

## 참고 자료

- [Build a basic LLM chat app — Streamlit 공식 튜토리얼](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps) - `st.chat_message`와 `st.chat_input`을 사용한 대화형 앱 구축의 공식 가이드
- [Streamlit Chat elements API Reference](https://docs.streamlit.io/develop/api-reference/chat) - 채팅 위젯의 전체 파라미터와 사용법을 확인할 수 있는 API 레퍼런스
- [LangChain Streamlit Callback Integration](https://python.langchain.com/docs/integrations/callbacks/streamlit/) - LangChain의 StreamlitCallbackHandler를 사용한 실시간 스트리밍 출력 연동 가이드
- [Streamlit Session State API](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) - 세션 상태 관리의 공식 문서, 초기화 패턴과 주의사항 포함
- [st.file_uploader API Reference](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) - 파일 업로드 위젯의 전체 옵션과 다중 파일 처리 방법
- [st.download_button API Reference](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button) - 파일 다운로드 버튼의 JSON/CSV 내보내기 패턴

---
### 🔗 Related Sessions
- [faiss](../07-임베딩과-벡터-스토어/03-벡터-스토어-구축---faiss와-chroma.md) (prerequisite)
- [runnablewithmessagehistory](../10-메모리와-대화-관리/02-runnablewithmessagehistory.md) (prerequisite)
- [ingestionpipeline](../18-실전-프로젝트-1-지능형-문서-qa-시스템/02-문서-수집과-인덱싱-파이프라인.md) (prerequisite)
- [qapipeline](../18-실전-프로젝트-1-지능형-문서-qa-시스템/03-검색과-생성-파이프라인.md) (prerequisite)
- [conversationalqa](../18-실전-프로젝트-1-지능형-문서-qa-시스템/04-대화-관리와-메모리.md) (prerequisite)
- [citedanswer](../09-ragretrieval-augmented-generation-구축/02-rag-프롬프트-최적화.md) (prerequisite)
- [sessionmanager](../10-메모리와-대화-관리/05-멀티턴-대화-시스템-구축.md) (prerequisite)
