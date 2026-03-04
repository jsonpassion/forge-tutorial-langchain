# LangSmith 트레이싱 심화

> LangSmith의 프로젝트 관리, 트레이스 필터링, 피드백 수집, 주석 큐를 활용하여 LLM 애플리케이션의 실행을 체계적으로 추적하고 개선하는 방법을 배웁니다.

## 개요

이 섹션에서는 LangSmith의 핵심 기능인 프로젝트와 실행(Run) 관리, 트레이스 필터링과 검색, 프로그래밍 방식의 피드백 수집, 그리고 주석(Annotation) 큐를 깊이 있게 다룹니다. 앞서 [16.1: 콜백 시스템 이해](ch16/session_16_1.md)와 [16.2: 커스텀 콜백 핸들러](ch16/session_16_2.md)에서 콜백을 통해 실행 이벤트를 추적하는 방법을 배웠다면, 이번 섹션에서는 그 데이터가 **LangSmith 플랫폼에서 어떻게 구조화되고, 검색되며, 평가에 활용되는지** 전체 그림을 완성합니다.

**선수 지식**: 콜백 시스템의 동작 원리(16.1), 커스텀 콜백 핸들러 작성법(16.2), LangChain 기본 체인 구성(Ch5 LCEL)
**학습 목표**:
- LangSmith의 프로젝트, 트레이스, 런의 계층 구조를 이해하고 프로그래밍 방식으로 관리할 수 있다
- LangSmith SDK를 사용하여 트레이스를 필터링하고 검색할 수 있다
- 런에 프로그래밍 방식으로 피드백을 생성하고 조회할 수 있다
- 주석 큐를 생성하고 휴먼 피드백 루프를 구축할 수 있다

## 왜 알아야 할까?

콜백 핸들러로 로그를 찍고 토큰을 추적하는 것은 시작에 불과합니다. 실제 프로덕션 환경에서는 이런 질문들이 쏟아지거든요:

- "지난주에 비용이 갑자기 뛴 체인이 뭐였지?"
- "사용자가 '답변이 틀렸다'고 한 케이스만 모아서 볼 수 없나?"
- "품질 검수팀이 체계적으로 LLM 응답을 평가할 수 있는 워크플로우가 필요한데..."

LangSmith는 이 모든 질문에 답할 수 있는 **관찰 가능성(Observability) 플랫폼**입니다. 단순한 로그 뷰어가 아니라, 트레이스를 구조화하고, 검색하고, 평가하고, 개선 사이클을 돌리는 **피드백 루프 전체**를 지원하죠. Netflix가 마이크로서비스 장애를 분산 트레이싱으로 추적하는 것처럼, LangSmith는 LLM 앱의 "분산 트레이싱 + 품질 관리 시스템"인 셈입니다.

## 핵심 개념

### 개념 1: 프로젝트, 트레이스, 런의 계층 구조

> 💡 **비유**: 프로젝트(Project)는 **도서관**, 트레이스(Trace)는 도서관 안의 **책 한 권**, 런(Run)은 책의 **각 장(Chapter)**이라고 생각하면 됩니다. 도서관에 여러 권의 책이 있고, 각 책은 여러 장으로 구성되듯이 — 하나의 프로젝트 안에 여러 트레이스가 있고, 각 트레이스는 여러 런으로 구성됩니다.

LangSmith의 데이터 모델은 세 가지 핵심 계층으로 이루어져 있습니다:

| 계층 | 설명 | 비유 |
|------|------|------|
| **Project** | 트레이스의 컬렉션. 애플리케이션 또는 서비스 단위 | 도서관 |
| **Trace** | 단일 요청의 전체 실행 흐름 | 책 한 권 |
| **Run** | 트레이스 내 개별 실행 단계 (LLM, 체인, 도구 등) | 책의 각 장 |

추가로 **Thread**라는 개념도 있는데요, 대화형 애플리케이션에서 여러 턴(turn)의 트레이스를 하나의 대화 세션으로 묶어주는 역할을 합니다.

프로젝트는 환경 변수로 간편하게 설정할 수 있습니다:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# .env 파일에 설정
# LANGCHAIN_TRACING_V2=true        # 트레이싱 활성화
# LANGCHAIN_API_KEY=lsv2_pt_...    # LangSmith API 키
# LANGCHAIN_PROJECT=my-chatbot-v2  # 프로젝트 이름

# 또는 코드에서 직접 설정
os.environ["LANGCHAIN_PROJECT"] = "my-chatbot-v2"
```

LangSmith SDK를 사용하면 프로그래밍 방식으로 프로젝트를 관리할 수 있습니다:

```python
from langsmith import Client

client = Client()

# 프로젝트 목록 조회
for project in client.list_projects():
    print(f"프로젝트: {project.name}, 런 수: {project.run_count}")

# 특정 프로젝트의 루트 트레이스 조회
root_runs = client.list_runs(
    project_name="my-chatbot-v2",
    is_root=True,  # 루트 트레이스만 조회
)
for run in root_runs:
    print(f"트레이스 ID: {run.trace_id}")
    print(f"  상태: {run.status}")
    print(f"  지연 시간: {run.latency}s")
    print(f"  토큰: {run.total_tokens}")
```

### 개념 2: 메타데이터와 태그로 실행 추적 강화하기

> 💡 **비유**: 도서관의 책에 **분류 스티커**와 **색인 카드**를 붙이는 것과 같습니다. 태그(Tag)는 "소설", "추리" 같은 분류 스티커고, 메타데이터(Metadata)는 "저자: 김작가, 출판년도: 2025" 같은 상세 정보가 적힌 색인 카드입니다.

실행에 메타데이터와 태그를 추가하면 나중에 강력한 필터링이 가능해집니다. LangChain의 `config` 딕셔너리를 통해 런마다 메타데이터를 주입할 수 있거든요:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 체인 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    ("human", "{question}")
])
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
chain = prompt | llm | StrOutputParser()

# 메타데이터와 태그를 포함하여 실행
result = chain.invoke(
    {"question": "LangChain이 뭔가요?"},
    config={
        "metadata": {
            "user_id": "user_123",          # 사용자 식별
            "session_id": "sess_abc",        # 세션 식별
            "app_version": "2.1.0",          # 앱 버전
            "environment": "production",     # 실행 환경
        },
        "tags": ["production", "chatbot", "v2.1"],  # 분류 태그
        "run_name": "사용자 질문 응답",               # 커스텀 런 이름
    }
)
print(result)
```

`@traceable` 데코레이터를 사용하면 LangChain 외부 코드도 트레이싱할 수 있습니다:

```python
from langsmith import traceable

@traceable(
    name="문서_전처리",
    metadata={"pipeline": "rag", "step": "preprocessing"},
    tags=["rag", "preprocessing"]
)
def preprocess_document(text: str) -> str:
    """문서를 전처리하여 청크로 분할합니다."""
    # 전처리 로직
    cleaned = text.strip().lower()
    return cleaned

# 이 함수 호출이 자동으로 LangSmith에 트레이싱됩니다
result = preprocess_document("  Hello World!  ")
```

### 개념 3: 트레이스 필터링과 검색

> 💡 **비유**: LangSmith의 필터링 시스템은 이메일의 **고급 검색 필터**와 비슷합니다. Gmail에서 "보낸 사람: alice, 첨부파일 있음, 지난 주" 같은 조건으로 검색하듯이, LangSmith에서도 "에러 발생, gpt-4o 모델, 지난 24시간" 같은 복합 조건으로 트레이스를 찾을 수 있죠.

LangSmith는 키-값 기반 필터링을 지원하며, 런당 최대 100개의 고유 키를 인덱싱합니다. SDK의 `list_runs`는 매우 유연한 필터링 옵션을 제공합니다:

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# 1. 최근 24시간 내 에러가 발생한 런만 조회
error_runs = client.list_runs(
    project_name="my-chatbot-v2",
    error=True,                                    # 에러 런만
    start_time=datetime.now() - timedelta(hours=24) # 최근 24시간
)
for run in error_runs:
    print(f"에러 런: {run.id}")
    print(f"  에러: {run.error}")
    print(f"  입력: {run.inputs}")

# 2. 특정 런 타입으로 필터링 (llm, chain, tool, retriever)
llm_runs = client.list_runs(
    project_name="my-chatbot-v2",
    run_type="llm",  # LLM 호출만 필터
)

# 3. 필요한 필드만 선택하여 조회 (성능 최적화)
lightweight_runs = client.list_runs(
    project_name="my-chatbot-v2",
    is_root=True,
    select=["inputs", "outputs", "feedback_stats"],  # 필요한 필드만
)

# 4. LangSmith 쿼리 언어를 사용한 고급 필터링
filtered_runs = client.list_runs(
    project_name="my-chatbot-v2",
    filter='and(eq(metadata_key, "environment"), eq(metadata_value, "production"))',
)

# 5. 피드백 기반 필터링
low_score_runs = client.list_runs(
    project_name="my-chatbot-v2",
    filter='and(eq(feedback_key, "user_score"), lt(feedback_score, 0.5))',
)
print("낮은 평가를 받은 런:")
for run in low_score_runs:
    print(f"  {run.id}: {run.outputs}")
```

> ⚠️ **흔한 오해**: `list_runs`가 모든 런을 메모리에 한꺼번에 로드한다고 생각하기 쉽지만, 실제로는 **제너레이터(generator)**를 반환합니다. 따라서 수백만 개의 런이 있어도 메모리 걱정 없이 순회할 수 있습니다. 다만, 전체 런을 리스트로 변환하면(`list(client.list_runs(...))`) 메모리 문제가 생길 수 있으니 주의하세요!

### 개념 4: 프로그래밍 방식 피드백 수집

> 💡 **비유**: 피드백은 음식점의 **별점 리뷰** 시스템과 같습니다. 각 리뷰(피드백)는 특정 방문(런)에 대해, 특정 기준(키)으로, 점수(스코어)와 코멘트를 남기는 구조입니다. "맛 4점, 서비스 5점, 분위기 3점"처럼 하나의 런에 여러 기준의 피드백을 붙일 수 있죠.

LangSmith의 피드백은 `run_id`, `key`, `score` 세 가지 핵심 요소로 구성됩니다:

```python
from langsmith import Client
import uuid

client = Client()

# 1. 특정 런에 피드백 생성
run_id = "여기에_실제_run_id"  # 실제 사용 시 런 ID 입력

# 수치 점수 피드백
client.create_feedback(
    run_id=run_id,
    key="correctness",      # 피드백 기준 (키)
    score=0.8,               # 0~1 사이 점수
    comment="대부분 정확하나 날짜가 틀림",  # 선택적 코멘트
)

# 카테고리 피드백
client.create_feedback(
    run_id=run_id,
    key="sentiment",
    value="positive",        # 카테고리 값
    comment="사용자가 만족한 응답",
)

# 수정 제안 피드백
client.create_feedback(
    run_id=run_id,
    key="correction",
    correction={             # 올바른 출력 제안
        "output": "수정된 정확한 답변 내용"
    },
    comment="날짜를 2025년 3월로 수정",
)
```

자동화된 피드백 수집 파이프라인도 구축할 수 있습니다. 예를 들어, 사용자의 엄지척(thumbs up/down)을 피드백으로 연결하는 패턴:

```python
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.context import collect_runs

client = Client()

prompt = ChatPromptTemplate.from_messages([
    ("system", "질문에 간결하게 답하세요."),
    ("human", "{question}")
])
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm | StrOutputParser()

# collect_runs 컨텍스트 매니저로 런 ID 캡처
with collect_runs() as cb:
    result = chain.invoke({"question": "Python의 GIL이 뭔가요?"})
    run_id = cb.traced_runs[0].id  # 루트 런 ID 캡처

print(f"응답: {result}")
print(f"런 ID: {run_id}")

# 사용자 피드백 시뮬레이션 (실제로는 UI에서 수집)
user_liked = True  # 사용자가 좋아요를 눌렀다고 가정

client.create_feedback(
    run_id=run_id,
    key="user_score",
    score=1.0 if user_liked else 0.0,
    feedback_source_type="app",  # 앱에서 수집된 피드백
    comment="사용자 직접 평가",
)
print("피드백이 LangSmith에 기록되었습니다!")
```

### 개념 5: 주석(Annotation) 큐

> 💡 **비유**: 주석 큐는 **서류 결재함**과 같습니다. 검토가 필요한 서류(런)를 결재함(큐)에 넣어두면, 결재자(어노테이터)가 순서대로 꺼내서 검토하고 의견(피드백)을 적는 거죠. 여러 결재함을 만들어 용도별로 분류할 수도 있고, 특정 조건의 서류만 자동으로 결재함에 들어가게 할 수도 있습니다.

주석 큐는 **휴먼-인-더-루프(Human-in-the-Loop)** 품질 관리의 핵심입니다. LangSmith는 두 가지 스타일의 주석 큐를 지원합니다:

1. **단일 런 주석 큐**: 한 번에 하나의 런을 검토하며 피드백을 작성
2. **페어와이즈(Pairwise) 주석 큐**: 두 런을 나란히 비교하여 어떤 출력이 더 좋은지 판단 (A/B 테스트)

SDK로 주석 큐를 생성하고 런을 할당하는 방법:

```python
from langsmith import Client

client = Client()

# 1. 주석 큐 생성
queue = client.create_annotation_queue(
    name="품질 검수 큐",
    description="프로덕션 챗봇 응답 품질 검수를 위한 큐"
)
print(f"큐 생성 완료: {queue.id}")

# 2. 특정 조건의 런을 큐에 추가
# 예: 사용자 평가가 낮은 런을 자동으로 큐에 넣기
low_score_runs = client.list_runs(
    project_name="my-chatbot-v2",
    filter='and(eq(feedback_key, "user_score"), lt(feedback_score, 0.5))',
)

run_ids = [run.id for run in low_score_runs]
if run_ids:
    client.add_runs_to_annotation_queue(
        queue_id=queue.id,
        run_ids=run_ids
    )
    print(f"{len(run_ids)}개의 런이 큐에 추가되었습니다.")
```

주석 큐를 활용한 **자동 품질 관리 파이프라인** 패턴:

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

def auto_triage_runs(
    project_name: str,
    queue_name: str,
    hours: int = 24,
    error_queue_name: str = "에러 검수 큐"
):
    """최근 런을 자동 분류하여 적절한 큐에 할당합니다."""

    # 에러 런 → 에러 검수 큐
    error_queue = client.create_annotation_queue(
        name=error_queue_name,
        description="에러가 발생한 런 검수"
    )
    error_runs = client.list_runs(
        project_name=project_name,
        error=True,
        start_time=datetime.now() - timedelta(hours=hours),
    )
    error_ids = [r.id for r in error_runs]
    if error_ids:
        client.add_runs_to_annotation_queue(
            queue_id=error_queue.id,
            run_ids=error_ids
        )
        print(f"에러 큐에 {len(error_ids)}개 할당")

    # 낮은 평가 런 → 품질 검수 큐
    quality_queue = client.create_annotation_queue(
        name=queue_name,
        description="낮은 사용자 평가를 받은 런 검수"
    )
    low_score_runs = client.list_runs(
        project_name=project_name,
        filter='and(eq(feedback_key, "user_score"), lt(feedback_score, 0.5))',
        start_time=datetime.now() - timedelta(hours=hours),
    )
    low_ids = [r.id for r in low_score_runs]
    if low_ids:
        client.add_runs_to_annotation_queue(
            queue_id=quality_queue.id,
            run_ids=low_ids
        )
        print(f"품질 큐에 {len(low_ids)}개 할당")

# 매일 자동 실행 (예: cron, Airflow 등)
auto_triage_runs(
    project_name="my-chatbot-v2",
    queue_name="일일 품질 검수",
)
```

## 실습: 직접 해보기

아래는 LangSmith 트레이싱의 핵심 기능을 한 번에 체험할 수 있는 통합 실습 코드입니다. 프로젝트 관리, 메타데이터 부착, 피드백 생성, 검색, 주석 큐 할당까지 전 과정을 다룹니다.

```python
"""
LangSmith 트레이싱 심화 실습
- 프로젝트 관리, 피드백 수집, 주석 큐를 종합적으로 실습합니다.

사전 준비:
  pip install langchain langchain-openai langsmith python-dotenv

.env 파일:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=lsv2_pt_...
  OPENAI_API_KEY=sk-...
"""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.context import collect_runs
from langsmith import Client

# 환경 변수 로드
load_dotenv()

# 프로젝트 이름 설정
PROJECT_NAME = "langsmith-tracing-lab"
os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME

# LangSmith 클라이언트 초기화
client = Client()

# ── 1단계: 체인 구성 및 메타데이터와 함께 실행 ──
print("=" * 60)
print("1단계: 체인 실행 (메타데이터 + 태그 부착)")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 Python 전문가입니다. 간결하게 답변하세요."),
    ("human", "{question}")
])
llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = prompt | llm | StrOutputParser()

# 여러 질문을 다른 메타데이터로 실행
questions = [
    {"question": "리스트 컴프리헨션이 뭔가요?", "difficulty": "beginner"},
    {"question": "GIL의 한계를 우회하는 방법은?", "difficulty": "advanced"},
    {"question": "데코레이터의 동작 원리를 설명해주세요.", "difficulty": "intermediate"},
]

run_ids = []  # 런 ID 수집용

for q in questions:
    with collect_runs() as cb:
        result = chain.invoke(
            {"question": q["question"]},
            config={
                "metadata": {
                    "difficulty": q["difficulty"],  # 난이도 메타데이터
                    "category": "python-basics",
                    "timestamp": datetime.now().isoformat(),
                },
                "tags": ["lab", q["difficulty"]],
                "run_name": f"Q: {q['question'][:20]}...",
            }
        )
        run_id = cb.traced_runs[0].id
        run_ids.append(run_id)

    print(f"\n질문: {q['question']}")
    print(f"  난이도: {q['difficulty']}")
    print(f"  런 ID: {run_id}")
    print(f"  답변: {result[:80]}...")

# ── 2단계: 프로그래밍 방식 피드백 생성 ──
print("\n" + "=" * 60)
print("2단계: 피드백 생성")
print("=" * 60)

# 시뮬레이션된 사용자 피드백 (실제로는 UI에서 수집)
feedback_data = [
    {"score": 1.0, "comment": "완벽한 설명!"},
    {"score": 0.3, "comment": "너무 어렵게 설명함"},
    {"score": 0.7, "comment": "좋지만 예제가 부족"},
]

for run_id, fb in zip(run_ids, feedback_data):
    # 사용자 만족도 피드백
    client.create_feedback(
        run_id=run_id,
        key="user_score",
        score=fb["score"],
        comment=fb["comment"],
        feedback_source_type="app",
    )

    # 정확도 자동 평가 피드백 (실제로는 평가 체인 사용)
    client.create_feedback(
        run_id=run_id,
        key="auto_accuracy",
        score=0.9,  # 자동 평가 점수
        comment="자동 평가: 핵심 키워드 포함 확인",
        feedback_source_type="model",
    )
    print(f"런 {str(run_id)[:8]}... → 사용자 점수: {fb['score']}, 코멘트: {fb['comment']}")

# ── 3단계: 트레이스 검색 및 필터링 ──
print("\n" + "=" * 60)
print("3단계: 트레이스 검색")
print("=" * 60)

# 프로젝트의 모든 루트 런 조회
print("\n[모든 루트 런]")
root_runs = client.list_runs(
    project_name=PROJECT_NAME,
    is_root=True,
)
for run in root_runs:
    status = "✓" if run.status == "success" else "✗"
    print(f"  {status} {run.name} | 토큰: {run.total_tokens} | 지연: {run.latency:.2f}s")

# 피드백 조회
print("\n[피드백 조회]")
for run_id in run_ids:
    feedbacks = list(client.list_feedback(run_ids=[run_id]))
    for fb in feedbacks:
        print(f"  런 {str(run_id)[:8]}... → {fb.key}: {fb.score} ({fb.comment})")

# ── 4단계: 주석 큐 생성 및 런 할당 ──
print("\n" + "=" * 60)
print("4단계: 주석 큐")
print("=" * 60)

# 낮은 점수 런을 검수 큐에 할당
queue = client.create_annotation_queue(
    name=f"품질 검수 - {datetime.now().strftime('%Y%m%d')}",
    description="사용자 만족도가 낮은 런을 검수합니다."
)
print(f"큐 생성: {queue.name} (ID: {queue.id})")

# 점수 0.5 미만인 런만 큐에 추가
low_score_ids = [
    run_id for run_id, fb in zip(run_ids, feedback_data)
    if fb["score"] < 0.5
]

if low_score_ids:
    client.add_runs_to_annotation_queue(
        queue_id=queue.id,
        run_ids=low_score_ids,
    )
    print(f"  → {len(low_score_ids)}개의 런이 큐에 할당됨")
else:
    print("  → 낮은 점수 런 없음")

print("\n실습 완료! LangSmith 대시보드에서 결과를 확인하세요.")
print(f"  프로젝트: {PROJECT_NAME}")
print(f"  URL: https://smith.langchain.com")

# ── 실행 결과 (예시) ──
# ============================================================
# 1단계: 체인 실행 (메타데이터 + 태그 부착)
# ============================================================
# 질문: 리스트 컴프리헨션이 뭔가요?
#   난이도: beginner
#   런 ID: a1b2c3d4-...
#   답변: 리스트 컴프리헨션은 기존 리스트를 기반으로 새로운 리스트를 간결하게 생성하는...
#
# 2단계: 피드백 생성
# 런 a1b2c3d4... → 사용자 점수: 1.0, 코멘트: 완벽한 설명!
# 런 e5f6g7h8... → 사용자 점수: 0.3, 코멘트: 너무 어렵게 설명함
#
# 4단계: 주석 큐
# 큐 생성: 품질 검수 - 20260303 (ID: ...)
#   → 1개의 런이 큐에 할당됨
```

## 더 깊이 알아보기

### LangSmith의 탄생 이야기

LangChain의 창시자 **Harrison Chase**는 2022년 말, 머신러닝 스타트업 Robust Intelligence에서 일하면서 개인 사이드 프로젝트로 LangChain을 시작했습니다. "거창한 계획 없이 개인 GitHub에 올린 단일 Python 패키지"에서 출발한 것이죠.

그런데 LangChain이 폭발적으로 성장하면서 개발자들의 공통된 불만이 나타났습니다: **"체인이 왜 이렇게 동작하는지 알 수가 없다!"** LLM 호출은 비결정적이고, 체인이 복잡해질수록 디버깅은 악몽이 됐거든요. 전통적인 소프트웨어의 스택 트레이스로는 프롬프트에 뭐가 들어갔는지, 모델이 뭘 반환했는지 추적하기 어려웠습니다.

이 문제를 해결하기 위해 2023년 7월, LangChain 팀은 **LangSmith**를 베타로 출시합니다. 분산 트레이싱에서 영감을 받은 이 플랫폼은 체인의 모든 실행 단계를 시각적으로 보여주는 "LLM 앱 전용 X-ray"였죠. 2024년 2월까지 5,000개 이상의 월간 기업 사용자를 확보했고, 2025년 10월에는 12.5억 달러 기업가치로 1.25억 달러 투자를 유치했습니다.

재미있는 것은 이름의 유래인데요. "Lang"은 LangChain에서, "Smith"는 **대장장이(blacksmith)**에서 따왔습니다. 대장장이가 금속을 두드려 도구를 만들듯, LangSmith는 LLM 앱을 다듬고 개선하는 도구라는 의미를 담고 있습니다.

### 관찰 가능성(Observability)의 세 기둥

분산 시스템의 관찰 가능성은 세 가지 신호로 구성됩니다:

| 신호 | 전통적 시스템 | LangSmith 대응 |
|------|-------------|---------------|
| **로그(Logs)** | 애플리케이션 텍스트 로그 | 런의 입력/출력, 에러 메시지 |
| **메트릭(Metrics)** | CPU, 메모리, 요청률 | 토큰 사용량, 지연 시간, 비용 |
| **트레이스(Traces)** | 분산 요청 추적 | 체인 실행 트리, 부모-자식 런 관계 |

LangSmith는 이 세 기둥을 LLM 앱에 맞게 재해석하여, 피드백(Feedback)이라는 네 번째 차원을 추가한 것이 차별점입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "LangSmith는 LangChain 전용이다." — 아닙니다! `@traceable` 데코레이터와 `wrap_openai`를 사용하면 순수 OpenAI SDK 코드, 심지어 어떤 Python 함수든 LangSmith로 트레이싱할 수 있습니다. LangChain 없이도 독립적으로 사용 가능합니다.

> 💡 **알고 계셨나요?**: LangSmith의 필터링 엔진은 내부적으로 **nltk 불용어(stop word) 리스트**를 사용합니다. 키워드 검색 시 "the", "is", "a" 같은 일반적인 단어와 JSON 구조 키워드는 자동으로 제외되기 때문에, 검색할 때는 **구체적인 기술 용어나 고유 식별자**를 사용하는 것이 효과적입니다.

> 🔥 **실무 팁**: 프로덕션 환경에서는 `LANGCHAIN_PROJECT`를 환경별로 분리하세요. 예를 들어 `my-app-dev`, `my-app-staging`, `my-app-prod`처럼 나누면 프로덕션 트레이스가 개발 데이터와 섞이지 않습니다. 메타데이터에 `git_commit_hash`를 넣으면 어떤 코드 버전에서 문제가 발생했는지 즉시 추적할 수 있습니다.

> 🔥 **실무 팁**: 주석 큐를 활용한 **주간 품질 리뷰** 패턴이 효과적입니다. 매주 월요일에 지난 주 낮은 평가 런을 자동으로 큐에 넣고, 팀원들이 돌아가며 검수하면 지속적인 품질 개선이 가능합니다. Cron이나 Airflow로 `auto_triage_runs` 함수를 스케줄링하면 완전 자동화됩니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **Project** | 트레이스의 컬렉션으로, `LANGCHAIN_PROJECT` 환경 변수로 설정 |
| **Trace / Run** | 트레이스는 단일 요청의 전체 실행 흐름, 런은 트레이스 내 개별 단계 |
| **Metadata / Tags** | 런에 부착하는 키-값 쌍과 분류 레이블로, 필터링과 검색에 활용 |
| **`@traceable`** | LangChain 외부 코드도 트레이싱할 수 있는 데코레이터 |
| **`list_runs()`** | 프로젝트의 런을 조건별로 검색하는 SDK 메서드 (제너레이터 반환) |
| **`create_feedback()`** | 런에 점수, 값, 코멘트를 부착하는 SDK 메서드 |
| **Annotation Queue** | 휴먼 리뷰어가 체계적으로 런을 검수하기 위한 큐 시스템 |
| **Pairwise Queue** | 두 런을 나란히 비교하여 A/B 평가를 수행하는 큐 |

## 다음 섹션 미리보기

이번 섹션에서 LangSmith로 트레이스를 관리하고 피드백을 수집하는 방법을 배웠다면, 다음 섹션 **[16.4: LLM 애플리케이션 평가 파이프라인](ch16/session_16_4.md)**에서는 수집된 데이터를 활용해 **체계적인 평가(Evaluation)**를 수행하는 방법을 다룹니다. 데이터셋 생성, 자동화된 평가 메트릭, 그리고 LangSmith Evaluation 기능을 활용한 A/B 테스트까지 — 피드백 루프의 다음 단계로 나아갑니다.

## 참고 자료

- [LangSmith 공식 문서 — Observability Concepts](https://docs.smith.langchain.com/observability/concepts) - 프로젝트, 트레이스, 런, 피드백의 핵심 개념을 정의하는 공식 레퍼런스
- [LangSmith Python SDK — Client API Reference](https://langsmith-sdk.readthedocs.io/en/latest/client.html) - `list_runs`, `create_feedback`, `create_annotation_queue` 등 모든 SDK 메서드의 상세 시그니처
- [LangSmith 주석 큐 가이드](https://docs.langchain.com/langsmith/annotation-queues) - 주석 큐 생성, 피드백 기준 설정, 페어와이즈 비교까지 단계별 가이드
- [LangSmith 트레이스 필터링 가이드](https://docs.langchain.com/langsmith/filter-traces-in-application) - 쿼리 언어, 메타데이터/태그 기반 필터링, 트레이스 뷰 내 검색 방법
- [Harrison Chase — LangChain 3년의 회고](https://blog.langchain.com/three-years-langchain/) - LangChain과 LangSmith의 탄생 배경, 성장 과정, 미래 비전을 담은 창시자의 회고록
- [LangSmith Tracing Deep Dive — Beyond the Docs](https://medium.com/@aviadr1/langsmith-tracing-deep-dive-beyond-the-docs-75016c91f747) - 공식 문서에서 다루지 않는 실전 트레이싱 패턴과 팁

---
### 🔗 Related Sessions
- [lcel](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [chain](../01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [callback_handler](../16-콜백과-관찰-가능성/01-콜백-시스템-이해.md) (prerequisite)
- [callback_propagation](../16-콜백과-관찰-가능성/01-콜백-시스템-이해.md) (prerequisite)
- [constructor_callback](../16-콜백과-관찰-가능성/01-콜백-시스템-이해.md) (prerequisite)
- [request_callback](../16-콜백과-관찰-가능성/01-콜백-시스템-이해.md) (prerequisite)
