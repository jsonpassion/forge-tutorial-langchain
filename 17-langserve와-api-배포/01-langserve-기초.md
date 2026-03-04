# LangServe 기초

> LangChain 체인을 단 몇 줄의 코드로 REST API로 배포하는 방법을 배웁니다

## 개요

이 섹션에서는 LangServe를 사용하여 LangChain의 Runnable 객체를 REST API로 배포하는 기초를 다룹니다. `add_routes` 함수 하나로 `/invoke`, `/stream`, `/batch` 엔드포인트가 자동 생성되고, Playground UI에서 바로 테스트할 수 있는 과정을 실습합니다.

**선수 지식**: [Ch5: LCEL 마스터](05-lcellangchain-expression-language-마스터/01-lcel-기초와-파이프-연산자.md)에서 배운 파이프 연산자(`|`)와 Runnable 인터페이스, [Ch16: 콜백과 관찰 가능성](16-콜백과-관찰-가능성/01-콜백-시스템-이해.md)에서 다룬 스트리밍 개념
**학습 목표**:
- LangServe의 역할과 아키텍처를 이해할 수 있다
- `add_routes`로 체인을 REST API 엔드포인트로 노출할 수 있다
- Playground UI를 활용하여 배포된 체인을 테스트할 수 있다
- 자동 생성되는 입출력 스키마의 구조를 파악할 수 있다

## 왜 알아야 할까?

지금까지 우리는 노트북이나 스크립트에서 LangChain 체인을 실행해 왔습니다. 하지만 실제 서비스에서는 어떨까요? 웹 애플리케이션, 모바일 앱, 다른 백엔드 서비스에서 여러분의 체인을 호출하려면 **API**가 필요합니다.

물론 FastAPI로 직접 엔드포인트를 만들 수도 있겠죠. 하지만 그러려면 입력 검증, 스트리밍 응답 처리, 배치 요청 관리, API 문서화까지 모두 직접 구현해야 합니다. LangServe는 이 모든 것을 `add_routes` 함수 하나로 해결해줍니다. 마치 Django REST Framework가 Django 모델을 API로 바꿔주듯, LangServe는 LangChain Runnable을 REST API로 바꿔주는 거죠.

> 🔥 **실무 팁**: LangChain 팀은 새 프로젝트에는 LangGraph Platform을 권장하고 있습니다. 하지만 LangServe는 여전히 유지보수되고 있으며, 빠른 프로토타이핑과 간단한 체인 배포에는 LangServe의 간결함이 큰 장점입니다. LangServe를 이해하면 LangGraph Platform으로의 전환도 수월해집니다.

## 핵심 개념

### 개념 1: LangServe란 무엇인가?

> 💡 **비유**: LangServe는 **요리사와 손님 사이의 웨이터**와 같습니다. 여러분이 만든 체인(요리)을 손님(클라이언트)에게 전달하려면 주문을 받고, 요리를 서빙하고, 여러 테이블의 주문을 동시에 처리하는 웨이터가 필요하죠. LangServe가 바로 그 웨이터 역할을 합니다. 주문서 양식(입출력 스키마)도 자동으로 만들어주고, 코스 요리처럼 하나씩 내오는 것(스트리밍)도 지원합니다.

LangServe는 LangChain의 Runnable 객체를 REST API로 배포할 수 있게 해주는 라이브러리입니다. 내부적으로 **FastAPI**를 기반으로 하며, **Pydantic**을 사용하여 데이터 검증을 수행합니다.

LangServe가 자동으로 생성하는 엔드포인트는 다음과 같습니다:

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/invoke` | POST | 단일 입력에 대한 동기 호출 |
| `/batch` | POST | 여러 입력을 한 번에 처리 |
| `/stream` | POST | 스트리밍 응답 반환 |
| `/stream_log` | POST | 중간 단계 포함 스트리밍 |
| `/stream_events` | POST | 이벤트 기반 스트리밍 (v0.0.40+) |
| `/input_schema` | GET | 입력 JSON Schema |
| `/output_schema` | GET | 출력 JSON Schema |
| `/config_schema` | GET | 설정 JSON Schema |
| `/playground` | GET | 대화형 테스트 UI |

하나의 `add_routes` 호출로 이 모든 것이 생성된다는 게 놀랍지 않나요?

### 개념 2: add_routes — 핵심 함수

> 💡 **비유**: `add_routes`는 **자판기 설치 기사**와 같습니다. 여러분이 만든 음료(체인)를 넣어주면, 자판기(FastAPI 앱)에 동전 투입구(입력), 음료 배출구(출력), 메뉴판(스키마)을 자동으로 설치해줍니다. 경로(`path`)만 지정하면 끝이죠.

`add_routes`의 기본 사용법은 매우 간단합니다:

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI

# FastAPI 앱 생성
app = FastAPI(
    title="My LangChain API",
    version="1.0",
    description="LangServe로 배포한 LangChain API",
)

# 체인을 API로 등록 — 이 한 줄이 전부입니다!
add_routes(app, ChatOpenAI(model="gpt-4o"), path="/chat")
```

`add_routes` 함수의 주요 매개변수를 살펴보겠습니다:

```python
add_routes(
    app,                          # FastAPI 앱 또는 APIRouter
    runnable,                     # LangChain Runnable 객체 (체인, 모델 등)
    path="/my-chain",             # API 경로 (기본값: "")
    input_type=Optional[Type],    # 입력 타입 명시 (자동 추론도 가능)
    output_type=Optional[Type],   # 출력 타입 명시 (자동 추론도 가능)
    per_req_config_modifier=None, # 요청별 설정 수정 함수
    enabled_endpoints=None,       # 활성화할 엔드포인트 목록
)
```

여러 체인을 하나의 앱에 등록할 수도 있습니다:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 모델 정의
model = ChatOpenAI(model="gpt-4o")

# 번역 체인
translate_prompt = ChatPromptTemplate.from_template(
    "Translate the following text to {language}: {text}"
)
translate_chain = translate_prompt | model | StrOutputParser()

# 요약 체인
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in 3 sentences: {text}"
)
summarize_chain = summarize_prompt | model | StrOutputParser()

# 각 체인을 별도 경로에 등록
add_routes(app, translate_chain, path="/translate")
add_routes(app, summarize_chain, path="/summarize")
```

### 개념 3: Playground UI — 브라우저에서 바로 테스트

서버를 실행하면 각 체인의 Playground에 접속할 수 있습니다. 예를 들어 `/translate` 경로로 등록했다면, `http://localhost:8000/translate/playground/`에서 대화형 UI를 사용할 수 있죠.

Playground가 제공하는 기능은 다음과 같습니다:

- **입력 폼 자동 생성**: 체인의 입력 스키마를 분석하여 적절한 입력 필드를 자동으로 만들어줍니다
- **스트리밍 출력 표시**: 응답이 실시간으로 표시됩니다
- **중간 단계 확인**: 체인의 각 단계별 출력을 확인할 수 있습니다
- **설정 공유**: URL을 통해 특정 설정을 다른 사람과 공유할 수 있습니다

Playground는 별도의 설정 없이 `add_routes`를 호출하면 자동으로 활성화됩니다. 개발 중에 Postman이나 curl 없이도 체인을 빠르게 테스트할 수 있어 매우 편리합니다.

### 개념 4: 입출력 스키마 자동 생성

> 💡 **비유**: 입출력 스키마 자동 생성은 **레스토랑 메뉴판 자동 인쇄**와 같습니다. 요리(체인)를 등록하면 재료(입력), 완성된 요리 사진(출력), 알레르기 정보(타입 정보)가 적힌 메뉴판이 자동으로 만들어지는 거죠.

LangServe는 Runnable 객체의 타입 정보를 분석하여 **JSON Schema**를 자동으로 생성합니다. 이 스키마는 다음 용도로 사용됩니다:

1. **입력 검증**: 잘못된 형식의 요청을 자동으로 거부하고, 상세한 에러 메시지를 반환합니다
2. **API 문서 생성**: `/docs` 경로에서 Swagger UI로 전체 API 문서를 확인할 수 있습니다
3. **Playground 폼 생성**: 입력 스키마에 맞는 UI 위젯을 자동으로 구성합니다

스키마를 직접 확인하려면 `GET /my-chain/input_schema`와 `GET /my-chain/output_schema`를 호출하면 됩니다:

```python
import requests

# 입력 스키마 확인
response = requests.get("http://localhost:8000/translate/input_schema")
print(response.json())
# {
#   "title": "translate_chain_input",
#   "type": "object",
#   "properties": {
#     "language": {"title": "Language", "type": "string"},
#     "text": {"title": "Text", "type": "string"}
#   },
#   "required": ["language", "text"]
# }
```

필요하다면 `input_type`과 `output_type`을 Pydantic 모델로 명시하여 스키마를 더 구체적으로 제어할 수도 있습니다:

```python
from pydantic import BaseModel, Field

class TranslateInput(BaseModel):
    """번역 요청 입력"""
    text: str = Field(description="번역할 텍스트")
    language: str = Field(description="대상 언어", default="English")

add_routes(
    app,
    translate_chain,
    path="/translate",
    input_type=TranslateInput,  # 명시적 입력 타입
)
```

## 실습: 직접 해보기

완전한 LangServe 서버를 처음부터 만들어 보겠습니다.

**1단계: 패키지 설치**

```bash
# LangServe 설치 (서버 + 클라이언트 모두 포함)
pip install "langserve[all]" langchain-openai python-dotenv
```

**2단계: 환경 변수 설정 (.env 파일)**

```
OPENAI_API_KEY=your-api-key-here
```

**3단계: 서버 코드 작성 (server.py)**

```python
"""LangServe 기초 실습 — 첫 번째 API 서버"""

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# .env에서 API 키 로드
load_dotenv()

# ── 모델 설정 ──────────────────────────────────────
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.7

model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# ── 체인 1: 농담 생성기 ─────────────────────────────
joke_prompt = ChatPromptTemplate.from_template(
    "{topic}에 대한 재미있는 농담을 하나 만들어주세요. "
    "한국어로 답변하세요."
)
joke_chain = joke_prompt | model | StrOutputParser()

# ── 체인 2: 개념 설명기 ─────────────────────────────
explain_prompt = ChatPromptTemplate.from_template(
    "{concept}을(를) {level} 수준으로 쉽게 설명해주세요. "
    "비유를 활용하고 한국어로 답변하세요."
)
explain_chain = explain_prompt | model | StrOutputParser()

# ── FastAPI 앱 생성 ─────────────────────────────────
app = FastAPI(
    title="LangServe 실습 API",
    version="1.0",
    description="LangServe로 배포한 첫 번째 LangChain API 서버",
)

# ── 체인을 API로 등록 ───────────────────────────────
add_routes(app, joke_chain, path="/joke")       # 농담 생성 엔드포인트
add_routes(app, explain_chain, path="/explain")  # 개념 설명 엔드포인트
add_routes(app, model, path="/chat")             # 모델 직접 호출 엔드포인트

if __name__ == "__main__":
    import uvicorn
    # 개발 서버 실행 (포트 8000)
    uvicorn.run(app, host="localhost", port=8000)
```

**4단계: 서버 실행**

```bash
python server.py
```

서버가 시작되면 다음 URL로 접속해보세요:
- Swagger 문서: `http://localhost:8000/docs`
- 농담 생성 Playground: `http://localhost:8000/joke/playground/`
- 개념 설명 Playground: `http://localhost:8000/explain/playground/`

**5단계: 클라이언트로 API 호출 (client.py)**

```python
"""LangServe 클라이언트 — 서버에 배포된 체인 호출"""

from langserve import RemoteRunnable

# ── RemoteRunnable로 원격 체인 연결 ─────────────────
joke_chain = RemoteRunnable("http://localhost:8000/joke/")
explain_chain = RemoteRunnable("http://localhost:8000/explain/")

# ── 동기 호출 (invoke) ──────────────────────────────
result = joke_chain.invoke({"topic": "프로그래밍"})
print("🎭 농담:", result)
# 🎭 농담: 프로그래머가 왜 안경을 쓸까요? C#을 못 봐서요!

# ── 스트리밍 호출 (stream) ──────────────────────────
print("\n📖 설명: ", end="")
for chunk in explain_chain.stream({
    "concept": "REST API",
    "level": "초등학생",
}):
    print(chunk, end="", flush=True)  # 토큰 단위로 출력
print()

# ── 배치 호출 (batch) ───────────────────────────────
topics = [
    {"topic": "인공지능"},
    {"topic": "고양이"},
    {"topic": "커피"},
]
results = joke_chain.batch(topics)
for topic, result in zip(topics, results):
    print(f"\n🎭 {topic['topic']}: {result}")
```

**6단계: HTTP 클라이언트로 직접 호출 (선택)**

LangServe 클라이언트 없이 `requests`로도 호출할 수 있습니다:

```python
"""HTTP 클라이언트로 LangServe API 직접 호출"""

import requests

# invoke 엔드포인트 호출
response = requests.post(
    "http://localhost:8000/joke/invoke",
    json={"input": {"topic": "파이썬"}},  # input 키로 감싸야 합니다
)
print(response.json())
# {"output": "파이썬이 길을 건넜습니다. 왜냐고요? ...", "metadata": {...}}

# 입력 스키마 확인
schema = requests.get("http://localhost:8000/joke/input_schema")
print(schema.json())
```

> ⚠️ **흔한 오해**: HTTP로 직접 호출할 때 입력 데이터를 `{"topic": "파이썬"}`처럼 보내면 안 됩니다. 반드시 `{"input": {"topic": "파이썬"}}` 형태로 `input` 키로 한 번 감싸야 합니다. 반면 `RemoteRunnable`을 사용하면 이 래핑을 자동으로 처리해줍니다.

## 더 깊이 알아보기

### LangServe의 탄생 이야기

LangServe는 2023년 10월, LangChain 생태계의 "배포 문제"를 해결하기 위해 탄생했습니다. Harrison Chase가 LangChain을 2022년 10월에 오픈소스로 공개한 이후, 커뮤니티에서 가장 많이 받은 질문 중 하나가 바로 **"만든 체인을 어떻게 배포하나요?"**였습니다.

당시 개발자들은 FastAPI로 직접 엔드포인트를 만들거나, Streamlit으로 데모를 만들었는데, 스트리밍 응답 처리, 동시 요청 관리, 입력 검증 등을 매번 직접 구현해야 했습니다. LangServe는 이 반복 작업을 제거하기 위해 만들어졌죠. LCEL(LangChain Expression Language)의 통합 Runnable 인터페이스 덕분에, 어떤 체인이든 동일한 방식으로 API화할 수 있게 된 것입니다.

흥미로운 점은 LangServe의 이름입니다. "Serve"는 테니스에서 공을 넘기는 동작을 의미하는데, LangServe의 공식 이모지가 🏓(탁구)인 이유가 여기에 있습니다. 체인이라는 공을 서버에서 클라이언트로 "서브"한다는 의미를 담고 있죠.

### LangServe의 현재와 미래

2024년 말부터 LangChain 팀은 새로운 프로젝트에는 **LangGraph Platform**을 권장하고 있습니다. LangGraph Platform은 상태 지속성, 메모리 관리, Human-in-the-Loop, 크론 작업, 웹훅 등 프로덕션에 필요한 고급 기능을 기본 제공합니다. 하지만 LangServe는 여전히 버그 수정이 이루어지고 있으며, 간단한 체인 배포나 학습 목적으로는 여전히 최적의 선택입니다. LangServe에서 익힌 개념(엔드포인트 구조, 스키마, 스트리밍)은 LangGraph Platform에서도 그대로 적용됩니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "LangServe는 deprecated(사용 중단)되었다"라고 알고 계신 분이 많은데, 정확하지 않습니다. LangChain 팀은 새 프로젝트에 LangGraph Platform을 *권장*하지만, LangServe는 여전히 유지보수되고 있으며 버그 수정도 계속됩니다. 특히 간단한 체인 배포에는 LangServe가 훨씬 간결합니다.

> 💡 **알고 계셨나요?**: LangServe의 Playground에는 보안 취약점 이력이 있습니다. v0.0.13~v0.0.15에서 Playground 엔드포인트를 통한 임의 파일 접근 취약점이 발견되어 v0.0.16에서 수정되었습니다. 항상 최신 버전(현재 0.3.x)을 사용하세요!

> 🔥 **실무 팁**: 개발 중에는 `uvicorn.run(app, host="localhost", port=8000)`으로 실행하지만, 프로덕션에서는 `uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4`처럼 멀티 워커를 사용하세요. `--reload` 플래그를 추가하면 코드 변경 시 자동 재시작도 가능합니다.

> 🔥 **실무 팁**: LangServe가 Pydantic v2와 호환되려면 **0.3.0 이상** 버전이 필요합니다. `pip install "langserve[all]>=0.3.0"`으로 설치하면 Pydantic v2 환경에서도 OpenAPI 문서가 정상적으로 생성됩니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| LangServe | LangChain Runnable을 REST API로 배포하는 라이브러리 (FastAPI 기반) |
| `add_routes` | 체인을 FastAPI 앱에 등록하는 핵심 함수. 한 줄로 9개 엔드포인트 생성 |
| 자동 엔드포인트 | `/invoke`, `/batch`, `/stream`, `/stream_log`, `/stream_events` + 스키마 3종 |
| Playground | `/playground/` 경로에서 접근하는 대화형 테스트 UI |
| 입출력 스키마 | Runnable의 타입 정보에서 JSON Schema를 자동 생성하여 입력 검증과 문서화에 활용 |
| `RemoteRunnable` | 서버에 배포된 체인을 로컬 체인처럼 호출할 수 있는 클라이언트 |
| 설치 | `pip install "langserve[all]"` (서버+클라이언트) 또는 `[server]`/`[client]` 분리 설치 |

## 다음 섹션 미리보기

이번 섹션에서 LangServe의 기초를 익혔다면, 다음 섹션에서는 **FastAPI 통합과 미들웨어 설정**을 깊이 다룹니다. CORS 설정, 인증 미들웨어 추가, 커스텀 엔드포인트 결합 등 실제 프로덕션 환경에서 필요한 서버 구성 방법을 배우게 됩니다. `per_req_config_modifier`를 활용한 요청별 인증 처리도 함께 살펴보겠습니다.

## 참고 자료

- [LangServe 공식 문서](https://python.langchain.com/docs/langserve) - LangChain 공식 사이트의 LangServe 가이드로, 설치부터 배포까지 전체 과정을 다룹니다
- [LangServe GitHub 리포지토리](https://github.com/langchain-ai/langserve) - 소스 코드, 예제 프로젝트, 최신 릴리스 정보를 확인할 수 있습니다
- [LangServe GitHub Releases](https://github.com/langchain-ai/langserve/releases) - 버전별 변경사항과 최신 릴리스(0.3.3) 정보
- [Koyeb - Using LangServe to build REST APIs](https://www.koyeb.com/tutorials/using-langserve-to-build-rest-apis-for-langchain-applications) - 클라우드 환경에 LangServe를 배포하는 실전 튜토리얼
- [LangGraph Platform 권장 안내](https://github.com/langchain-ai/langserve/issues/791) - LangServe와 LangGraph Platform의 관계 및 마이그레이션 가이드

---
### 🔗 Related Sessions
- [lcel](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [runnable](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [chain](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
