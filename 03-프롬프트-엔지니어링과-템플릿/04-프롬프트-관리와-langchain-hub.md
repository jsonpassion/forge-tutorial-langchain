# 프롬프트 관리와 LangChain Hub

> LangChain Hub를 활용한 프롬프트 검색, 공유, 버전 관리, 그리고 평가 기초를 배웁니다

## 개요

이 섹션에서는 프로덕션 환경에서 프롬프트를 체계적으로 관리하는 방법을 배웁니다. 지금까지 코드 안에서 프롬프트를 직접 작성했다면, 이번에는 한 단계 나아가 **LangChain Hub**에서 검증된 프롬프트를 가져오고, 자신만의 프롬프트를 공유하며, 버전을 관리하는 방법을 익힙니다. 마지막으로 프롬프트가 실제로 "잘 작동하는지" 평가하는 기초도 함께 다룹니다.

**선수 지식**: [세션 3.1: ChatPromptTemplate 기초](./01-chatprompttemplate-기초.md)에서 배운 `ChatPromptTemplate` 생성과 `invoke()`, [세션 3.3: Few-shot 프롬프팅](./03-few-shot-프롬프팅.md)에서 배운 `FewShotChatMessagePromptTemplate`

**학습 목표**:
- LangChain Hub에서 프롬프트를 검색하고 `hub.pull()`로 가져올 수 있다
- `hub.push()`로 자신의 프롬프트를 Hub에 공유할 수 있다
- 커밋 해시와 태그를 활용한 프롬프트 버전 관리 전략을 수립할 수 있다
- 데이터셋 기반의 프롬프트 평가 기초를 이해하고 적용할 수 있다

## 왜 알아야 할까?

프로젝트 초기에는 프롬프트 하나둘을 코드에 직접 넣어도 괜찮습니다. 하지만 프로젝트가 커지면 어떻게 될까요? 프롬프트가 10개, 50개, 100개로 늘어나면요?

"지난주에 잘 작동하던 프롬프트가 뭐였지?", "이 프롬프트 누가 바꿨어?", "프로덕션에 배포된 프롬프트 버전이 뭐야?" — 이런 질문이 매일 쏟아지게 됩니다. 코드는 Git으로 버전 관리를 하면서, 정작 LLM 애플리케이션의 **핵심**인 프롬프트는 관리하지 않는 건 이상하지 않나요?

LangChain Hub와 LangSmith의 프롬프트 관리 기능은 이 문제를 정면으로 해결합니다. 커뮤니티가 검증한 프롬프트를 재활용하고, 팀 내에서 프롬프트를 공유하며, Git처럼 버전을 추적하고, 변경 전후의 품질을 비교 평가할 수 있습니다.

## 핵심 개념

### 개념 1: LangChain Hub — 프롬프트의 앱스토어

> 💡 **비유**: LangChain Hub는 **프롬프트의 앱스토어**라고 생각하면 됩니다. 앱스토어에서 다른 개발자가 만든 앱을 다운로드하듯, Hub에서 다른 개발자가 만든 프롬프트를 `pull`로 가져옵니다. 내가 만든 좋은 프롬프트는 `push`로 올려서 공유할 수도 있죠.

LangChain Hub([smith.langchain.com/hub](https://smith.langchain.com/hub))는 LangSmith 플랫폼 위에서 운영되는 프롬프트 저장소입니다. 이름, 용도, 모델별로 프롬프트를 검색할 수 있고, 커뮤니티가 공유한 수백 개의 프롬프트를 즉시 사용할 수 있습니다.

Hub에서 프롬프트를 가져오는 방법은 놀라울 정도로 간단합니다:

```python
from langchain import hub

# Hub에서 유명한 RAG 프롬프트 가져오기
rag_prompt = hub.pull("rlm/rag-prompt")

# 가져온 프롬프트의 내용 확인
print(type(rag_prompt))  # <class 'langchain_core.prompts.chat.ChatPromptTemplate'>
print(rag_prompt.input_variables)  # ['context', 'question']

# 바로 사용 가능!
result = rag_prompt.invoke({
    "context": "LangChain은 LLM 기반 앱 개발 프레임워크입니다.",
    "question": "LangChain이 뭔가요?"
})
print(result)
```

`hub.pull()`의 인자는 `"소유자/프롬프트이름"` 형식입니다. `"rlm/rag-prompt"`에서 `rlm`은 소유자(Harrison Chase의 별칭), `rag-prompt`는 프롬프트 이름이죠.

특정 버전을 가져오고 싶다면 **커밋 해시**를 뒤에 붙입니다:

```python
# 특정 버전의 프롬프트 가져오기 (커밋 해시 지정)
specific_prompt = hub.pull("rlm/rag-prompt:c9839f14")

# 태그로 가져오기 (예: production 버전)
prod_prompt = hub.pull("rlm/rag-prompt:prod")
```

### 개념 2: 프롬프트 공유 — hub.push()

내가 만든 프롬프트를 Hub에 올리는 것도 간단합니다. 먼저 LangSmith API 키가 필요한데요, 환경 변수로 설정해두면 됩니다.

```python
import os
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# LangSmith API 키 설정 (.env 파일 권장)
os.environ["LANGSMITH_API_KEY"] = "your-api-key-here"

# 프롬프트 생성
my_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {domain} 분야의 전문가입니다. "
               "초보자도 이해할 수 있게 쉽게 설명해주세요."),
    ("human", "{question}")
])

# Hub에 푸시 — URL이 반환됩니다
url = hub.push(
    "my-handle/domain-expert",     # 소유자/프롬프트이름
    my_prompt,                      # 업로드할 프롬프트 객체
    new_repo_description="분야별 전문가 프롬프트",  # 설명
    new_repo_is_public=True         # 공개 여부
)
print(f"프롬프트 URL: {url}")
# 출력: https://smith.langchain.com/hub/my-handle/domain-expert
```

이미 존재하는 프롬프트에 새 버전을 올리면 자동으로 **새로운 커밋**이 생성됩니다. Git의 커밋과 비슷한 개념이죠.

```python
# 프롬프트 수정 후 다시 푸시 → 새 커밋 생성
updated_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {domain} 분야의 전문가입니다. "
               "핵심만 간결하게 3줄 이내로 답변해주세요."),  # 수정됨
    ("human", "{question}")
])

# 같은 이름으로 푸시하면 새 버전이 됩니다
url = hub.push("my-handle/domain-expert", updated_prompt)
```

### 개념 3: 버전 관리 — 커밋과 태그

> 💡 **비유**: 프롬프트 버전 관리는 **요리 레시피북**과 같습니다. 요리사가 레시피를 계속 개선하면서 "v1: 소금 1스푼", "v2: 소금 반 스푼"처럼 변경 이력을 남기듯, 프롬프트도 매 변경마다 커밋 해시가 남습니다. 그리고 "이번 시즌 대표 메뉴"처럼 태그를 붙여서 특정 버전을 고정할 수 있죠.

LangSmith의 프롬프트 버전 관리는 두 가지 핵심 개념으로 이루어집니다:

**커밋(Commit)**: `hub.push()`를 할 때마다 고유한 커밋 해시가 생성됩니다. 이 해시로 정확한 버전을 참조할 수 있습니다.

**태그(Tag)**: 커밋 해시는 외우기 어렵습니다. 그래서 `dev`, `staging`, `prod` 같은 사람이 읽기 쉬운 레이블을 붙일 수 있는데, 이것이 태그입니다.

```python
from langsmith import Client

client = Client()

# 프롬프트 푸시 후 태그 설정
# 1) 개발 환경용 태그
client.push_prompt(
    "my-handle/domain-expert",
    object=my_prompt,
    tags=["dev"]         # "dev" 태그 부여
)

# 2) 검증 완료 후 프로덕션 태그로 승격
client.push_prompt(
    "my-handle/domain-expert",
    object=updated_prompt,
    tags=["prod", "v2"]  # "prod"와 "v2" 태그 동시 부여
)
```

코드에서 태그로 프롬프트를 가져오면, **코드 변경 없이** 프롬프트만 교체할 수 있습니다:

```python
from langchain import hub

# 프로덕션 코드에서는 항상 "prod" 태그를 참조
prompt = hub.pull("my-handle/domain-expert:prod")

# prompt 내용이 바뀌어도 코드는 그대로!
# Hub에서 "prod" 태그를 새 커밋으로 옮기기만 하면 됩니다
```

이 패턴의 핵심은 **코드 배포와 프롬프트 배포를 분리**할 수 있다는 것입니다. 프롬프트만 수정하고 싶을 때 코드를 재배포할 필요가 없거든요.

### 개념 4: 프롬프트 평가 기초

프롬프트를 변경했을 때 "정말 나아졌는지" 어떻게 확인할까요? 감으로요? 그건 위험합니다. LangSmith는 **데이터셋 기반 평가** 기능을 제공합니다.

평가에는 세 가지 요소가 필요합니다:

1. **데이터셋**: 테스트 입력과 기대 출력의 모음
2. **타겟 함수**: 평가할 프롬프트 + 모델 조합
3. **평가자(Evaluator)**: 출력을 점수 매기는 함수

```python
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

client = Client()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 1단계: 데이터셋 생성
dataset = client.create_dataset("prompt-eval-dataset")

# 테스트 케이스 추가
client.create_examples(
    inputs=[
        {"question": "파이썬에서 리스트와 튜플의 차이는?"},
        {"question": "REST API란 무엇인가요?"},
        {"question": "Git의 브랜치란?"},
    ],
    outputs=[
        {"answer": "리스트는 변경 가능(mutable), 튜플은 변경 불가(immutable)"},
        {"answer": "HTTP 메서드를 사용하여 리소스를 조작하는 아키텍처 스타일"},
        {"answer": "독립적인 작업 흐름을 위한 코드 분기"},
    ],
    dataset_id=dataset.id
)

# 2단계: 타겟 함수 정의
prompt_v1 = ChatPromptTemplate.from_messages([
    ("system", "기술 질문에 한 줄로 답하세요."),
    ("human", "{question}")
])
chain_v1 = prompt_v1 | llm

def target_v1(inputs: dict) -> dict:
    """평가 대상 함수"""
    response = chain_v1.invoke(inputs)
    return {"answer": response.content}

# 3단계: 평가 실행
results = client.evaluate(
    target_v1,
    data="prompt-eval-dataset",
    experiment_prefix="v1-prompt-test"  # 실험 이름
)
print(f"평가 결과: {results}")
```

> 🔥 **실무 팁**: 처음에는 10~20개의 수작업 테스트 케이스로 시작하세요. "좋은 응답"이 무엇인지 정의하는 것이 평가의 첫걸음입니다. 완벽한 데이터셋을 만들려다 시작도 못하는 것보다, 작게 시작해서 점점 늘리는 게 훨씬 낫습니다.

## 실습: 직접 해보기

Hub에서 프롬프트를 가져와 수정하고, 버전을 관리하는 전체 워크플로우를 실습합니다.

```python
"""
프롬프트 관리 실습: Hub 연동 + 버전 관리 + A/B 비교
필요 패키지: pip install langchain langchain-openai langsmith python-dotenv
"""
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()
# .env 파일에 아래 키들을 설정하세요:
# OPENAI_API_KEY=sk-...
# LANGSMITH_API_KEY=lsv2_...
# LANGCHAIN_TRACING_V2=true

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── 1단계: Hub에서 프롬프트 가져오기 ──
print("=" * 50)
print("1단계: Hub에서 프롬프트 가져오기")
print("=" * 50)

# 유명한 RAG 프롬프트 가져오기
rag_prompt = hub.pull("rlm/rag-prompt")
print(f"프롬프트 타입: {type(rag_prompt).__name__}")
print(f"입력 변수: {rag_prompt.input_variables}")

# 가져온 프롬프트 사용
chain = rag_prompt | llm
response = chain.invoke({
    "context": "LangChain Hub는 프롬프트 검색과 공유를 위한 플랫폼입니다. "
               "hub.pull()과 hub.push() API를 제공합니다.",
    "question": "LangChain Hub에서 프롬프트를 어떻게 가져오나요?"
})
print(f"\n응답: {response.content}\n")

# ── 2단계: 커스텀 프롬프트 만들고 Hub에 올리기 ──
print("=" * 50)
print("2단계: 커스텀 프롬프트 생성 및 업로드")
print("=" * 50)

# 한국어 기술 설명 프롬프트 (v1)
tech_prompt_v1 = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 IT 기술 용어 해설가입니다.\n"
     "주어진 기술 용어를 다음 형식으로 설명하세요:\n"
     "1. 한 줄 정의\n"
     "2. 비유를 통한 설명\n"
     "3. 실제 사용 예시"),
    ("human", "'{term}' 용어를 설명해주세요.")
])

# Hub에 푸시 (LangSmith 계정 필요)
# url = hub.push(
#     "my-handle/korean-tech-glossary",
#     tech_prompt_v1,
#     new_repo_description="한국어 기술 용어 해설 프롬프트",
#     new_repo_is_public=True
# )
# print(f"업로드 완료: {url}")

# 로컬에서 바로 테스트
chain_v1 = tech_prompt_v1 | llm
response_v1 = chain_v1.invoke({"term": "API"})
print(f"[v1 응답]\n{response_v1.content}\n")

# ── 3단계: 프롬프트 수정 (v2) ──
print("=" * 50)
print("3단계: 프롬프트 개선 (v2)")
print("=" * 50)

# 더 구조화된 버전 (v2)
tech_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 IT 기술 용어 해설가입니다.\n"
     "주어진 기술 용어를 다음 형식으로 설명하세요:\n\n"
     "📌 **한 줄 정의**: (15자 이내)\n"
     "🎯 **비유**: (초등학생도 이해할 수 있는 비유)\n"
     "💻 **코드 예시**: (Python 3줄 이내)\n"
     "⚡ **핵심 포인트**: (실무에서 가장 중요한 한 가지)"),
    ("human", "'{term}' 용어를 설명해주세요.")
])

chain_v2 = tech_prompt_v2 | llm
response_v2 = chain_v2.invoke({"term": "API"})
print(f"[v2 응답]\n{response_v2.content}\n")

# ── 4단계: A/B 비교 (간단한 평가) ──
print("=" * 50)
print("4단계: 프롬프트 A/B 비교")
print("=" * 50)

test_terms = ["API", "Docker", "Git"]

for term in test_terms:
    r1 = chain_v1.invoke({"term": term})
    r2 = chain_v2.invoke({"term": term})

    # 간단한 구조화 점수: 이모지/마크다운 포맷 포함 여부
    v1_structured = sum(1 for marker in ["📌", "🎯", "💻", "⚡"]
                        if marker in r1.content)
    v2_structured = sum(1 for marker in ["📌", "🎯", "💻", "⚡"]
                        if marker in r2.content)

    print(f"\n용어: {term}")
    print(f"  v1 구조화 점수: {v1_structured}/4")
    print(f"  v2 구조화 점수: {v2_structured}/4")
    print(f"  v1 응답 길이: {len(r1.content)}자")
    print(f"  v2 응답 길이: {len(r2.content)}자")

print("\n✅ 실습 완료! v2가 더 구조화된 응답을 생성하는지 확인해보세요.")
```

## 더 깊이 알아보기

### LangChain Hub의 탄생 이야기

LangChain Hub의 역사를 이해하려면 LangChain 자체의 탄생부터 살펴봐야 합니다. 2022년 10월, Harrison Chase는 머신러닝 스타트업 Robust Intelligence에서 일하며 한 가지 문제를 발견했습니다 — 개발자들이 LLM 애플리케이션을 만들 때마다 비슷한 코드를 반복해서 작성하고 있었던 거죠. 그는 단 10일 만에(10월 16~25일) LangChain의 첫 버전을 만들었는데, 처음에는 Python의 `formatter.format()`을 감싸는 단순한 프롬프트 래퍼에 불과했습니다.

프로젝트가 폭발적으로 성장하면서, 커뮤니티에서 자연스럽게 "잘 만든 프롬프트를 공유하고 싶다"는 요구가 생겼습니다. 이에 Harrison은 GitHub 저장소(`hwchase17/langchain-hub`)로 프롬프트를 모으기 시작했고, 이것이 훗날 LangSmith 플랫폼 위의 **LangChain Hub**로 진화했습니다.

흥미로운 점은, 초기 Hub가 단순한 텍스트 파일 모음이었다는 겁니다. JSON 파일에 프롬프트를 저장하고 GitHub PR로 공유하는 방식이었죠. 지금의 Hub는 버전 관리, 태그, Playground 테스트까지 지원하는 본격적인 프롬프트 관리 플랫폼이 되었습니다. "좋은 추상화는 커뮤니티의 필요에서 자란다"는 오픈소스의 교훈이 잘 드러나는 사례입니다.

### 프롬프트 엔지니어링의 "테스트" 문화

소프트웨어 개발에서 단위 테스트가 당연해지기까지 수십 년이 걸렸습니다. Kent Beck이 1998년에 JUnit을 만들고, TDD(Test-Driven Development)를 주창한 이후에야 "코드를 쓰기 전에 테스트를 먼저 쓴다"는 문화가 자리잡았죠.

프롬프트 엔지니어링도 지금 비슷한 변곡점에 서 있습니다. 2023~2024년까지만 해도 프롬프트를 "감"으로 평가하는 팀이 대부분이었습니다. 하지만 LangSmith의 평가 프레임워크, Braintrust, Promptfoo 같은 도구들이 등장하면서 "프롬프트도 테스트해야 한다"는 인식이 빠르게 확산되고 있습니다. 앞서 배운 데이터셋 기반 평가는 이 흐름의 시작점입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Hub에서 가져온 프롬프트는 그대로 쓰면 된다." — Hub의 프롬프트는 **출발점**이지 완성품이 아닙니다. 대부분 영어 기반이고, 특정 도메인에 최적화되어 있지 않습니다. 반드시 자신의 use case에 맞게 수정하고 테스트한 후 사용하세요.

> 💡 **알고 계셨나요?**: `hub.pull()`로 가져온 객체는 일반 `ChatPromptTemplate`과 동일합니다. 즉, [세션 3.2](./02-고급-프롬프트-패턴.md)에서 배운 프롬프트 합성(`+` 연산자)이나 `MessagesPlaceholder`를 그대로 적용할 수 있습니다. Hub 프롬프트를 기반으로 자신만의 확장 프롬프트를 만드는 것이 가장 효율적인 활용법이죠.

> 🔥 **실무 팁**: 프로덕션 환경에서는 반드시 **태그 기반 배포** 패턴을 사용하세요. `hub.pull("my-org/prompt:prod")`처럼 `prod` 태그를 참조하면, 코드 배포 없이 프롬프트만 교체할 수 있습니다. 배포 흐름은 `dev` → `staging` → `prod` 순으로 태그를 승격시키는 것이 안전합니다.

> ⚠️ **흔한 오해**: "`LANGSMITH_API_KEY`만 있으면 Hub를 쓸 수 있다." — `hub.pull()`로 **공개 프롬프트를 가져오는 것**은 API 키 없이도 가능합니다. 하지만 `hub.push()`로 프롬프트를 **업로드**하거나, 비공개 프롬프트에 접근하려면 LangSmith API 키가 필요합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| LangChain Hub | 프롬프트를 검색, 공유, 관리하는 LangSmith 기반 플랫폼 |
| `hub.pull()` | Hub에서 프롬프트를 가져오는 함수. `"소유자/이름:버전"` 형식 |
| `hub.push()` | 프롬프트를 Hub에 업로드하는 함수. 자동으로 새 커밋 생성 |
| 커밋 해시 | 프롬프트의 고유 버전 식별자. Git 커밋 해시와 유사 |
| 태그 | 커밋에 붙이는 사람이 읽기 쉬운 레이블 (`dev`, `prod` 등) |
| 태그 기반 배포 | 코드 변경 없이 태그만 옮겨서 프롬프트를 교체하는 패턴 |
| 데이터셋 기반 평가 | 테스트 입력/기대 출력 + 타겟 함수 + 평가자로 프롬프트 품질 측정 |

## 다음 섹션 미리보기

지금까지 챕터 3에서 `ChatPromptTemplate` 기초부터 고급 패턴, Few-shot 프롬프팅, 그리고 프롬프트 관리까지 여정을 함께했습니다. 마지막 세션인 **세션 3.5: 프롬프트 엔지니어링 베스트 프랙티스**에서는 이 모든 기법을 종합하여, 실전에서 프롬프트를 설계하고 반복 개선하는 전략을 배웁니다. 체계적인 프롬프트 디자인 원칙, 디버깅 기법, 그리고 프로덕션 환경에서의 프롬프트 최적화 노하우를 다룰 예정입니다.

## 참고 자료

- [LangChain Hub 공식 페이지](https://smith.langchain.com/hub) — Hub에서 직접 프롬프트를 검색하고 Playground에서 테스트해볼 수 있습니다
- [Manage Prompts — LangSmith 공식 문서](https://docs.langchain.com/langsmith/manage-prompts) — 프롬프트 생성, 버전 관리, 태그, SDK 연동 방법을 상세히 다루는 공식 가이드
- [LangSmith 평가 퀵스타트](https://docs.langchain.com/langsmith/evaluation-quickstart) — 데이터셋 기반 평가를 처음 시작할 때 참고할 공식 튜토리얼
- [langchain.hub API 레퍼런스](https://python.langchain.com/api_reference/langchain/hub.html) — `hub.pull()`, `hub.push()`의 전체 파라미터와 사용법
- [LangSmith Prompt Versioning Cookbook](https://github.com/langchain-ai/langsmith-cookbook/blob/main/hub-examples/retrieval-qa-chain-versioned/prompt-versioning.ipynb) — 프롬프트 버전 관리를 RAG 체인에 적용하는 실전 예제 노트북

---
### 🔗 Related Sessions
- [lcel](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [invoke](01-langchain-소개와-개발-환경-설정/04-첫-번째-langchain-애플리케이션.md) (prerequisite)
- [chatprompttemplate](01-langchain-소개와-개발-환경-설정/04-첫-번째-langchain-애플리케이션.md) (prerequisite)
- [messagesplaceholder](./02-고급-프롬프트-패턴.md) (prerequisite)
- [prompt_composition](./02-고급-프롬프트-패턴.md) (prerequisite)
