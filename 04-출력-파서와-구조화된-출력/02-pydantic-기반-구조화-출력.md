# Pydantic 기반 구조화 출력

> PydanticOutputParser와 Pydantic 모델을 활용하여 LLM 응답을 타입 안전하고 검증된 Python 객체로 변환하는 방법을 학습합니다.

## 개요

이 섹션에서는 앞서 [세션 4.1: 출력 파서 기초](./01-출력-파서-기초.md)에서 배운 JsonOutputParser를 한 단계 발전시켜, **Pydantic 모델 기반의 타입 검증 출력 파싱**을 다룹니다. 단순히 JSON 형태로 파싱하는 것을 넘어, Python의 강력한 데이터 검증 라이브러리인 Pydantic을 활용하여 LLM의 출력이 우리가 정의한 스키마에 정확히 부합하는지 자동으로 검증하고, 중첩된 객체나 리스트도 안전하게 파싱하는 방법을 배웁니다.

**선수 지식**: 세션 4.1에서 배운 OutputParser의 개념, `format_instructions`, `partial_variables`, 그리고 LCEL 파이프 연산자(`|`)를 사용한 체인 구성
**학습 목표**:
- PydanticOutputParser의 동작 원리와 JsonOutputParser와의 차이점을 이해한다
- Pydantic v2 모델을 정의하여 LLM 출력 스키마를 설계할 수 있다
- 중첩 객체(nested model)와 리스트를 포함한 복잡한 구조를 파싱할 수 있다
- Field의 `description`, `default`, 커스텀 `validator`를 활용한 데이터 검증을 구현할 수 있다

## 왜 알아야 할까?

실무에서 LLM을 활용한 애플리케이션을 만들 때, "LLM이 돌려준 답을 내 코드에서 어떻게 쓰지?"라는 문제에 반드시 부딪히게 됩니다. 세션 4.1에서 JsonOutputParser로 딕셔너리를 받는 법을 배웠지만, 딕셔너리에는 한 가지 치명적인 약점이 있죠 — **타입 보장이 안 됩니다**.

`result["price"]`가 문자열 `"15000"`인지, 정수 `15000`인지, 아니면 아예 없는 키인지 런타임에서야 알 수 있습니다. 이런 불확실성은 프로덕션에서 예측 불가능한 버그로 이어지거든요.

PydanticOutputParser는 이 문제를 깔끔하게 해결합니다:
- **타입 검증**: `price: int`로 선언하면 문자열이 들어와도 자동 변환하거나 에러를 발생시킵니다
- **자동 문서화**: 모델 정의 자체가 API 스펙 문서가 됩니다
- **IDE 지원**: 자동 완성, 타입 체크가 모두 작동합니다
- **중첩 구조**: 복잡한 비즈니스 데이터도 깔끔하게 모델링할 수 있습니다

## 핵심 개념

### 개념 1: PydanticOutputParser란?

> 💡 **비유**: JsonOutputParser가 "택배 상자를 열어서 내용물을 꺼내는 것"이라면, PydanticOutputParser는 **"세관 검사대"**와 같습니다. 상자를 열기만 하는 게 아니라, 내용물이 신고서(스키마)와 일치하는지 하나하나 확인하고, 규격에 맞지 않으면 통과시키지 않죠.

PydanticOutputParser는 `langchain_core.output_parsers`에서 제공하는 출력 파서로, **JsonOutputParser를 상속**하면서 Pydantic 모델 기반의 검증 기능을 추가한 것입니다. 내부적으로 다음 순서로 동작합니다:

1. LLM의 텍스트 출력에서 JSON을 추출 (JsonOutputParser의 역할)
2. 추출된 JSON을 Pydantic 모델의 `model_validate()`로 검증
3. 검증 통과 시 **Pydantic 모델 인스턴스**를 반환 (딕셔너리가 아닙니다!)
4. 검증 실패 시 `OutputParserException`을 발생시켜 명확한 에러 정보 제공

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Pydantic 모델 정의
class MovieReview(BaseModel):
    """영화 리뷰 데이터 모델"""
    title: str = Field(description="영화 제목")
    rating: float = Field(description="평점 (1.0~10.0)")
    summary: str = Field(description="한 줄 요약")

# 파서 생성 — pydantic_object에 모델 클래스를 전달
parser = PydanticOutputParser(pydantic_object=MovieReview)

# format_instructions 확인
print(parser.get_format_instructions())
# 출력: The output should be formatted as a JSON instance that conforms to the
#        JSON schema below. ... {"properties": {"title": {"description": "영화 제목", ...}, ...}}
```

`get_format_instructions()`가 반환하는 포맷 지시문에는 Pydantic 모델에서 자동 생성된 **JSON Schema**가 포함됩니다. 이 스키마에는 각 필드의 이름, 타입, description이 모두 들어있어서 LLM이 정확한 형식으로 응답하도록 유도하거든요.

### 개념 2: Pydantic v2 모델 설계하기

> 💡 **비유**: Pydantic 모델을 설계하는 것은 **주문서 양식을 만드는 것**과 비슷합니다. "이름" 칸에는 텍스트만, "수량" 칸에는 숫자만 쓸 수 있고, "배송 주소"는 필수이지만 "요청사항"은 선택인 것처럼, 각 필드의 타입과 필수 여부를 명확히 정의하는 거죠.

LangChain에서 PydanticOutputParser와 함께 사용하는 Pydantic 모델은 `BaseModel`을 상속하고, `Field`로 각 필드의 메타데이터를 지정합니다. 여기서 **`description`이 매우 중요**한데, 이 설명이 LLM에게 전달되는 JSON Schema에 포함되기 때문입니다.

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Sentiment(str, Enum):
    """감성 분석 결과 열거형"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ProductAnalysis(BaseModel):
    """제품 분석 결과 모델"""
    product_name: str = Field(
        description="분석 대상 제품명"
    )
    sentiment: Sentiment = Field(
        description="전반적 감성 (positive/negative/neutral)"
    )
    score: float = Field(
        description="감성 점수 (-1.0 ~ 1.0)",
        ge=-1.0,  # 최솟값 제약
        le=1.0    # 최댓값 제약
    )
    keywords: list[str] = Field(
        description="핵심 키워드 목록 (최대 5개)",
        max_length=5
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="개선 권고사항 (없으면 null)"
    )
```

핵심 패턴을 정리하면:
- **`description`**: LLM이 해당 필드에 무엇을 넣어야 하는지 안내 (필수!)
- **`default`**: 기본값 설정 → LLM이 해당 필드를 빠뜨려도 OK
- **`Optional[T]`**: None 허용 (default=None과 함께 사용)
- **`ge`, `le`, `max_length`**: Pydantic의 필드 제약 조건
- **Enum 타입**: LLM의 출력을 미리 정의된 값으로 제한

### 개념 3: 중첩 모델과 리스트 파싱

> 💡 **비유**: 중첩 모델은 **러시안 인형(마트료시카)**과 같습니다. 큰 인형 안에 작은 인형이, 그 안에 더 작은 인형이 들어있듯이, 하나의 모델 안에 다른 모델이 필드로 포함될 수 있습니다. LLM이 이런 복잡한 구조도 한 번에 생성하도록 할 수 있죠.

실무에서는 단일 객체보다 **중첩된 구조**를 다루는 경우가 훨씬 많습니다. 예를 들어 "레스토랑 리뷰 분석"이라면 레스토랑 정보 안에 여러 개의 리뷰가 포함되고, 각 리뷰에는 평가 항목이 있는 식이죠.

```python
from pydantic import BaseModel, Field

# 1단계: 가장 안쪽 모델부터 정의
class RatingDetail(BaseModel):
    """개별 평가 항목"""
    category: str = Field(description="평가 카테고리 (맛, 서비스, 분위기 등)")
    score: int = Field(description="점수 (1~5)", ge=1, le=5)
    comment: str = Field(description="한 줄 코멘트")

# 2단계: 중간 모델
class Review(BaseModel):
    """개별 리뷰"""
    reviewer: str = Field(description="리뷰어 이름 또는 닉네임")
    date: str = Field(description="리뷰 날짜 (YYYY-MM-DD 형식)")
    ratings: list[RatingDetail] = Field(description="세부 평가 항목 목록")
    overall_score: float = Field(description="종합 점수 (1.0~5.0)")

# 3단계: 최상위 모델
class RestaurantAnalysis(BaseModel):
    """레스토랑 분석 결과"""
    name: str = Field(description="레스토랑 이름")
    cuisine: str = Field(description="요리 종류 (한식, 양식 등)")
    reviews: list[Review] = Field(description="리뷰 목록")
    average_score: float = Field(description="전체 평균 점수")
    strengths: list[str] = Field(description="주요 강점 목록")
    weaknesses: list[str] = Field(description="주요 약점 목록")
```

이렇게 중첩 모델을 정의하면, `get_format_instructions()`가 **전체 중첩 구조의 JSON Schema**를 자동으로 생성합니다. LLM은 이 스키마를 보고 올바른 중첩 JSON을 만들어내죠.

**중요한 팁**: 중첩 깊이가 3단계를 넘어가면 LLM이 구조를 혼동할 확률이 높아집니다. 가능하면 2~3단계 이내로 유지하세요.

### 개념 4: 필드 검증과 커스텀 Validator

> 💡 **비유**: Pydantic의 validator는 **공장의 품질 검사관**입니다. 컨베이어 벨트(파싱 파이프라인)를 타고 내려온 제품(데이터)이 규격에 맞는지 하나하나 확인하고, 규격 미달이면 반려하거나 수정해서 다시 보냅니다.

Pydantic v2에서는 `field_validator`와 `model_validator` 데코레이터를 사용하여 커스텀 검증 로직을 추가할 수 있습니다.

```python
from pydantic import BaseModel, Field, field_validator, model_validator

class BookSummary(BaseModel):
    """도서 요약 모델"""
    title: str = Field(description="도서 제목")
    author: str = Field(description="저자명")
    year: int = Field(description="출판 연도")
    genre: str = Field(description="장르 (소설, 에세이, 기술서 등)")
    summary: str = Field(description="3문장 이내 요약")
    page_count: int = Field(description="총 페이지 수", gt=0)
    
    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int) -> int:
        """출판 연도가 합리적인 범위인지 검증"""
        if v < 1450 or v > 2026:
            raise ValueError(f"출판 연도 {v}는 유효 범위(1450~2026)를 벗어납니다")
        return v
    
    @field_validator("summary")
    @classmethod
    def validate_summary_length(cls, v: str) -> str:
        """요약이 너무 길지 않은지 검증"""
        sentences = [s.strip() for s in v.split(".") if s.strip()]
        if len(sentences) > 3:
            raise ValueError(f"요약이 {len(sentences)}문장입니다. 3문장 이내로 작성해주세요")
        return v
    
    @model_validator(mode="after")
    def validate_consistency(self) -> "BookSummary":
        """모델 전체의 일관성 검증"""
        if self.genre == "기술서" and self.page_count < 50:
            raise ValueError("기술서인데 50페이지 미만은 현실적이지 않습니다")
        return self
```

- **`@field_validator`**: 개별 필드 값을 검증합니다. 데코레이터에 필드 이름을 전달하고, 검증 실패 시 `ValueError`를 발생시킵니다.
- **`@model_validator(mode="after")`**: 전체 모델이 초기화된 후 필드 간 관계를 검증합니다. 여러 필드의 값을 조합한 로직 검증에 유용합니다.

LLM 출력이 이 검증을 통과하지 못하면 `OutputParserException`이 발생하며, 에러 메시지에 어떤 필드의 어떤 조건이 위반되었는지 상세히 표시됩니다.

### 개념 5: LCEL 체인에서 PydanticOutputParser 활용

앞서 세션 4.1에서 배운 `partial_variables` 패턴을 PydanticOutputParser에도 동일하게 적용할 수 있습니다. LCEL 파이프 연산자로 체인을 구성하면 깔끔하게 연결되죠.

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class TechConcept(BaseModel):
    """기술 개념 설명 모델"""
    name: str = Field(description="기술 개념 이름")
    category: str = Field(description="카테고리 (프론트엔드, 백엔드, 데이터 등)")
    explanation: str = Field(description="초보자를 위한 쉬운 설명")
    use_cases: list[str] = Field(description="대표적인 사용 사례 3가지")
    difficulty: int = Field(description="학습 난이도 (1~5)", ge=1, le=5)

# 파서 생성
parser = PydanticOutputParser(pydantic_object=TechConcept)

# 프롬프트 템플릿 — format_instructions를 partial_variables로 주입
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 IT 교육 전문가입니다. 주어진 기술 개념을 분석해주세요.\n{format_instructions}"),
    ("human", "{concept}에 대해 설명해주세요.")
])

# partial_variables로 format_instructions 주입
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# LCEL 체인 구성: 프롬프트 → 모델 → 파서
model = ChatOpenAI(model="gpt-4o", temperature=0.3)
chain = prompt | model | parser

# 실행 — 결과는 TechConcept 인스턴스!
result = chain.invoke({"concept": "Docker"})

print(type(result))       # <class '__main__.TechConcept'>
print(result.name)        # Docker
print(result.category)    # 백엔드
print(result.use_cases)   # ['마이크로서비스 배포', '개발 환경 통일', 'CI/CD 파이프라인']
print(result.difficulty)  # 3
```

반환값이 딕셔너리가 아니라 **`TechConcept` 인스턴스**라는 점을 주목하세요. `result.name`, `result.use_cases[0]`처럼 점(`.`) 표기법으로 접근할 수 있고, IDE의 자동 완성도 완벽하게 지원됩니다.

## 실습: 직접 해보기

여러 영화의 정보를 중첩 모델로 한번에 파싱하는 완전한 예제를 만들어보겠습니다.

```python
"""
실습: PydanticOutputParser로 영화 추천 데이터 구조화하기
필요 패키지: pip install langchain-core langchain-openai pydantic python-dotenv
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# ── 1단계: Pydantic 모델 정의 ──

class Actor(BaseModel):
    """배우 정보"""
    name: str = Field(description="배우 이름")
    role: str = Field(description="극 중 역할명")

class Movie(BaseModel):
    """개별 영화 정보"""
    title: str = Field(description="영화 제목")
    year: int = Field(description="개봉 연도")
    genre: str = Field(description="장르 (액션, 드라마, SF 등)")
    rating: float = Field(description="평점 (1.0~10.0)")
    plot_summary: str = Field(description="줄거리 요약 (2~3문장)")
    cast: list[Actor] = Field(description="주요 출연진 (최대 3명)")
    watch_reason: str = Field(description="이 영화를 봐야 하는 이유 한 줄")
    
    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: float) -> float:
        """평점 범위 검증"""
        if v < 1.0 or v > 10.0:
            raise ValueError(f"평점 {v}는 1.0~10.0 범위를 벗어납니다")
        return round(v, 1)  # 소수점 1자리로 반올림

class MovieRecommendations(BaseModel):
    """영화 추천 결과"""
    theme: str = Field(description="추천 테마 (예: '비 오는 날 감성 영화')")
    movies: list[Movie] = Field(description="추천 영화 목록 (3편)")
    overall_comment: str = Field(description="추천 목록에 대한 총평")

# ── 2단계: 파서 및 체인 구성 ──

# PydanticOutputParser 생성
parser = PydanticOutputParser(pydantic_object=MovieRecommendations)

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 영화 평론가입니다. 사용자의 요청에 맞는 영화를 추천해주세요.\n"
     "정확한 정보를 기반으로 답변하세요.\n"
     "{format_instructions}"),
    ("human", "{request}")
])

# format_instructions 주입
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# LLM 모델 초기화
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5  # 약간의 창의성 허용
)

# LCEL 체인: 프롬프트 → 모델 → Pydantic 파서
chain = prompt | model | parser

# ── 3단계: 실행 ──

result = chain.invoke({
    "request": "비 오는 날 혼자 보기 좋은 감성 영화 3편 추천해주세요"
})

# ── 4단계: 결과 활용 ──

# result는 MovieRecommendations 인스턴스
print(f"🎬 추천 테마: {result.theme}")
print(f"📝 총평: {result.overall_comment}")
print("-" * 50)

for i, movie in enumerate(result.movies, 1):
    print(f"\n{'='*40}")
    print(f"  #{i} {movie.title} ({movie.year})")
    print(f"  장르: {movie.genre} | 평점: {movie.rating}/10")
    print(f"  줄거리: {movie.plot_summary}")
    print(f"  출연진:")
    for actor in movie.cast:
        print(f"    - {actor.name} ({actor.role} 역)")
    print(f"  👉 {movie.watch_reason}")

# Pydantic 모델이므로 직렬화도 간편
import json
print("\n\n📋 JSON 직렬화:")
print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
```

**예상 출력:**
```
🎬 추천 테마: 비 오는 날 감성 영화
📝 총평: 빗소리를 배경으로 잔잔한 감동을 느낄 수 있는 영화들입니다.
--------------------------------------------------

========================================
  #1 노팅힐 (1999)
  장르: 로맨스 | 평점: 7.7/10
  줄거리: 런던의 작은 서점 주인이 할리우드 스타와 ...
  출연진:
    - 휴 그랜트 (윌리엄 태커 역)
    - 줄리아 로버츠 (안나 스콧 역)
    - 리스 이판스 (스파이크 역)
  👉 비 오는 런던 거리만큼 완벽한 로맨스 영화
...
```

## 더 깊이 알아보기

### Pydantic의 탄생 스토리

Pydantic은 2017년 영국 개발자 **사무엘 콜빈(Samuel Colvin)**이 만들었습니다. 그는 당시 Python 웹 프레임워크에서 데이터 검증 코드를 반복적으로 작성하는 것에 지쳐 있었거든요. "Python에는 타입 힌트가 있는데, 왜 이걸 실제 검증에 활용하지 않지?"라는 의문에서 출발한 프로젝트였습니다.

초기에는 소규모 유틸리티였지만, FastAPI의 세바스찬 라미레즈(Sebastián Ramírez)가 FastAPI의 핵심 기반으로 Pydantic을 채택하면서 폭발적으로 성장했습니다. 2023년에 출시된 **Pydantic v2**는 핵심 검증 엔진을 Rust로 재작성하여 v1 대비 **5~50배 빠른 성능**을 달성했죠. LangChain도 이 Pydantic v2를 기반으로 출력 파싱 시스템을 구축했습니다.

### PydanticOutputParser의 내부 동작

PydanticOutputParser는 `JsonOutputParser`를 상속합니다. 소스 코드를 살펴보면 핵심 메서드인 `_parse_obj`에서 Pydantic v2의 `model_validate()`를 호출하고, 검증 실패 시 `OutputParserException`으로 감싸서 던집니다:

```python
# langchain_core 내부 코드 (간략화)
def _parse_obj(self, obj: dict) -> TBaseModel:
    try:
        return self.pydantic_object.model_validate(obj)
    except pydantic.ValidationError as e:
        raise self._parser_exception(e, obj) from e
```

이 구조 덕분에 Pydantic의 **모든 검증 기능**(타입 강제 변환, 제약 조건, 커스텀 validator)이 LLM 출력 파싱에도 그대로 적용됩니다.

### JsonOutputParser vs PydanticOutputParser 비교

| 구분 | JsonOutputParser | PydanticOutputParser |
|------|-----------------|---------------------|
| 반환 타입 | `dict` | Pydantic 모델 인스턴스 |
| 타입 검증 | 없음 (JSON 파싱만) | Pydantic 기반 완전 검증 |
| IDE 지원 | `dict` 접근 (자동완성 제한) | 점 표기법 + 자동완성 |
| 직렬화 | `json.dumps(result)` | `result.model_dump()` |
| 부분 스트리밍 | 지원 | 부분 파싱 시 None 반환 가능 |
| 사용 난이도 | 쉬움 | 모델 정의 필요 |

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "PydanticOutputParser에 모델을 넘기면 LLM이 Pydantic을 이해하는 거 아닌가요?" — 아닙니다! LLM은 Pydantic을 전혀 모릅니다. `get_format_instructions()`가 모델을 **JSON Schema 텍스트**로 변환해서 프롬프트에 넣어주는 것이고, LLM은 그냥 그 JSON 스키마에 맞춰 텍스트를 생성할 뿐입니다. 실제 검증은 **파서가 응답을 받은 후** Pydantic이 수행합니다.

> 💡 **알고 계셨나요?**: Pydantic v2의 `model_validate()`는 **타입 강제 변환(coercion)**을 지원합니다. LLM이 `"rating": "8.5"`처럼 숫자를 문자열로 보내도, `rating: float` 필드라면 자동으로 `8.5`로 변환됩니다. 이 덕분에 LLM의 사소한 형식 오류에도 견고하게 동작하죠. 이를 **"관대한 파싱(lax parsing)"**이라고 하며, Pydantic v2의 기본 동작입니다.

> 🔥 **실무 팁**: Field의 `description`을 작성할 때는 **예시를 포함**하세요. `description="장르"` 보다 `description="장르 (예: 액션, 드라마, SF, 코미디)"`가 LLM의 출력 품질을 눈에 띄게 향상시킵니다. 또한 `list[str]` 필드에는 `description="핵심 키워드 목록 (최대 5개)"`처럼 **개수 힌트**를 주면 LLM이 적절한 수의 항목을 생성합니다.

> 🔥 **실무 팁**: 중첩 모델이 복잡해질수록 LLM이 실패할 확률이 높아집니다. 이럴 때는 `temperature`를 0에 가깝게 낮추고, 모델 성능이 높은 GPT-4o나 Claude Sonnet 4.6 이상을 사용하세요. 세션 4.4에서 배울 `OutputFixingParser`를 함께 사용하면 파싱 실패를 자동으로 복구할 수도 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| PydanticOutputParser | JsonOutputParser를 상속하며, Pydantic 모델 기반의 타입 검증 파싱을 수행하는 파서 |
| `pydantic_object` | 파서에 전달하는 Pydantic 모델 클래스. 이 모델의 스키마가 LLM에 전달됨 |
| Field(description=...) | 필드의 설명을 정의. JSON Schema에 포함되어 LLM이 참고함 |
| 중첩 모델 | 모델 안에 다른 모델을 필드로 포함하는 패턴. 2~3단계 이내 권장 |
| `field_validator` | Pydantic v2의 개별 필드 검증 데코레이터 |
| `model_validator` | Pydantic v2의 모델 전체 검증 데코레이터. 필드 간 관계 검증에 유용 |
| 타입 강제 변환 | Pydantic v2가 문자열 `"8.5"`를 float `8.5`로 자동 변환하는 기능 |
| `model_dump()` | Pydantic 인스턴스를 딕셔너리로 변환. JSON 직렬화에 유용 |

## 다음 섹션 미리보기

이번 세션에서 PydanticOutputParser의 강력한 구조화 능력을 확인했지만, 한 가지 궁금증이 남습니다 — "모든 LLM 출력을 Parser로 후처리해야 하나요?" 다음 세션 **4.3: with_structured_output 활용**에서는 LLM이 **네이티브하게** 구조화된 출력을 생성하는 방법을 배웁니다. 파서 없이도 Pydantic 모델에 맞는 출력을 받을 수 있는 `model.with_structured_output()` API를 다루며, PydanticOutputParser와의 차이점과 각각의 적합한 사용 시나리오를 비교해볼 것입니다.

## 참고 자료

- [PydanticOutputParser API Reference — LangChain](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html) — PydanticOutputParser의 공식 API 레퍼런스로, 모든 메서드와 파라미터를 확인할 수 있습니다
- [Pydantic 공식 문서 — Models](https://docs.pydantic.dev/latest/concepts/models/) — BaseModel, Field, validator 등 Pydantic v2의 핵심 기능을 상세히 다룹니다
- [Pydantic 공식 문서 — Validators](https://docs.pydantic.dev/latest/concepts/validators/) — field_validator와 model_validator의 사용법과 고급 패턴을 설명합니다
- [PydanticOutputParser 소스 코드 — GitHub](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/pydantic.py) — PydanticOutputParser의 실제 구현 코드를 확인할 수 있습니다
- [Control LLM output with LangChain's structured and Pydantic output parsers](https://atamel.dev/posts/2024/12-09_control_llm_output_langchain_structured_pydantic/) — PydanticOutputParser와 with_structured_output을 비교하는 실전 튜토리얼입니다

---
### 🔗 Related Sessions
- [chain](01-langchain-소개와-개발-환경-설정/01-llm-애플리케이션의-진화와-langchain.md) (prerequisite)
- [output_parser](./01-출력-파서-기초.md) (prerequisite)
- [json_output_parser](./01-출력-파서-기초.md) (prerequisite)
- [format_instructions](./01-출력-파서-기초.md) (prerequisite)
- [partial_variables](03-프롬프트-엔지니어링과-템플릿/01-chatprompttemplate-기초.md) (prerequisite)
