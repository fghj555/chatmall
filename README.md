# ChatMall 이커머스 챗봇

자연어 처리 기능을 활용한 AI 기반 쇼핑몰 챗봇. 
상품 검색부터 주문 완료까지 대화형으로 처리.

## 주요 기능

- **AI 상품 검색**: OpenAI GPT와 벡터 유사도를 활용한 자연어 상품 검색
- **대화형 주문 프로세스**: 옵션 선택, 수량 입력, 배송 정보 수집까지 단계별 주문 처리
- **다국어 지원**: 한국어, 영어 언어 감지 및 적절한 언어로 응답
- **Google Sheets 연동**: 주문 정보 자동 저장 및 관리
- **Redis 세션 관리**: 대화 기록 및 컨텍스트 유지
- **Milvus 벡터 데이터베이스**: 임베딩 기반 고도화된 상품 검색
- **묶음 배송 계산**: 상품별 묶음 단위 배송비 자동 계산

## 기술 스택

- **웹 프레임워크**: FastAPI
- **AI/ML**: OpenAI GPT-4.1 mini, LangChain, OpenAI Embeddings
- **데이터베이스**: Milvus (벡터 데이터베이스), Redis (세션 저장소)
- **API 연동**: Facebook Messenger API, Google Sheets API
- **배포 환경**: CloudType
- **언어**: Python 3.8+

## 설치 및 설정

### 시스템 요구사항

- Python 3.8 이상
- Redis 서버
- Milvus 데이터베이스 접근 권한
- Facebook 앱 (Messenger 권한 포함)
- Google 서비스 계정 (Sheets API 권한)
- OpenAI API 키

### 설치 과정

1. 저장소 클론
```bash
git clone https://github.com/your-username/facebook-chatbot.git
cd facebook-chatbot
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
# OpenAI 설정
OPENAI_API_KEY=your_openai_api_key_here

# Facebook Messenger 설정
VERIFY_TOKEN=your_facebook_verify_token
PAGE_ACCESS_TOKEN=your_facebook_page_access_token

# Redis 설정
REDIS_URL=redis://localhost:6379/0

# Google Sheets 설정
GOOGLE_CREDENTIALS_JSON=your_google_service_account_json_string
```

5. 데이터 파일 준비

다음 파일들이 프로젝트 디렉토리에 있어야 합니다:
- `카테고리목록.csv`: 상품 카테고리 목록
- `user_data.json`: 사용자 세션 데이터 (자동 생성)
- `order_data.json`: 주문 정보 저장소 (자동 생성)

## 세부 설정 가이드

### Facebook Messenger 설정

1. [Facebook 개발자 콘솔](https://developers.facebook.com/)에서 Facebook 앱 생성
2. 앱에 Messenger 제품 추가
3. 페이지 액세스 토큰 생성
4. 웹훅 URL 설정: `https://your-domain.com/webhook`
5. 웹훅 이벤트 설정: `messages`, `messaging_postbacks`
6. 앱 검토 및 승인 요청

### Google Sheets 연동 설정

1. Google Cloud Console에서 서비스 계정 생성
2. Google Sheets API 활성화
3. 서비스 계정 JSON 키 다운로드
4. 대상 스프레드시트를 서비스 계정 이메일과 공유
5. 코드에서 스프레드시트 URL 설정 (124번째 줄 근처)

### Milvus 데이터베이스 설정

Milvus 연결 매개변수 설정:
```python
connections.connect(
    alias="default",
    host="your_milvus_host",
    port="19530"
)
```

## 실행 방법

### 로컬 개발 환경

#### Python 직접 실행
```bash
python facebook_chatbot.py
```

#### Docker 실행
```bash
# Docker 이미지 빌드
docker build -t facebook-chatbot .

# Docker 컨테이너 실행
docker run -p 5051:5051 --env-file .env facebook-chatbot
```

서버가 `http://localhost:5051`에서 실행됩니다.

### CloudType 배포 (Docker 기반)

1. CloudType 계정 생성 및 로그인
2. GitHub 저장소와 연결
3. **배포 타입을 "Dockerfile"로 선택**
4. 환경변수를 CloudType 설정에 추가:
   - `OPENAI_API_KEY`
   - `VERIFY_TOKEN`
   - `PAGE_ACCESS_TOKEN`
   - `REDIS_URL`
   - `GOOGLE_CREDENTIALS_JSON`
5. 포트를 5051로 설정
6. Dockerfile 경로 확인 (루트 디렉토리)
7. 배포 실행

배포 후 제공되는 URL을 Facebook 웹훅 설정에 사용하세요.

### Docker 설정

프로젝트에 `Dockerfile`이 포함되어 있어야 합니다:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5051

CMD ["python", "facebook_chatbot.py"]
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 메인 페이지 (HTML) |
| `GET` | `/webhook` | Facebook 웹훅 인증 |
| `POST` | `/webhook` | Facebook 메시지 처리 |
| `POST` | `/chatmall` | AI 검색 테스트 (디버깅용) |
| `GET` | `/debug-status` | 시스템 상태 조회 |
| `GET` | `/preview` | 상품 상세 페이지 미리보기 |

## 프로젝트 구조

```
facebook_chatbot/
├── facebook_chatbot.py          # 메인 애플리케이션
├── Dockerfile                   # Docker 설정 파일
├── requirements.txt             # Python 패키지 의존성
├── .env                        # 환경변수 (생성 필요)
├── 카테고리목록.csv            # 상품 카테고리 데이터
├── user_data.json             # 사용자 임시 데이터 (자동 생성)
├── order_data.json            # 주문 정보 (자동 생성)
├── templates/                 # HTML 템플릿
│   └── index.html
└── README.md                  # 프로젝트 설명서
```

## 주요 기능 상세

### 1. AI 상품 검색
- 사용자 입력을 자연어로 분석
- 카테고리 자동 예측
- Milvus 벡터 검색으로 관련 상품 추출
- LLM을 통한 최종 상품 5개 선별

### 2. 주문 프로세스
- 상품 선택 → 옵션 선택 → 수량 입력 → 주문 확인
- 묶음 배송 자동 계산
- 배송 정보 수집 (이름, 주소, 연락처, 이메일)
- Google Sheets 자동 저장

### 3. 세션 관리
- Redis를 통한 대화 기록 유지
- 사용자별 임시 데이터 관리
- 중복 메시지 처리 방지

### 환경별 설정

#### 개발 환경
```python
API_URL = "http://localhost:5051"
```

#### CloudType 프로덕션 환경 (Docker 기반)
```python
API_URL = "https://port-0-chatmall2-mddsxz1wc930914e.sel5.cloudtype.app"
```

### Docker 환경변수

Docker 실행 시 환경변수를 전달하는 방법:

1. `.env` 파일 사용:
```bash
docker run -p 5051:5051 --env-file .env facebook-chatbot
```

2. 개별 환경변수 전달:
```bash
docker run -p 5051:5051 \
  -e OPENAI_API_KEY=your_key \
  -e VERIFY_TOKEN=your_token \
  facebook-chatbot
```

## 트러블슈팅

### 일반적인 문제

1. **Facebook 웹훅 인증 실패**
   - VERIFY_TOKEN이 올바른지 확인
   - HTTPS 연결 확인

2. **Google Sheets 연결 실패**
   - 서비스 계정 JSON 형식 확인
   - 스프레드시트 공유 권한 확인
   - 실패 시 자동으로 홈 버튼 제공

3. **Milvus 연결 오류**
   - 호스트 및 포트 설정 확인
   - 네트워크 연결 상태 점검

4. **Redis 연결 문제**
   - Redis 서버 실행 상태 확인
   - URL 형식 검증

### 디버깅 도구

- `/debug-status`: 시스템 전체 상태 확인
- `/chatmall`: AI 검색 기능 단독 테스트
- 콘솔 로그를 통한 상세 실행 과정 추적
