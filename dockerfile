# 1. 베이스 이미지 (원하는 Python 버전으로 맞추기)
FROM python:3.8.20

# 2. 작업 디렉토리 생성 및 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (Ubuntu 패키지 예시 + redis-server 설치)
RUN apt-get update && \
    apt-get install -y redis-server redis-tools && \
    rm -rf /var/lib/apt/lists/*

# 4. Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. .env 파일 복사 (load_dotenv() 쓸 경우 필요)
# COPY .env .

# 6. 애플리케이션 소스 전체 복사
COPY . .

# 7. 환경변수 설정
ENV REDIS_URL=redis://localhost:6379/0
ENV PORT=5051

# 8. 포트 노출
EXPOSE 5051

# 9. 컨테이너 시작 시 Redis 백그라운드 실행 + Python 실행
#    수정된 부분: BeeMall_Chatbot → facebook_chatbot, 포트 8011 → 5051
CMD ["sh", "-c", "redis-server --daemonize yes && sleep 3 && python facebook_chatbot.py"]
