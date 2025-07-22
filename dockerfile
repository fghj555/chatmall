# 1. 베이스 이미지 (Python 3.8.20)
FROM python:3.8.20

# 2. 작업 디렉토리 생성 및 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (Redis 포함)
RUN apt-get update && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/*

# 4. Redis 설정 (비밀번호 추가)
RUN echo "requirepass chatmall2025" >> /etc/redis/redis.conf

# 5. Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. .env 파일 복사 (있다면)
# COPY .env .

# 7. 애플리케이션 소스 전체 복사
COPY . .

# 8. 환경변수 설정 (Redis URL)
ENV REDIS_URL=redis://:chatmall2025@localhost:6379/0
ENV PORT=5051

# 9. 포트 노출 (CloudType 포트에 맞춤)
EXPOSE 5051

# 10. 컨테이너 시작 시 Redis + FastAPI 실행
# CMD ["sh", "-c", "redis-server --daemonize yes && uvicorn BeeMall_Chatbot:app --host 0.0.0.0 --port 8011"]
CMD ["sh", "-c", "redis-server /etc/redis/redis.conf --daemonize yes && sleep 2 && python facebook_chatbot.py"]