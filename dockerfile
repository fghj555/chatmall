# ===== 쓰기 권한 해결 Dockerfile =====
FROM python:3.8.20

# 환경변수
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Redis 설치
RUN apt-get update && \
    apt-get install -y redis-server redis-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

# JSON 파일 생성 및 권한 설정
RUN touch user_data.json order_data.json && \
    chmod 777 user_data.json order_data.json && \
    chmod 777 /app

# 환경변수
ENV REDIS_URL=redis://localhost:6379/0
ENV PORT=5051

# 포트 노출
EXPOSE 5051

# 시작 명령
CMD ["sh", "-c", "redis-server --daemonize yes && sleep 3 && python facebook_chatbot.py"]
