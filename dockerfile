# ===== CloudType 호환 Dockerfile =====
FROM python:3.8.20-slim

# 환경변수
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 사용자 생성 (CloudType 스타일)
RUN groupadd -r python && useradd --no-log-init -r -g python python

# Redis 설치
RUN apt-get update && \
    apt-get install -y redis-server redis-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 소유권 설정
RUN chown python:python /app

# 사용자 전환
USER python

# 의존성 복사 및 설치
COPY --chown=python:python requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY --chown=python:python . ./

# 환경변수
ENV PATH="/home/python/.local/bin:$PATH"
ENV REDIS_URL=redis://localhost:6379/0
ENV PORT=5051

# 포트 노출
EXPOSE 5051

# 루트 권한으로 전환 (Redis 실행용)
USER root

# 시작 스크립트
CMD ["sh", "-c", "redis-server --daemonize yes --port 6379 && sleep 5 && su python -c 'python facebook_chatbot.py'"]
