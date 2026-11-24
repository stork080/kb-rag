FROM python:3.11-slim

WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so first request is fast (cold start)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app.py .

EXPOSE 7860

# 2 workers = safe concurrency under the 8s budget
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2", "--timeout-keep-alive", "30"]
