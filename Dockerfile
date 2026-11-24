FROM python:3.11-slim

WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

# Install Python deps first (this layer caches when requirements.txt doesn't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed ONNX model into the image so the first
# request at runtime is fast and doesn't need network access
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY app.py .

EXPOSE 7860

# 1 worker (not 2) — saves ~150 MB, and FastAPI sync routes run in a
# threadpool, so concurrency is still handled under the 8 s budget.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "30"]
