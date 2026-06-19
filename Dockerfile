FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ agent/
COPY static/ static/
COPY api.py .

RUN pip install --no-cache-dir -e .

ENV PORT=8000

EXPOSE 8000

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
