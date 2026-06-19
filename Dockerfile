FROM python:3.11-slim

# System dependencies for duckduckgo-search
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY agent/ agent/
COPY tests/ tests/

# Install the package in editable mode so `constrained-agent` CLI is available
RUN pip install --no-cache-dir -e .

# Default: show help. Override with a task string.
ENTRYPOINT ["constrained-agent"]
CMD ["--help"]
