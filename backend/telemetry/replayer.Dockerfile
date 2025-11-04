FROM python:3.12-slim

# Keep things speedy and quiet
ENV UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Install uv (single static binary)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Python dependencies via uv (system site-packages to keep image small)
RUN uv pip install --system --no-cache \
      confluent-kafka sqlalchemy psycopg2-binary python-dotenv

# Your code
COPY telemetry /app/telemetry

CMD ["python", "-u", "telemetry/replayer/replayer.py"]
