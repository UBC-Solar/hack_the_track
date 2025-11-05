FROM python:3.12-slim

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system --no-cache \
      confluent-kafka psycopg2-binary

COPY telemetry/aligned/tick_consumer.py /app/tick_consumer.py

CMD ["uv", "run", "python3", "tick_consumer.py"]