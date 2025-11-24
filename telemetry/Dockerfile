FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy the necessary files, including the telemetry module
COPY pyproject.toml uv.lock ./
COPY data_ingress.py ./

# Use uv to sync dependencies (no dev deps for production)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Set PYTHONPATH to include /app directory to ensure telemetry module is detected
ENV PYTHONPATH=/app

CMD ["uv", "run", "python3", "data_ingress.py"]