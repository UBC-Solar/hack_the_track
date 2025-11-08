# Use Astral's uv image with Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Install dependencies first for caching
COPY pyproject.toml uv.lock ./

COPY load_gps_data.py main.py telemetry/ ./

# Use uv to sync dependencies (no dev deps for production)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Expose port and run
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
