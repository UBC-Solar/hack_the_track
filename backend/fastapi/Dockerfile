# Use Astral's uv image with Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Install dependencies first for caching
COPY pyproject.toml uv.lock ./

# Use uv to sync dependencies (no dev deps for production)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copy the rest of the app
COPY . .

# Ensure executables are available
ENV PATH="/app/.venv/bin:$PATH"

# Switch to a non-root user for security
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot
USER nonroot

# Expose port and run
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
