FROM python:3.14-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Sync API deps - no torch yet
RUN uv sync --frozen --only-group docker --no-install-project

# Install CPU torch directly, overriding any CUDA version
RUN uv pip install torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --reinstall

# copy pii redactor and api to workdir
COPY pii_redaction/ pii_redaction/
COPY api/ api/                   

# Set environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# add --no-sync so uv doesn't install default project dependencies
CMD ["uv", "run", "--no-sync", "python", "-m", "api.app.main"]