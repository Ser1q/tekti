FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY evals/ ./evals/

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-sync", "tekti"]
CMD ["--help"]
