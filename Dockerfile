# GateOrchestra API (FastAPI + the chat assistant).
# Build:  docker build -t gateorchestra-api .
# Run:    docker run --env-file .env -p 3700:3700 gateorchestra-api
# Secrets come from the environment at run time; none are baked into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY shared ./shared
COPY gate ./gate
COPY integration ./integration
COPY agents ./agents
COPY dataset ./dataset
COPY evaluation ./evaluation
COPY assistant ./assistant
COPY api ./api
COPY scripts ./scripts
COPY configs ./configs
COPY results ./results

RUN pip install . && python -m spacy download en_core_web_sm \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/logs && chown -R app /app/logs

USER app
EXPOSE 3700
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3700/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3700", "--proxy-headers", "--forwarded-allow-ips", "*"]
