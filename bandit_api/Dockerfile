FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install package + API dependencies.
COPY pyproject.toml README.md ./
COPY bandit_core ./bandit_core
COPY bandit_api ./bandit_api
RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000

CMD ["uvicorn", "bandit_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
