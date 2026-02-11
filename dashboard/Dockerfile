FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install package + dashboard dependencies.
COPY pyproject.toml README.md ./
COPY bandit_core ./bandit_core
COPY bandit_api ./bandit_api
COPY dashboard ./dashboard
RUN pip install --no-cache-dir -e ".[dashboard,api]"

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
