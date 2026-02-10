# Multi-Armed Bandit Framework

Production-ready, containerized Python application for experimenting with pluggable Multi-Armed Bandit strategies.

## What’s included (planned)
- **Core strategies**: Thompson Sampling, Epsilon-Greedy, UCB1
- **API**: FastAPI service for `/decision` + `/reward`, backed by Redis
- **Dashboard**: Streamlit app for live stats + simulator playground
- **Local stack**: Docker + Docker Compose (API + Redis + Dashboard)
- **AWS**: SAM template for Lambda (container image) + ElastiCache Redis

## Local development (dependencies)
This repo uses a `pyproject.toml` with optional dependency groups.

Install everything for local dev:

```bash
python -m pip install -U pip
pip install -e ".[api,dashboard,bench,dev]"
```

## Environment variables
See `.env.example`.

## API quick start (Phase 2)
Install API deps, then run:

```bash
pip install -e ".[api]"
uvicorn bandit_api.main:app --reload
```

Main endpoints:
- `POST /experiments`
- `GET /experiments/{experiment_id}`
- `POST /experiments/{experiment_id}/arms`
- `GET /decision?experiment_id=...`
- `POST /reward`
- `POST /config` (hot-swap strategy / tune params)

## Docker Compose (Phase 3)
Bring up API + Redis + Dashboard:

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000` (`/health`)
- Dashboard: `http://localhost:8501`
- Redis: `localhost:6379`

