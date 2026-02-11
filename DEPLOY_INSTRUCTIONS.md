# Deployment Instructions — Hybrid Model

**Architecture:** Render (API) + Streamlit Community Cloud (Dashboard)

The API runs on Render's free tier (sleeps after 15 min of inactivity).
The dashboard runs on Streamlit Community Cloud (always on, no cold starts).
When a visitor opens the dashboard, it automatically wakes the API.

---

## 1. Upstash Redis

1. Create a free Redis database at [console.upstash.com](https://console.upstash.com)
2. Copy the `rediss://...` connection URL (TLS-enabled)
3. You will use this URL as `REDIS_URL` in both Render and Streamlit Cloud

---

## 2. Deploy the API on Render

1. Go to **Render.com → Blueprints → New Blueprint Instance**
2. Connect your GitHub repo — Render reads `render.yaml` automatically
3. Set the environment variable when prompted:
   - `REDIS_URL` = your Upstash `rediss://...` URL
4. Deploy and note the service URL (e.g. `https://mab-api.onrender.com`)

---

## 3. Deploy the Dashboard on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect your GitHub repo
3. Set:
   - **Main file path:** `dashboard/app.py`
   - **Python version:** 3.11
4. Under **Advanced settings → Secrets** (or Environment variables), add:
   - `API_URL` = your Render API URL (e.g. `https://mab-api.onrender.com`)
   - `REDIS_URL` = your Upstash `rediss://...` URL
5. Deploy

Streamlit Cloud reads `requirements.txt` at the repo root to install dependencies.

---

## 4. Verify

| Check | URL | Expected |
|-------|-----|----------|
| API health | `https://<mab-api>.onrender.com/health` | `{"status": "ok"}` |
| API docs | `https://<mab-api>.onrender.com/docs` | Swagger UI |
| Dashboard | `https://<app>.streamlit.app` | Spinner → live data |
| Simulator tab | Same dashboard, second tab | Runs locally, no API needed |

---

## Environment Variables Summary

| Variable | Render (API) | Streamlit Cloud (Dashboard) |
|----------|-------------|----------------------------|
| `REDIS_URL` | Required | Required |
| `API_URL` | — | Required (Render API URL) |
| `ACTIVE_STRATEGY` | Optional (`THOMPSON`) | — |
| `LOG_LEVEL` | Optional (`INFO`) | — |
| `LIVE_REFRESH_SECONDS` | — | Optional (default `3`) |
