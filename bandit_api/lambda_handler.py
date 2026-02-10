"""AWS Lambda adapter for the FastAPI ASGI app."""
from __future__ import annotations

from mangum import Mangum

from bandit_api.main import app

# Lambda entrypoint for API Gateway HTTP API events.
handler = Mangum(app)

