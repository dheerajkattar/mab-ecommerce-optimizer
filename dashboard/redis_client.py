"""Redis helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from typing import Any

import redis

from bandit_api.state.keys import arm_state_key, experiment_meta_key, experiments_index_key


class DashboardRedisClient:
    """Small helper wrapper for reading bandit metadata and arm stats."""

    def __init__(self, redis_url: str) -> None:
        self._client = redis.from_url(redis_url)

    def ping(self) -> bool:
        return bool(self._client.ping())

    def list_experiment_ids(self) -> list[str]:
        raw = self._client.smembers(experiments_index_key())
        return sorted([v.decode("utf-8") for v in raw])

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        raw = self._client.hgetall(experiment_meta_key(experiment_id))
        if not raw:
            return None
        decoded = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}
        return {
            "experiment_id": experiment_id,
            "arm_ids": json.loads(decoded.get("arm_ids", "[]")),
            "strategy": decoded.get("strategy") or None,
            "strategy_params": json.loads(decoded.get("strategy_params", "{}")),
        }

    def get_arm_state(self, experiment_id: str, arm_id: str) -> dict[str, float]:
        raw = self._client.hgetall(arm_state_key(experiment_id, arm_id))
        if not raw:
            return {}
        return {k.decode("utf-8"): float(v) for k, v in raw.items()}
