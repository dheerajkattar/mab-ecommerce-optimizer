"""Redis-backed stores for arm state and experiment metadata."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import redis

from bandit_api.state.keys import (
    arm_state_key,
    experiment_arm_pattern,
    experiment_meta_key,
    experiments_index_key,
)
from bandit_core.state.base import BanditStateStore


class RedisBanditStateStore(BanditStateStore):
    """BanditStateStore implementation using Redis hashes."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis = redis_client

    def get_arm_state(self, experiment_id: str, arm_id: str) -> Dict[str, float]:
        raw = self.redis.hgetall(arm_state_key(experiment_id, arm_id))
        if not raw:
            return {}
        return {k.decode("utf-8"): float(v) for k, v in raw.items()}

    def get_experiment_state(
        self, experiment_id: str, arm_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        states: Dict[str, Dict[str, float]] = {}
        for arm_id in arm_ids:
            states[arm_id] = self.get_arm_state(experiment_id, arm_id)
        return states

    def set_arm_state(
        self, experiment_id: str, arm_id: str, state: Dict[str, float]
    ) -> None:
        key = arm_state_key(experiment_id, arm_id)
        self.redis.delete(key)
        if state:
            self.redis.hset(key, mapping={k: str(v) for k, v in state.items()})

    def increment(
        self, experiment_id: str, arm_id: str, key: str, amount: float = 1.0
    ) -> float:
        val = self.redis.hincrbyfloat(arm_state_key(experiment_id, arm_id), key, amount)
        return float(val)

    def initialize_arm(
        self,
        experiment_id: str,
        arm_id: str,
        default_state: Dict[str, float],
    ) -> None:
        key = arm_state_key(experiment_id, arm_id)
        if self.redis.exists(key):
            return
        if default_state:
            self.redis.hset(key, mapping={k: str(v) for k, v in default_state.items()})
        else:
            self.redis.hset(key, mapping={})

    def reset_experiment(self, experiment_id: str) -> None:
        for key in self.redis.scan_iter(match=experiment_arm_pattern(experiment_id)):
            self.redis.delete(key)


class RedisExperimentStore:
    """Stores experiment definitions and per-experiment strategy config."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis = redis_client

    def create_experiment(
        self,
        experiment_id: str,
        arm_ids: List[str],
        strategy: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = experiment_meta_key(experiment_id)
        meta = {
            "arm_ids": json.dumps(arm_ids),
            "strategy": strategy or "",
            "strategy_params": json.dumps(strategy_params or {}),
        }
        self.redis.hset(key, mapping=meta)
        self.redis.sadd(experiments_index_key(), experiment_id)
        return self.get_experiment(experiment_id)  # type: ignore[return-value]

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        raw = self.redis.hgetall(experiment_meta_key(experiment_id))
        if not raw:
            return None

        decoded = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}
        return {
            "experiment_id": experiment_id,
            "arm_ids": json.loads(decoded.get("arm_ids", "[]")),
            "strategy": decoded.get("strategy") or None,
            "strategy_params": json.loads(decoded.get("strategy_params", "{}")),
        }

    def add_arms(self, experiment_id: str, arm_ids: List[str]) -> Optional[Dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return None

        current = experiment["arm_ids"]
        merged = sorted(set(current + arm_ids))
        self.redis.hset(experiment_meta_key(experiment_id), "arm_ids", json.dumps(merged))
        return self.get_experiment(experiment_id)

    def set_strategy_config(
        self,
        experiment_id: str,
        strategy: str,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        key = experiment_meta_key(experiment_id)
        if not self.redis.exists(key):
            return None
        self.redis.hset(
            key,
            mapping={
                "strategy": strategy,
                "strategy_params": json.dumps(strategy_params or {}),
            },
        )
        return self.get_experiment(experiment_id)

