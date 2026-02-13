"""Redis key naming helpers."""

from __future__ import annotations


def experiments_index_key() -> str:
    return "bandit:experiments"


def experiment_meta_key(experiment_id: str) -> str:
    return f"bandit:experiment:{experiment_id}:meta"


def arm_state_key(experiment_id: str, arm_id: str) -> str:
    return f"bandit:experiment:{experiment_id}:arm:{arm_id}"


def experiment_arm_pattern(experiment_id: str) -> str:
    return f"bandit:experiment:{experiment_id}:arm:*"
