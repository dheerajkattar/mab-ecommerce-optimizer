"""Fast, dict-backed state store for local benchmarking and tests."""

from __future__ import annotations

from collections import defaultdict

from bandit_core.state.base import BanditStateStore


class InMemoryStateStore(BanditStateStore):
    """Thread-unsafe, zero-dependency in-memory store.

    Useful for:
    * unit tests (no Redis needed)
    * the ``benchmark_strategies.py`` simulation script
    """

    def __init__(self) -> None:
        # {experiment_id: {arm_id: {key: float}}}
        self._data: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    # ---- read ---------------------------------------------------------------

    def get_arm_state(self, experiment_id: str, arm_id: str) -> dict[str, float]:
        return dict(self._data[experiment_id].get(arm_id, {}))

    def get_experiment_state(
        self, experiment_id: str, arm_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        exp = self._data[experiment_id]
        return {aid: dict(exp.get(aid, {})) for aid in arm_ids}

    # ---- write --------------------------------------------------------------

    def set_arm_state(self, experiment_id: str, arm_id: str, state: dict[str, float]) -> None:
        self._data[experiment_id][arm_id] = dict(state)

    def increment(self, experiment_id: str, arm_id: str, key: str, amount: float = 1.0) -> float:
        arm = self._data[experiment_id].setdefault(arm_id, {})
        arm[key] = arm.get(key, 0.0) + amount
        return arm[key]

    # ---- lifecycle ----------------------------------------------------------

    def initialize_arm(
        self,
        experiment_id: str,
        arm_id: str,
        default_state: dict[str, float],
    ) -> None:
        if arm_id not in self._data[experiment_id]:
            self._data[experiment_id][arm_id] = dict(default_state)

    def reset_experiment(self, experiment_id: str) -> None:
        self._data.pop(experiment_id, None)
