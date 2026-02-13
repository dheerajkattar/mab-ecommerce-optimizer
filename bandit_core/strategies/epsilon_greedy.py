"""Epsilon-Greedy strategy with configurable exploration rate."""

from __future__ import annotations

from typing import Any

import numpy as np

from bandit_core.state.base import BanditStateStore
from bandit_core.strategies.base import BaseBanditStrategy


class EpsilonGreedyStrategy(BaseBanditStrategy):
    """Classic Epsilon-Greedy.

    With probability ``epsilon`` a random arm is chosen (explore); otherwise
    the arm with the highest observed mean reward is selected (exploit).
    """

    name = "epsilon_greedy"

    def __init__(
        self,
        store: BanditStateStore,
        *,
        epsilon: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(store, **kwargs)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon
        self._rng = np.random.default_rng(kwargs.get("seed"))

    # -- arm state layout -----------------------------------------------------

    def default_arm_state(self) -> dict[str, float]:
        return {"count": 0.0, "value_sum": 0.0}

    # -- core API -------------------------------------------------------------

    def select_arm(
        self,
        experiment_id: str,
        arm_ids: list[str],
        user_context: dict[str, Any] | None = None,
    ) -> str:
        # Explore
        if self._rng.random() < self.epsilon:
            return str(self._rng.choice(arm_ids))

        # Exploit – pick the arm with the highest empirical mean
        states = self.store.get_experiment_state(experiment_id, arm_ids)

        best_arm: str | None = None
        best_mean = -1.0

        for arm_id in arm_ids:
            s = states.get(arm_id, self.default_arm_state())
            count = s.get("count", 0.0)
            mean = s.get("value_sum", 0.0) / count if count > 0 else 0.0
            if mean > best_mean:
                best_mean = mean
                best_arm = arm_id

        # If all arms are at 0 pulls, pick randomly
        if best_arm is None:
            return str(self._rng.choice(arm_ids))
        return best_arm

    def update(
        self,
        experiment_id: str,
        arm_id: str,
        reward: float,
    ) -> None:
        self.store.increment(experiment_id, arm_id, "count", 1.0)
        self.store.increment(experiment_id, arm_id, "value_sum", reward)
