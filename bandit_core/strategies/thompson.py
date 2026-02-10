"""Thompson Sampling for Bernoulli (Beta-distributed) rewards."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from bandit_core.state.base import BanditStateStore
from bandit_core.strategies.base import BaseBanditStrategy


class ThompsonSamplingStrategy(BaseBanditStrategy):
    """Beta–Bernoulli Thompson Sampling.

    Each arm maintains ``alpha`` (successes + 1) and ``beta`` (failures + 1)
    parameters for a Beta distribution.  At decision time a sample is drawn
    from each arm's posterior and the arm with the highest sample wins.
    """

    name = "thompson"

    def __init__(self, store: BanditStateStore, **kwargs: Any) -> None:
        super().__init__(store, **kwargs)
        self._rng = np.random.default_rng(kwargs.get("seed"))

    # -- arm state layout -----------------------------------------------------

    def default_arm_state(self) -> Dict[str, float]:
        return {"alpha": 1.0, "beta": 1.0}

    # -- core API -------------------------------------------------------------

    def select_arm(
        self,
        experiment_id: str,
        arm_ids: List[str],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        states = self.store.get_experiment_state(experiment_id, arm_ids)

        best_arm: Optional[str] = None
        best_sample = -1.0

        for arm_id in arm_ids:
            s = states.get(arm_id, self.default_arm_state())
            alpha = s.get("alpha", 1.0)
            beta = s.get("beta", 1.0)
            sample = float(self._rng.beta(alpha, beta))
            if sample > best_sample:
                best_sample = sample
                best_arm = arm_id

        assert best_arm is not None
        return best_arm

    def update(
        self,
        experiment_id: str,
        arm_id: str,
        reward: float,
    ) -> None:
        # Fractional rewards in [0, 1] map to proportional Beta updates.
        # e.g. reward=0.5 -> alpha += 0.5, beta += 0.5.
        self.store.increment(experiment_id, arm_id, "alpha", reward)
        self.store.increment(experiment_id, arm_id, "beta", 1.0 - reward)
