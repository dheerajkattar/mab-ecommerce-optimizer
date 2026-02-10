"""UCB1 (Upper Confidence Bound) strategy."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from bandit_core.state.base import BanditStateStore
from bandit_core.strategies.base import BaseBanditStrategy


class UCB1Strategy(BaseBanditStrategy):
    r"""UCB1 algorithm (Auer et al., 2002).

    At each round *t*, the arm with the highest index is chosen:

    .. math::

        \text{UCB}_i = \bar{x}_i + c \sqrt{\frac{2 \ln t}{n_i}}

    where :math:`\bar{x}_i` is the empirical mean, :math:`n_i` is the pull
    count for arm *i*, *t* is the total number of pulls across all arms, and
    *c* is the ``exploration_weight`` (defaults to 1.0).
    """

    name = "ucb1"

    def __init__(
        self,
        store: BanditStateStore,
        *,
        exploration_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(store, **kwargs)
        self.c = exploration_weight
        self._rng = np.random.default_rng(kwargs.get("seed"))

    # -- arm state layout -----------------------------------------------------

    def default_arm_state(self) -> Dict[str, float]:
        return {"count": 0.0, "value_sum": 0.0}

    # -- core API -------------------------------------------------------------

    def select_arm(
        self,
        experiment_id: str,
        arm_ids: List[str],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        states = self.store.get_experiment_state(experiment_id, arm_ids)

        # Phase 1: play each arm at least once (in random order)
        unplayed = [
            aid for aid in arm_ids
            if states.get(aid, self.default_arm_state()).get("count", 0.0) == 0
        ]
        if unplayed:
            return str(self._rng.choice(unplayed))

        # Total pulls across all arms
        total_pulls = sum(
            states[aid].get("count", 0.0) for aid in arm_ids
        )
        log_total = math.log(total_pulls) if total_pulls > 0 else 0.0

        best_arm: Optional[str] = None
        best_ucb = -math.inf

        for arm_id in arm_ids:
            s = states[arm_id]
            n_i = s.get("count", 0.0)
            # Defensive fallback: skip corrupted/non-positive counts so we can
            # still evaluate remaining valid arms.
            if n_i <= 0:
                continue
            mean = s.get("value_sum", 0.0) / n_i
            bonus = self.c * math.sqrt(2.0 * log_total / n_i)
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_id

        # If all counts were invalid, pick an arm randomly as a safe fallback.
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
