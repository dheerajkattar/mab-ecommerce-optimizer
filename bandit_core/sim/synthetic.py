"""Synthetic Bernoulli environment for benchmarking bandit strategies."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from bandit_core.state.memory import InMemoryStateStore
from bandit_core.strategies.base import BaseBanditStrategy


class BernoulliBanditEnv:
    """Simulates a set of arms with fixed (but hidden) conversion rates.

    Parameters
    ----------
    arm_rates : dict
        Mapping of ``{arm_id: true_conversion_rate}``.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        arm_rates: Dict[str, float],
        seed: Optional[int] = None,
    ) -> None:
        self.arm_rates = arm_rates
        self.arm_ids = list(arm_rates.keys())
        self.best_rate = max(arm_rates.values())
        self._rng = np.random.default_rng(seed)

    def pull(self, arm_id: str) -> float:
        """Draw a Bernoulli reward (0.0 or 1.0) for the given arm."""
        return 1.0 if self._rng.random() < self.arm_rates[arm_id] else 0.0

    def regret(self, arm_id: str) -> float:
        """Instantaneous regret from choosing *arm_id* instead of the best."""
        return self.best_rate - self.arm_rates[arm_id]


def run_simulation(
    strategy: BaseBanditStrategy,
    env: BernoulliBanditEnv,
    experiment_id: str,
    n_rounds: int,
) -> Tuple[List[float], List[float]]:
    """Run a strategy against a synthetic environment.

    Returns
    -------
    cumulative_regrets : list[float]
        Cumulative regret after each round.
    cumulative_rewards : list[float]
        Cumulative reward after each round.
    """
    strategy.initialize_experiment(experiment_id, env.arm_ids)

    cumulative_regret = 0.0
    cumulative_reward = 0.0
    regrets: List[float] = []
    rewards: List[float] = []

    for _ in range(n_rounds):
        arm_id = strategy.select_arm(experiment_id, env.arm_ids)
        reward = env.pull(arm_id)
        strategy.update(experiment_id, arm_id, reward)

        cumulative_regret += env.regret(arm_id)
        cumulative_reward += reward
        regrets.append(cumulative_regret)
        rewards.append(cumulative_reward)

    return regrets, rewards
