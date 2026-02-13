"""Factory for selecting and constructing bandit strategy instances."""

from __future__ import annotations

from typing import Any, Protocol

from bandit_core.state.base import BanditStateStore
from bandit_core.strategies.base import BaseBanditStrategy
from bandit_core.strategies.epsilon_greedy import EpsilonGreedyStrategy
from bandit_core.strategies.thompson import ThompsonSamplingStrategy
from bandit_core.strategies.ucb1 import UCB1Strategy


class ExperimentLookup(Protocol):
    """Minimal contract needed by StrategyFactory."""

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None: ...


class StrategyFactory:
    """Resolve a strategy name and build a matching strategy object."""

    SUPPORTED_STRATEGIES = {"thompson", "epsilon_greedy", "ucb1"}

    def __init__(
        self,
        state_store: BanditStateStore,
        experiment_store: ExperimentLookup,
        default_strategy: str,
    ) -> None:
        self.state_store = state_store
        self.experiment_store = experiment_store
        self.default_strategy = default_strategy

    @staticmethod
    def normalize_strategy_name(name: str) -> str:
        val = name.strip().lower().replace("-", "_")
        aliases = {
            "ts": "thompson",
            "thompson_sampling": "thompson",
            "epsilon": "epsilon_greedy",
            "eps_greedy": "epsilon_greedy",
            "epsilon-greedy": "epsilon_greedy",
            "ucb": "ucb1",
        }
        return aliases.get(val, val)

    @classmethod
    def validate_strategy_name(cls, name: str) -> str:
        normalized = cls.normalize_strategy_name(name)
        if normalized not in cls.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{name}'. Valid values: THOMPSON, EPSILON_GREEDY, UCB1."
            )
        return normalized

    def build_for_experiment(self, experiment_id: str) -> BaseBanditStrategy:
        config = self.experiment_store.get_experiment(experiment_id) or {}
        configured_strategy = config.get("strategy")
        params: dict[str, Any] = dict(config.get("strategy_params") or {})

        strategy_name = configured_strategy or self.default_strategy
        normalized = self.validate_strategy_name(strategy_name)

        if normalized == "thompson":
            return ThompsonSamplingStrategy(self.state_store, **params)
        if normalized == "epsilon_greedy":
            return EpsilonGreedyStrategy(self.state_store, **params)
        if normalized == "ucb1":
            return UCB1Strategy(self.state_store, **params)

        # Kept as a safeguard if class mapping and SUPPORTED_STRATEGIES diverge.
        raise ValueError(
            f"Unsupported strategy '{strategy_name}'. Valid values: THOMPSON, EPSILON_GREEDY, UCB1."
        )
