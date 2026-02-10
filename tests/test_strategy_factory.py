"""Tests for StrategyFactory behavior and hot-swap compatibility."""
from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

from bandit_api.strategies.factory import StrategyFactory
from bandit_core.state.memory import InMemoryStateStore


class FakeExperimentStore:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(experiment_id)

    def set_experiment(
        self,
        experiment_id: str,
        *,
        arm_ids: Any,
        strategy: Optional[str],
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data[experiment_id] = {
            "experiment_id": experiment_id,
            "arm_ids": arm_ids,
            "strategy": strategy,
            "strategy_params": strategy_params or {},
        }


class TestStrategyFactory(unittest.TestCase):
    def test_alias_resolution(self) -> None:
        store = InMemoryStateStore()
        exp_store = FakeExperimentStore()
        exp_store.set_experiment("exp_1", arm_ids=["A", "B"], strategy="UCB", strategy_params={})

        factory = StrategyFactory(
            state_store=store,
            experiment_store=exp_store,  # type: ignore[arg-type]
            default_strategy="THOMPSON",
        )
        strategy = factory.build_for_experiment("exp_1")
        self.assertEqual(strategy.name, "ucb1")

    def test_thompson_normalizes_to_consistent_strategy_name(self) -> None:
        store = InMemoryStateStore()
        exp_store = FakeExperimentStore()
        exp_store.set_experiment("exp_ts", arm_ids=["A", "B"], strategy="THOMPSON", strategy_params={})

        factory = StrategyFactory(
            state_store=store,
            experiment_store=exp_store,  # type: ignore[arg-type]
            default_strategy="UCB1",
        )
        strategy = factory.build_for_experiment("exp_ts")
        self.assertEqual(strategy.name, "thompson")

    def test_default_strategy_used_when_experiment_has_no_override(self) -> None:
        store = InMemoryStateStore()
        exp_store = FakeExperimentStore()
        exp_store.set_experiment("exp_2", arm_ids=["A", "B"], strategy=None, strategy_params={})

        factory = StrategyFactory(
            state_store=store,
            experiment_store=exp_store,  # type: ignore[arg-type]
            default_strategy="EPSILON_GREEDY",
        )
        strategy = factory.build_for_experiment("exp_2")
        self.assertEqual(strategy.name, "epsilon_greedy")

    def test_hot_swap_does_not_break_select(self) -> None:
        store = InMemoryStateStore()
        exp_store = FakeExperimentStore()
        exp_store.set_experiment("exp_3", arm_ids=["A", "B"], strategy="UCB1", strategy_params={})

        factory = StrategyFactory(
            state_store=store,
            experiment_store=exp_store,  # type: ignore[arg-type]
            default_strategy="UCB1",
        )

        # First run with UCB1 state fields.
        strategy = factory.build_for_experiment("exp_3")
        strategy.initialize_experiment("exp_3", ["A", "B"])
        arm = strategy.select_arm("exp_3", ["A", "B"])
        strategy.update("exp_3", arm, 1.0)

        # Hot-swap to Thompson; select should still work with mixed state keys.
        exp_store.set_experiment(
            "exp_3",
            arm_ids=["A", "B"],
            strategy="THOMPSON",
            strategy_params={},
        )
        swapped = factory.build_for_experiment("exp_3")
        chosen = swapped.select_arm("exp_3", ["A", "B"])
        self.assertIn(chosen, {"A", "B"})

    def test_validate_strategy_name_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            StrategyFactory.validate_strategy_name("NOT_A_STRATEGY")


if __name__ == "__main__":
    unittest.main()

