"""Phase 1 verification tests for core strategies and simulation."""

from __future__ import annotations

import unittest

from bandit_core.sim.synthetic import BernoulliBanditEnv, run_simulation
from bandit_core.state.memory import InMemoryStateStore
from bandit_core.strategies.epsilon_greedy import EpsilonGreedyStrategy
from bandit_core.strategies.thompson import ThompsonSamplingStrategy
from bandit_core.strategies.ucb1 import UCB1Strategy
from benchmark_strategies import DEFAULT_ARM_RATES, build_strategies


class TestPhase1Core(unittest.TestCase):
    def test_thompson_update_changes_beta_posterior(self) -> None:
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store, seed=1)
        experiment_id = "exp_ts"
        arm_id = "A"
        strategy.initialize_experiment(experiment_id, [arm_id])

        strategy.update(experiment_id, arm_id, reward=1.0)
        strategy.update(experiment_id, arm_id, reward=0.0)

        state = store.get_arm_state(experiment_id, arm_id)
        self.assertEqual(state["alpha"], 2.0)
        self.assertEqual(state["beta"], 2.0)

    def test_thompson_update_supports_fractional_rewards(self) -> None:
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store, seed=1)
        experiment_id = "exp_ts_fractional"
        arm_id = "A"
        strategy.initialize_experiment(experiment_id, [arm_id])

        strategy.update(experiment_id, arm_id, reward=0.5)
        state = store.get_arm_state(experiment_id, arm_id)
        self.assertEqual(state["alpha"], 1.5)
        self.assertEqual(state["beta"], 1.5)

    def test_epsilon_greedy_exploit_prefers_better_empirical_arm(self) -> None:
        store = InMemoryStateStore()
        strategy = EpsilonGreedyStrategy(store=store, epsilon=0.0, seed=7)
        experiment_id = "exp_eps"
        arm_ids = ["A", "B"]
        strategy.initialize_experiment(experiment_id, arm_ids)

        # Arm A: mean 0.25
        for reward in [1.0, 0.0, 0.0, 0.0]:
            strategy.update(experiment_id, "A", reward)
        # Arm B: mean 0.75
        for reward in [1.0, 1.0, 1.0, 0.0]:
            strategy.update(experiment_id, "B", reward)

        selected = strategy.select_arm(experiment_id, arm_ids)
        self.assertEqual(selected, "B")

    def test_ucb1_tries_each_arm_before_confidence_scoring(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store, seed=9)
        experiment_id = "exp_ucb"
        arm_ids = ["A", "B", "C"]
        strategy.initialize_experiment(experiment_id, arm_ids)

        seen = set()
        for _ in arm_ids:
            arm = strategy.select_arm(experiment_id, arm_ids)
            seen.add(arm)
            strategy.update(experiment_id, arm, reward=0.0)

        self.assertEqual(seen, set(arm_ids))

    def test_ucb1_skips_non_positive_counts_and_keeps_evaluating(self) -> None:
        class CorruptCountStore(InMemoryStateStore):
            def get_experiment_state(
                self, experiment_id: str, arm_ids: list[str]
            ) -> dict[str, dict[str, float]]:
                return {
                    # Negative count bypasses the "unplayed == 0" branch and
                    # would raise math-domain errors without defensive guards.
                    "A": {"count": -1.0, "value_sum": 0.0},
                    "B": {"count": 5.0, "value_sum": 2.0},
                }

        strategy = UCB1Strategy(store=CorruptCountStore(), seed=9)
        chosen = strategy.select_arm("exp_corrupt", ["A", "B"])
        self.assertEqual(chosen, "B")

    def test_ucb1_falls_back_when_all_counts_invalid(self) -> None:
        class AllInvalidCountStore(InMemoryStateStore):
            def get_experiment_state(
                self, experiment_id: str, arm_ids: list[str]
            ) -> dict[str, dict[str, float]]:
                return {
                    "A": {"count": -1.0, "value_sum": 0.0},
                    "B": {"count": -3.0, "value_sum": 1.0},
                }

        strategy = UCB1Strategy(store=AllInvalidCountStore(), seed=9)
        chosen = strategy.select_arm("exp_all_invalid", ["A", "B"])
        self.assertIn(chosen, {"A", "B"})

    def test_simulation_outputs_have_expected_shape(self) -> None:
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store, seed=42)
        env = BernoulliBanditEnv({"A": 0.1, "B": 0.2}, seed=42)

        regrets, rewards = run_simulation(
            strategy=strategy,
            env=env,
            experiment_id="sim_shape",
            n_rounds=200,
        )
        self.assertEqual(len(regrets), 200)
        self.assertEqual(len(rewards), 200)
        self.assertTrue(all(r >= 0.0 for r in regrets))

    def test_benchmark_strategy_builder_returns_three_algorithms(self) -> None:
        built = build_strategies(seed=42)
        labels = [label for label, _ in built]
        self.assertEqual(len(labels), 3)
        self.assertIn("Thompson Sampling", labels)
        self.assertIn("Epsilon-Greedy (eps=0.1)", labels)
        self.assertIn("UCB1", labels)
        self.assertEqual(set(DEFAULT_ARM_RATES.keys()), {"A", "B", "C", "D"})


if __name__ == "__main__":
    unittest.main()
