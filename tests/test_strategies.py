"""Targeted tests for strategy edge-cases not covered by test_phase1_core."""

from __future__ import annotations

import unittest

from bandit_core.state.memory import InMemoryStateStore
from bandit_core.strategies.epsilon_greedy import EpsilonGreedyStrategy
from bandit_core.strategies.thompson import ThompsonSamplingStrategy
from bandit_core.strategies.ucb1 import UCB1Strategy

# ---- Thompson Sampling ------------------------------------------------------


class TestThompsonSampling(unittest.TestCase):
    def test_select_arm_returns_valid_arm(self) -> None:
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store, seed=42)
        arm_ids = ["A", "B", "C"]
        strategy.initialize_experiment("exp", arm_ids)
        chosen = strategy.select_arm("exp", arm_ids)
        self.assertIn(chosen, set(arm_ids))

    def test_select_arm_uses_default_state_for_uninitialized_arms(self) -> None:
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store, seed=7)
        # Don't call initialize_experiment; arms have no stored state
        chosen = strategy.select_arm("exp", ["X", "Y"])
        self.assertIn(chosen, {"X", "Y"})

    def test_default_arm_state(self) -> None:
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store)
        self.assertEqual(strategy.default_arm_state(), {"alpha": 1.0, "beta": 1.0})

    def test_update_reward_zero_increments_beta_only(self) -> None:
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store, seed=1)
        strategy.initialize_experiment("exp", ["A"])
        strategy.update("exp", "A", 0.0)
        state = store.get_arm_state("exp", "A")
        self.assertEqual(state["alpha"], 1.0)  # unchanged from default
        self.assertEqual(state["beta"], 2.0)  # 1.0 + 1.0

    def test_converges_to_better_arm(self) -> None:
        """After enough updates, Thompson should mostly pick the better arm."""
        store = InMemoryStateStore()
        strategy = ThompsonSamplingStrategy(store=store, seed=42)
        strategy.initialize_experiment("exp", ["good", "bad"])
        # Give "good" arm a strong posterior
        for _ in range(50):
            strategy.update("exp", "good", 1.0)
        for _ in range(50):
            strategy.update("exp", "bad", 0.0)
        # Over 100 selections, the better arm should dominate
        picks = [strategy.select_arm("exp", ["good", "bad"]) for _ in range(100)]
        self.assertGreater(picks.count("good"), 80)


# ---- Epsilon-Greedy ----------------------------------------------------------


class TestEpsilonGreedy(unittest.TestCase):
    def test_epsilon_one_always_explores(self) -> None:
        store = InMemoryStateStore()
        strategy = EpsilonGreedyStrategy(store=store, epsilon=1.0, seed=42)
        strategy.initialize_experiment("exp", ["A", "B"])
        # Give A a clearly better mean
        for _ in range(10):
            strategy.update("exp", "A", 1.0)
        for _ in range(10):
            strategy.update("exp", "B", 0.0)
        # With eps=1.0, all selections are random, so both arms should appear
        picks = {strategy.select_arm("exp", ["A", "B"]) for _ in range(50)}
        self.assertEqual(picks, {"A", "B"})

    def test_invalid_epsilon_raises(self) -> None:
        store = InMemoryStateStore()
        with self.assertRaises(ValueError):
            EpsilonGreedyStrategy(store=store, epsilon=-0.1)
        with self.assertRaises(ValueError):
            EpsilonGreedyStrategy(store=store, epsilon=1.5)

    def test_update_increments_count_and_value_sum(self) -> None:
        store = InMemoryStateStore()
        strategy = EpsilonGreedyStrategy(store=store, epsilon=0.0, seed=1)
        strategy.initialize_experiment("exp", ["A"])
        strategy.update("exp", "A", 0.7)
        state = store.get_arm_state("exp", "A")
        self.assertEqual(state["count"], 1.0)
        self.assertAlmostEqual(state["value_sum"], 0.7)

    def test_zero_pull_arms_get_random_selection(self) -> None:
        store = InMemoryStateStore()
        strategy = EpsilonGreedyStrategy(store=store, epsilon=0.0, seed=42)
        strategy.initialize_experiment("exp", ["A", "B", "C"])
        # No updates, all arms at 0 pulls: should still return a valid arm
        chosen = strategy.select_arm("exp", ["A", "B", "C"])
        self.assertIn(chosen, {"A", "B", "C"})

    def test_default_arm_state(self) -> None:
        store = InMemoryStateStore()
        strategy = EpsilonGreedyStrategy(store=store)
        self.assertEqual(strategy.default_arm_state(), {"count": 0.0, "value_sum": 0.0})


# ---- UCB1 --------------------------------------------------------------------


class TestUCB1(unittest.TestCase):
    def test_exploitation_prefers_high_mean_arm(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store, seed=42)
        strategy.initialize_experiment("exp", ["A", "B"])
        # Both arms played enough to exit Phase 1, arm A has higher mean
        for _ in range(20):
            strategy.update("exp", "A", 1.0)
        for _ in range(20):
            strategy.update("exp", "B", 0.0)
        chosen = strategy.select_arm("exp", ["A", "B"])
        self.assertEqual(chosen, "A")

    def test_custom_exploration_weight(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store, exploration_weight=0.0, seed=42)
        strategy.initialize_experiment("exp", ["A", "B"])
        # With c=0, UCB reduces to pure exploitation (no confidence bonus)
        for _ in range(5):
            strategy.update("exp", "A", 1.0)
        for _ in range(5):
            strategy.update("exp", "B", 0.0)
        chosen = strategy.select_arm("exp", ["A", "B"])
        self.assertEqual(chosen, "A")

    def test_update_increments_count_and_value_sum(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store, seed=1)
        strategy.initialize_experiment("exp", ["A"])
        strategy.update("exp", "A", 0.5)
        state = store.get_arm_state("exp", "A")
        self.assertEqual(state["count"], 1.0)
        self.assertAlmostEqual(state["value_sum"], 0.5)

    def test_default_arm_state(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store)
        self.assertEqual(strategy.default_arm_state(), {"count": 0.0, "value_sum": 0.0})

    def test_name(self) -> None:
        store = InMemoryStateStore()
        self.assertEqual(UCB1Strategy(store=store).name, "ucb1")
        self.assertEqual(EpsilonGreedyStrategy(store=store).name, "epsilon_greedy")
        self.assertEqual(ThompsonSamplingStrategy(store=store).name, "thompson")


if __name__ == "__main__":
    unittest.main()
