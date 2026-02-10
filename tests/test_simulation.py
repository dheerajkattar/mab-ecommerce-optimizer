"""Tests for the synthetic environment and simulation runner."""
from __future__ import annotations

import unittest

from bandit_core.sim.synthetic import BernoulliBanditEnv, run_simulation
from bandit_core.state.memory import InMemoryStateStore
from bandit_core.strategies.epsilon_greedy import EpsilonGreedyStrategy
from bandit_core.strategies.ucb1 import UCB1Strategy


class TestBernoulliBanditEnv(unittest.TestCase):
    def test_pull_returns_zero_or_one(self) -> None:
        env = BernoulliBanditEnv({"A": 0.5}, seed=42)
        results = {env.pull("A") for _ in range(100)}
        self.assertEqual(results, {0.0, 1.0})

    def test_pull_zero_rate_always_returns_zero(self) -> None:
        env = BernoulliBanditEnv({"A": 0.0}, seed=42)
        self.assertTrue(all(env.pull("A") == 0.0 for _ in range(50)))

    def test_pull_one_rate_always_returns_one(self) -> None:
        env = BernoulliBanditEnv({"A": 1.0}, seed=42)
        self.assertTrue(all(env.pull("A") == 1.0 for _ in range(50)))

    def test_best_rate(self) -> None:
        env = BernoulliBanditEnv({"A": 0.1, "B": 0.5, "C": 0.3})
        self.assertEqual(env.best_rate, 0.5)

    def test_regret_for_best_arm_is_zero(self) -> None:
        env = BernoulliBanditEnv({"A": 0.1, "B": 0.5})
        self.assertEqual(env.regret("B"), 0.0)

    def test_regret_for_suboptimal_arm(self) -> None:
        env = BernoulliBanditEnv({"A": 0.1, "B": 0.5})
        self.assertAlmostEqual(env.regret("A"), 0.4)

    def test_arm_ids(self) -> None:
        env = BernoulliBanditEnv({"X": 0.1, "Y": 0.2, "Z": 0.3})
        self.assertEqual(set(env.arm_ids), {"X", "Y", "Z"})

    def test_seed_produces_reproducible_results(self) -> None:
        env1 = BernoulliBanditEnv({"A": 0.5}, seed=99)
        env2 = BernoulliBanditEnv({"A": 0.5}, seed=99)
        results1 = [env1.pull("A") for _ in range(20)]
        results2 = [env2.pull("A") for _ in range(20)]
        self.assertEqual(results1, results2)


class TestRunSimulation(unittest.TestCase):
    def test_cumulative_regret_is_monotonically_non_decreasing(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store, seed=42)
        env = BernoulliBanditEnv({"A": 0.1, "B": 0.5}, seed=42)
        regrets, _ = run_simulation(strategy, env, "exp", n_rounds=200)
        for i in range(1, len(regrets)):
            self.assertGreaterEqual(regrets[i], regrets[i - 1])

    def test_cumulative_rewards_are_non_negative(self) -> None:
        store = InMemoryStateStore()
        strategy = EpsilonGreedyStrategy(store=store, epsilon=0.1, seed=42)
        env = BernoulliBanditEnv({"A": 0.3, "B": 0.7}, seed=42)
        _, rewards = run_simulation(strategy, env, "exp", n_rounds=100)
        self.assertTrue(all(r >= 0.0 for r in rewards))

    def test_rewards_are_monotonically_non_decreasing(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store, seed=1)
        env = BernoulliBanditEnv({"A": 0.2, "B": 0.8}, seed=1)
        _, rewards = run_simulation(strategy, env, "exp", n_rounds=100)
        for i in range(1, len(rewards)):
            self.assertGreaterEqual(rewards[i], rewards[i - 1])

    def test_zero_rounds(self) -> None:
        store = InMemoryStateStore()
        strategy = UCB1Strategy(store=store, seed=1)
        env = BernoulliBanditEnv({"A": 0.5}, seed=1)
        regrets, rewards = run_simulation(strategy, env, "exp", n_rounds=0)
        self.assertEqual(regrets, [])
        self.assertEqual(rewards, [])


if __name__ == "__main__":
    unittest.main()
