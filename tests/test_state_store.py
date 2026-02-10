"""Tests for InMemoryStateStore – every method in the BanditStateStore contract."""
from __future__ import annotations

import unittest

from bandit_core.state.memory import InMemoryStateStore


class TestInMemoryStateStore(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStateStore()

    # ---- get_arm_state -------------------------------------------------------

    def test_get_arm_state_returns_empty_dict_for_unknown_arm(self) -> None:
        self.assertEqual(self.store.get_arm_state("exp", "missing"), {})

    def test_get_arm_state_returns_copy(self) -> None:
        self.store.set_arm_state("exp", "A", {"x": 1.0})
        state = self.store.get_arm_state("exp", "A")
        state["x"] = 999.0
        self.assertEqual(self.store.get_arm_state("exp", "A")["x"], 1.0)

    # ---- get_experiment_state ------------------------------------------------

    def test_get_experiment_state_returns_all_requested_arms(self) -> None:
        self.store.set_arm_state("exp", "A", {"x": 1.0})
        self.store.set_arm_state("exp", "B", {"x": 2.0})
        result = self.store.get_experiment_state("exp", ["A", "B", "C"])
        self.assertEqual(set(result.keys()), {"A", "B", "C"})
        self.assertEqual(result["A"]["x"], 1.0)
        self.assertEqual(result["C"], {})

    # ---- set_arm_state -------------------------------------------------------

    def test_set_arm_state_overwrites_existing(self) -> None:
        self.store.set_arm_state("exp", "A", {"x": 1.0, "y": 2.0})
        self.store.set_arm_state("exp", "A", {"z": 3.0})
        state = self.store.get_arm_state("exp", "A")
        self.assertEqual(state, {"z": 3.0})

    # ---- increment -----------------------------------------------------------

    def test_increment_creates_key_if_missing(self) -> None:
        result = self.store.increment("exp", "A", "count", 1.0)
        self.assertEqual(result, 1.0)
        self.assertEqual(self.store.get_arm_state("exp", "A")["count"], 1.0)

    def test_increment_adds_to_existing_value(self) -> None:
        self.store.set_arm_state("exp", "A", {"count": 5.0})
        result = self.store.increment("exp", "A", "count", 3.0)
        self.assertEqual(result, 8.0)

    def test_increment_returns_new_value(self) -> None:
        self.store.increment("exp", "A", "val", 2.5)
        result = self.store.increment("exp", "A", "val", 1.5)
        self.assertEqual(result, 4.0)

    # ---- initialize_arm ------------------------------------------------------

    def test_initialize_arm_sets_defaults(self) -> None:
        self.store.initialize_arm("exp", "A", {"alpha": 1.0, "beta": 1.0})
        self.assertEqual(
            self.store.get_arm_state("exp", "A"),
            {"alpha": 1.0, "beta": 1.0},
        )

    def test_initialize_arm_does_not_overwrite_existing(self) -> None:
        self.store.set_arm_state("exp", "A", {"alpha": 5.0, "beta": 3.0})
        self.store.initialize_arm("exp", "A", {"alpha": 1.0, "beta": 1.0})
        state = self.store.get_arm_state("exp", "A")
        self.assertEqual(state["alpha"], 5.0)

    # ---- reset_experiment ----------------------------------------------------

    def test_reset_experiment_clears_all_arms(self) -> None:
        self.store.set_arm_state("exp", "A", {"x": 1.0})
        self.store.set_arm_state("exp", "B", {"x": 2.0})
        self.store.reset_experiment("exp")
        self.assertEqual(self.store.get_arm_state("exp", "A"), {})
        self.assertEqual(self.store.get_arm_state("exp", "B"), {})

    def test_reset_experiment_does_not_affect_other_experiments(self) -> None:
        self.store.set_arm_state("exp1", "A", {"x": 1.0})
        self.store.set_arm_state("exp2", "A", {"x": 2.0})
        self.store.reset_experiment("exp1")
        self.assertEqual(self.store.get_arm_state("exp2", "A")["x"], 2.0)

    def test_reset_nonexistent_experiment_does_not_raise(self) -> None:
        self.store.reset_experiment("nope")  # should not raise

    # ---- ensure_arms ---------------------------------------------------------

    def test_ensure_arms_initializes_multiple_arms(self) -> None:
        self.store.ensure_arms("exp", ["A", "B"], {"count": 0.0})
        self.assertEqual(self.store.get_arm_state("exp", "A"), {"count": 0.0})
        self.assertEqual(self.store.get_arm_state("exp", "B"), {"count": 0.0})


if __name__ == "__main__":
    unittest.main()
