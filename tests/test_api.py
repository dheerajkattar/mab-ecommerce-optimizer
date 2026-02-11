"""FastAPI endpoint tests for phase 2.

These tests auto-skip when FastAPI (or optional runtime deps) are unavailable
in the current environment.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

FASTAPI_AVAILABLE = True
try:
    from fastapi.testclient import TestClient

    from bandit_api.main import create_app
    from bandit_api.settings import Settings
    from bandit_core.state.memory import InMemoryStateStore
except Exception:
    FASTAPI_AVAILABLE = False


class InMemoryExperimentStore:
    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def create_experiment(
        self,
        experiment_id: str,
        arm_ids: List[str],
        strategy: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._items[experiment_id] = {
            "experiment_id": experiment_id,
            "arm_ids": list(arm_ids),
            "strategy": strategy,
            "strategy_params": dict(strategy_params or {}),
        }
        return dict(self._items[experiment_id])

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        item = self._items.get(experiment_id)
        return dict(item) if item else None

    def add_arms(self, experiment_id: str, arm_ids: List[str]) -> Optional[Dict[str, Any]]:
        item = self._items.get(experiment_id)
        if item is None:
            return None
        merged = sorted(set(item["arm_ids"] + arm_ids))
        item["arm_ids"] = merged
        return dict(item)

    def set_strategy_config(
        self,
        experiment_id: str,
        strategy: str,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        item = self._items.get(experiment_id)
        if item is None:
            return None
        item["strategy"] = strategy
        item["strategy_params"] = dict(strategy_params or {})
        return dict(item)


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI runtime dependencies are not installed")
class TestBanditApi(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings(ACTIVE_STRATEGY="UCB1", REDIS_URL="redis://unused:6379/0")
        state_store = InMemoryStateStore()
        experiment_store = InMemoryExperimentStore()
        self.experiment_store = experiment_store
        app = create_app(
            settings=settings,
            state_store=state_store,
            experiment_store=experiment_store,
        )
        # Enter TestClient as a context manager so FastAPI lifespan events run.
        self._client_cm = TestClient(app)
        self.client = self._client_cm.__enter__()
        self.addCleanup(self._client_cm.__exit__, None, None, None)

    def test_experiment_create_decision_reward_flow(self) -> None:
        create_resp = self.client.post(
            "/experiments",
            json={
                "experiment_id": "exp_api_1",
                "arm_ids": ["A", "B"],
                "strategy": "THOMPSON",
                "strategy_params": {},
            },
        )
        self.assertEqual(create_resp.status_code, 201)

        decision_resp = self.client.get("/decision", params={"experiment_id": "exp_api_1"})
        self.assertEqual(decision_resp.status_code, 200)
        arm_id = decision_resp.json()["arm_id"]
        self.assertIn(arm_id, {"A", "B"})

        reward_resp = self.client.post(
            "/reward",
            json={"experiment_id": "exp_api_1", "arm_id": arm_id, "reward": 1.0},
        )
        self.assertEqual(reward_resp.status_code, 200)
        self.assertEqual(reward_resp.json()["status"], "ok")

    def test_config_hot_swap(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "exp_api_2", "arm_ids": ["A", "B"]},
        )
        config_resp = self.client.post(
            "/config",
            json={
                "experiment_id": "exp_api_2",
                "strategy": "EPSILON_GREEDY",
                "strategy_params": {"epsilon": 0.2},
            },
        )
        self.assertEqual(config_resp.status_code, 200)
        body = config_resp.json()
        self.assertEqual(body["strategy"], "epsilon_greedy")
        self.assertEqual(body["strategy_params"]["epsilon"], 0.2)

    def test_create_experiment_rejects_invalid_strategy_without_persisting(self) -> None:
        create_resp = self.client.post(
            "/experiments",
            json={
                "experiment_id": "exp_invalid_create",
                "arm_ids": ["A", "B"],
                "strategy": "INVALID_STRATEGY",
            },
        )
        self.assertEqual(create_resp.status_code, 400)
        self.assertIn("Unsupported strategy", create_resp.json()["detail"])
        self.assertIsNone(self.experiment_store.get_experiment("exp_invalid_create"))

    def test_config_rejects_invalid_strategy_without_updating(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "exp_invalid_config", "arm_ids": ["A", "B"]},
        )
        bad_config = self.client.post(
            "/config",
            json={
                "experiment_id": "exp_invalid_config",
                "strategy": "INVALID_STRATEGY",
                "strategy_params": {"epsilon": 0.9},
            },
        )
        self.assertEqual(bad_config.status_code, 400)
        current = self.experiment_store.get_experiment("exp_invalid_config")
        self.assertIsNotNone(current)
        self.assertIsNone(current["strategy"])

    def test_demo_experiment_seeded_with_deterministic_strategy_seed(self) -> None:
        # Trigger app startup/lifespan path via a request.
        self.client.get("/health")
        demo = self.experiment_store.get_experiment("demo-ecommerce-cta")
        self.assertIsNotNone(demo)
        self.assertEqual(demo["strategy"], "thompson")
        self.assertEqual(demo["strategy_params"].get("seed"), 42)


if __name__ == "__main__":
    unittest.main()

