"""Comprehensive API endpoint tests covering error paths and edge cases."""
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
    """Minimal in-memory experiment store for testing (no Redis)."""

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
class TestHealthEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings(ACTIVE_STRATEGY="UCB1", REDIS_URL="redis://unused:6379/0")
        app = create_app(
            settings=settings,
            state_store=InMemoryStateStore(),
            experiment_store=InMemoryExperimentStore(),
        )
        self.client = TestClient(app)

    def test_health_returns_ok(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI runtime dependencies are not installed")
class TestExperimentEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings(ACTIVE_STRATEGY="UCB1", REDIS_URL="redis://unused:6379/0")
        self.state_store = InMemoryStateStore()
        self.experiment_store = InMemoryExperimentStore()
        app = create_app(
            settings=settings,
            state_store=self.state_store,
            experiment_store=self.experiment_store,
        )
        self.client = TestClient(app)

    def _create_experiment(self, experiment_id: str = "test_exp", **kwargs: Any) -> Any:
        payload = {"experiment_id": experiment_id, "arm_ids": ["A", "B"], **kwargs}
        return self.client.post("/experiments", json=payload)

    # ---- POST /experiments ---------------------------------------------------

    def test_create_experiment_returns_201(self) -> None:
        resp = self._create_experiment()
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["experiment_id"], "test_exp")
        self.assertEqual(sorted(body["arm_ids"]), ["A", "B"])

    def test_create_duplicate_experiment_returns_409(self) -> None:
        self._create_experiment("dup")
        resp = self._create_experiment("dup")
        self.assertEqual(resp.status_code, 409)
        self.assertIn("already exists", resp.json()["detail"])

    def test_create_experiment_with_one_arm_rejected(self) -> None:
        resp = self.client.post(
            "/experiments",
            json={"experiment_id": "one_arm", "arm_ids": ["A"]},
        )
        self.assertEqual(resp.status_code, 422)

    def test_create_experiment_with_duplicate_arms_rejected(self) -> None:
        resp = self.client.post(
            "/experiments",
            json={"experiment_id": "dup_arms", "arm_ids": ["A", "A"]},
        )
        self.assertEqual(resp.status_code, 422)

    def test_create_experiment_with_empty_arm_ids_rejected(self) -> None:
        resp = self.client.post(
            "/experiments",
            json={"experiment_id": "empty", "arm_ids": [" ", "  "]},
        )
        self.assertEqual(resp.status_code, 422)

    def test_create_experiment_with_strategy(self) -> None:
        resp = self._create_experiment("with_strat", strategy="THOMPSON")
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["strategy"], "thompson")

    def test_create_experiment_with_strategy_params(self) -> None:
        resp = self._create_experiment(
            "with_params", strategy="EPSILON_GREEDY", strategy_params={"epsilon": 0.2}
        )
        self.assertEqual(resp.status_code, 201)

    # ---- GET /experiments/{id} -----------------------------------------------

    def test_get_experiment_returns_200(self) -> None:
        self._create_experiment("get_me")
        resp = self.client.get("/experiments/get_me")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["experiment_id"], "get_me")

    def test_get_nonexistent_experiment_returns_404(self) -> None:
        resp = self.client.get("/experiments/nope")
        self.assertEqual(resp.status_code, 404)

    # ---- POST /experiments/{id}/arms -----------------------------------------

    def test_add_arms_returns_merged_list(self) -> None:
        self._create_experiment("arm_test")
        resp = self.client.post(
            "/experiments/arm_test/arms",
            json={"arm_ids": ["C", "D"]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(sorted(resp.json()["arm_ids"]), ["A", "B", "C", "D"])

    def test_add_arms_to_nonexistent_experiment_returns_404(self) -> None:
        resp = self.client.post(
            "/experiments/nope/arms",
            json={"arm_ids": ["X"]},
        )
        self.assertEqual(resp.status_code, 404)

    def test_add_duplicate_arms_are_deduplicated(self) -> None:
        self._create_experiment("dedup")
        resp = self.client.post(
            "/experiments/dedup/arms",
            json={"arm_ids": ["A", "C"]},  # A already exists
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(sorted(resp.json()["arm_ids"]), ["A", "B", "C"])


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI runtime dependencies are not installed")
class TestDecisionEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings(ACTIVE_STRATEGY="UCB1", REDIS_URL="redis://unused:6379/0")
        self.experiment_store = InMemoryExperimentStore()
        app = create_app(
            settings=settings,
            state_store=InMemoryStateStore(),
            experiment_store=self.experiment_store,
        )
        self.client = TestClient(app)

    def test_decision_returns_valid_arm(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "dec", "arm_ids": ["A", "B"]},
        )
        resp = self.client.get("/decision", params={"experiment_id": "dec"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn(body["arm_id"], {"A", "B"})
        self.assertEqual(body["experiment_id"], "dec")
        self.assertIn("strategy", body)

    def test_decision_nonexistent_experiment_returns_404(self) -> None:
        resp = self.client.get("/decision", params={"experiment_id": "nope"})
        self.assertEqual(resp.status_code, 404)

    def test_decision_missing_experiment_id_returns_422(self) -> None:
        resp = self.client.get("/decision")
        self.assertEqual(resp.status_code, 422)


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI runtime dependencies are not installed")
class TestRewardEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings(ACTIVE_STRATEGY="UCB1", REDIS_URL="redis://unused:6379/0")
        self.experiment_store = InMemoryExperimentStore()
        app = create_app(
            settings=settings,
            state_store=InMemoryStateStore(),
            experiment_store=self.experiment_store,
        )
        self.client = TestClient(app)

    def test_reward_nonexistent_experiment_returns_404(self) -> None:
        resp = self.client.post(
            "/reward",
            json={"experiment_id": "nope", "arm_id": "A", "reward": 1.0},
        )
        self.assertEqual(resp.status_code, 404)

    def test_reward_invalid_arm_returns_400(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "rew", "arm_ids": ["A", "B"]},
        )
        resp = self.client.post(
            "/reward",
            json={"experiment_id": "rew", "arm_id": "Z", "reward": 1.0},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not in experiment", resp.json()["detail"])

    def test_reward_out_of_range_rejected(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "rew2", "arm_ids": ["A", "B"]},
        )
        resp = self.client.post(
            "/reward",
            json={"experiment_id": "rew2", "arm_id": "A", "reward": 1.5},
        )
        self.assertEqual(resp.status_code, 422)

    def test_negative_reward_rejected(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "rew3", "arm_ids": ["A", "B"]},
        )
        resp = self.client.post(
            "/reward",
            json={"experiment_id": "rew3", "arm_id": "A", "reward": -0.1},
        )
        self.assertEqual(resp.status_code, 422)

    def test_reward_success(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "rew_ok", "arm_ids": ["A", "B"]},
        )
        resp = self.client.post(
            "/reward",
            json={"experiment_id": "rew_ok", "arm_id": "A", "reward": 0.5},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI runtime dependencies are not installed")
class TestConfigEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings(ACTIVE_STRATEGY="UCB1", REDIS_URL="redis://unused:6379/0")
        self.experiment_store = InMemoryExperimentStore()
        app = create_app(
            settings=settings,
            state_store=InMemoryStateStore(),
            experiment_store=self.experiment_store,
        )
        self.client = TestClient(app)

    def test_config_nonexistent_experiment_returns_404(self) -> None:
        resp = self.client.post(
            "/config",
            json={"experiment_id": "nope", "strategy": "THOMPSON"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_config_invalid_strategy_returns_400(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "cfg", "arm_ids": ["A", "B"]},
        )
        resp = self.client.post(
            "/config",
            json={"experiment_id": "cfg", "strategy": "NOPE"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_config_swap_preserves_experiment(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "cfg2", "arm_ids": ["A", "B"]},
        )
        resp = self.client.post(
            "/config",
            json={"experiment_id": "cfg2", "strategy": "THOMPSON", "strategy_params": {}},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["strategy"], "thompson")
        # Experiment should still be accessible
        get_resp = self.client.get("/experiments/cfg2")
        self.assertEqual(get_resp.status_code, 200)

    def test_decision_after_config_swap_uses_new_strategy(self) -> None:
        self.client.post(
            "/experiments",
            json={"experiment_id": "cfg3", "arm_ids": ["A", "B"]},
        )
        self.client.post(
            "/config",
            json={"experiment_id": "cfg3", "strategy": "THOMPSON"},
        )
        resp = self.client.get("/decision", params={"experiment_id": "cfg3"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["strategy"], "thompson")


if __name__ == "__main__":
    unittest.main()
