"""FastAPI entrypoint for the Multi-Armed Bandit service."""
from __future__ import annotations

import logging
import random
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware

from bandit_api.models import DecisionResponse, RewardRequest, RewardResponse
from bandit_api.routers import config as config_router
from bandit_api.routers import experiments as experiments_router
from bandit_api.settings import Settings
from bandit_api.strategies.factory import StrategyFactory
from bandit_core.state.base import BanditStateStore

logger = logging.getLogger(__name__)

DEMO_EXPERIMENT_ID = "demo-ecommerce-cta"
DEMO_ARMS = ["blue-button", "green-button", "red-button"]
DEMO_TRUE_RATES = [0.12, 0.18, 0.09]
DEMO_ROUNDS = 200
DEMO_SEED = 42


def _seed_demo_data(
    experiment_store: Any,
    strategy_factory: StrategyFactory,
) -> None:
    """Seed a demo experiment so the dashboard isn't empty on first visit.

    Idempotent: skips if the experiment already exists in Redis.
    """
    if experiment_store.get_experiment(DEMO_EXPERIMENT_ID) is not None:
        logger.info("Demo experiment '%s' already exists, skipping seed.", DEMO_EXPERIMENT_ID)
        return

    logger.info("Seeding demo experiment '%s' with %d rounds...", DEMO_EXPERIMENT_ID, DEMO_ROUNDS)
    experiment_store.create_experiment(
        experiment_id=DEMO_EXPERIMENT_ID,
        arm_ids=DEMO_ARMS,
        strategy="thompson",
        strategy_params={"seed": DEMO_SEED},
    )

    strategy = strategy_factory.build_for_experiment(DEMO_EXPERIMENT_ID)
    strategy.initialize_experiment(DEMO_EXPERIMENT_ID, DEMO_ARMS)

    rng = random.Random(DEMO_SEED)
    for _ in range(DEMO_ROUNDS):
        arm_id = strategy.select_arm(
            experiment_id=DEMO_EXPERIMENT_ID,
            arm_ids=DEMO_ARMS,
            user_context=None,
        )
        true_rate = DEMO_TRUE_RATES[DEMO_ARMS.index(arm_id)]
        reward = 1.0 if rng.random() < true_rate else 0.0
        strategy.update(DEMO_EXPERIMENT_ID, arm_id, reward)

    logger.info("Demo experiment seeded successfully.")


def create_app(
    *,
    settings: Optional[Settings] = None,
    state_store: Optional[BanditStateStore] = None,
    experiment_store: Optional[Any] = None,
) -> FastAPI:
    """Create and configure the FastAPI app.

    Optional store injections make this function test-friendly.
    """
    app_settings = settings or Settings()

    if state_store is None or experiment_store is None:
        import redis

        from bandit_api.state.redis_store import RedisBanditStateStore, RedisExperimentStore

        redis_client = redis.from_url(app_settings.redis_url)
        resolved_state_store = state_store or RedisBanditStateStore(redis_client)
        resolved_experiment_store = experiment_store or RedisExperimentStore(redis_client)
    else:
        resolved_state_store = state_store
        resolved_experiment_store = experiment_store

    strategy_factory = StrategyFactory(
        state_store=resolved_state_store,
        experiment_store=resolved_experiment_store,
        default_strategy=app_settings.active_strategy,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _seed_demo_data(resolved_experiment_store, strategy_factory)
        yield

    app = FastAPI(
        title="Multi-Armed Bandit API",
        version="0.1.0",
        description="Strategy-agnostic bandit decisioning API backed by Redis.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = app_settings
    app.state.state_store = resolved_state_store
    app.state.experiment_store = resolved_experiment_store
    app.state.strategy_factory = strategy_factory

    app.include_router(experiments_router.router)
    app.include_router(config_router.router)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/decision", response_model=DecisionResponse)
    def get_decision(
        request: Request,
        experiment_id: str = Query(..., min_length=1),
    ) -> DecisionResponse:
        experiment = request.app.state.experiment_store.get_experiment(experiment_id)
        if experiment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment '{experiment_id}' was not found.",
            )

        arm_ids = experiment["arm_ids"]
        strategy = request.app.state.strategy_factory.build_for_experiment(experiment_id)
        strategy.initialize_experiment(experiment_id, arm_ids)

        arm_id = strategy.select_arm(
            experiment_id=experiment_id,
            arm_ids=arm_ids,
            user_context=None,
        )
        return DecisionResponse(
            experiment_id=experiment_id,
            arm_id=arm_id,
            strategy=strategy.name,
            metadata={"arms": arm_ids},
        )

    @app.post("/reward", response_model=RewardResponse)
    def post_reward(payload: RewardRequest, request: Request) -> RewardResponse:
        experiment = request.app.state.experiment_store.get_experiment(payload.experiment_id)
        if experiment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment '{payload.experiment_id}' was not found.",
            )
        if payload.arm_id not in experiment["arm_ids"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Arm '{payload.arm_id}' is not in experiment '{payload.experiment_id}'.",
            )

        strategy = request.app.state.strategy_factory.build_for_experiment(payload.experiment_id)
        strategy.initialize_experiment(payload.experiment_id, experiment["arm_ids"])
        strategy.update(payload.experiment_id, payload.arm_id, payload.reward)

        return RewardResponse(
            status="ok",
            experiment_id=payload.experiment_id,
            arm_id=payload.arm_id,
            strategy=strategy.name,
        )

    return app


app = create_app()

