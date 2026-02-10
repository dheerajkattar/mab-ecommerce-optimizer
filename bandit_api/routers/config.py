"""Configuration endpoints for per-experiment strategy tuning."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from bandit_api.models import ConfigRequest, ConfigResponse

router = APIRouter(tags=["config"])


@router.post("/config", response_model=ConfigResponse)
def set_strategy_config(payload: ConfigRequest, request: Request) -> ConfigResponse:
    experiment_store = request.app.state.experiment_store
    strategy_factory = request.app.state.strategy_factory

    try:
        normalized_strategy = strategy_factory.validate_strategy_name(payload.strategy)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    updated = experiment_store.set_strategy_config(
        experiment_id=payload.experiment_id,
        strategy=normalized_strategy,
        strategy_params=payload.strategy_params,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{payload.experiment_id}' was not found.",
        )

    # Ensure new strategy's default fields are initialized for existing arms.
    strategy = strategy_factory.build_for_experiment(payload.experiment_id)
    strategy.initialize_experiment(payload.experiment_id, updated["arm_ids"])
    return ConfigResponse(
        experiment_id=payload.experiment_id,
        strategy=strategy.name,
        strategy_params=updated["strategy_params"],
    )

