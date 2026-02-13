"""Experiment management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from bandit_api.models import AddArmsRequest, ExperimentCreateRequest, ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreateRequest, request: Request) -> ExperimentResponse:
    experiment_store = request.app.state.experiment_store
    strategy_factory = request.app.state.strategy_factory

    existing = experiment_store.get_experiment(payload.experiment_id)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment '{payload.experiment_id}' already exists.",
        )

    normalized_strategy = None
    if payload.strategy is not None:
        try:
            normalized_strategy = strategy_factory.validate_strategy_name(payload.strategy)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    created = experiment_store.create_experiment(
        experiment_id=payload.experiment_id,
        arm_ids=payload.arm_ids,
        strategy=normalized_strategy,
        strategy_params=payload.strategy_params,
    )
    strategy = strategy_factory.build_for_experiment(payload.experiment_id)
    strategy.initialize_experiment(payload.experiment_id, payload.arm_ids)
    return ExperimentResponse(**created)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: str, request: Request) -> ExperimentResponse:
    experiment_store = request.app.state.experiment_store
    experiment = experiment_store.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )
    return ExperimentResponse(**experiment)


@router.post("/{experiment_id}/arms", response_model=ExperimentResponse)
def add_arms(experiment_id: str, payload: AddArmsRequest, request: Request) -> ExperimentResponse:
    experiment_store = request.app.state.experiment_store
    strategy_factory = request.app.state.strategy_factory

    updated = experiment_store.add_arms(experiment_id, payload.arm_ids)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    strategy = strategy_factory.build_for_experiment(experiment_id)
    strategy.initialize_experiment(experiment_id, updated["arm_ids"])
    return ExperimentResponse(**updated)
