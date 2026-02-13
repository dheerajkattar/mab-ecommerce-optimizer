"""Pydantic models for request/response payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExperimentCreateRequest(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=120)
    arm_ids: list[str] = Field(min_length=2)
    strategy: str | None = Field(default=None, description="e.g. UCB1, THOMPSON, EPSILON_GREEDY")
    strategy_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arm_ids")
    @classmethod
    def validate_arms(cls, value: list[str]) -> list[str]:
        normalized = [arm.strip() for arm in value if arm.strip()]
        if len(normalized) < 2:
            raise ValueError("at least two non-empty arm IDs are required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("arm IDs must be unique")
        return normalized


class AddArmsRequest(BaseModel):
    arm_ids: list[str] = Field(min_length=1)

    @field_validator("arm_ids")
    @classmethod
    def validate_arms(cls, value: list[str]) -> list[str]:
        normalized = [arm.strip() for arm in value if arm.strip()]
        if not normalized:
            raise ValueError("arm_ids cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("arm IDs must be unique")
        return normalized


class ExperimentResponse(BaseModel):
    experiment_id: str
    arm_ids: list[str]
    strategy: str | None = None
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    experiment_id: str
    arm_id: str
    strategy: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardRequest(BaseModel):
    experiment_id: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    reward: float = Field(ge=0.0, le=1.0)
    user_context: dict[str, Any] | None = None


class RewardResponse(BaseModel):
    status: str
    experiment_id: str
    arm_id: str
    strategy: str


class ConfigRequest(BaseModel):
    experiment_id: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    experiment_id: str
    strategy: str
    strategy_params: dict[str, Any] = Field(default_factory=dict)
