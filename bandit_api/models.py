"""Pydantic models for request/response payloads."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ExperimentCreateRequest(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=120)
    arm_ids: List[str] = Field(min_length=2)
    strategy: Optional[str] = Field(default=None, description="e.g. UCB1, THOMPSON, EPSILON_GREEDY")
    strategy_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("arm_ids")
    @classmethod
    def validate_arms(cls, value: List[str]) -> List[str]:
        normalized = [arm.strip() for arm in value if arm.strip()]
        if len(normalized) < 2:
            raise ValueError("at least two non-empty arm IDs are required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("arm IDs must be unique")
        return normalized


class AddArmsRequest(BaseModel):
    arm_ids: List[str] = Field(min_length=1)

    @field_validator("arm_ids")
    @classmethod
    def validate_arms(cls, value: List[str]) -> List[str]:
        normalized = [arm.strip() for arm in value if arm.strip()]
        if not normalized:
            raise ValueError("arm_ids cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("arm IDs must be unique")
        return normalized


class ExperimentResponse(BaseModel):
    experiment_id: str
    arm_ids: List[str]
    strategy: Optional[str] = None
    strategy_params: Dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    experiment_id: str
    arm_id: str
    strategy: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RewardRequest(BaseModel):
    experiment_id: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    reward: float = Field(ge=0.0, le=1.0)
    user_context: Optional[Dict[str, Any]] = None


class RewardResponse(BaseModel):
    status: str
    experiment_id: str
    arm_id: str
    strategy: str


class ConfigRequest(BaseModel):
    experiment_id: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    strategy_params: Dict[str, Any] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    experiment_id: str
    strategy: str
    strategy_params: Dict[str, Any] = Field(default_factory=dict)

