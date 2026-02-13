"""Abstract base class shared by every bandit algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bandit_core.state.base import BanditStateStore


class BaseBanditStrategy(ABC):
    """Common interface for all Multi-Armed Bandit strategies.

    Every strategy receives a :class:`BanditStateStore` so it can read and
    write per-arm statistics without knowing where they live (memory, Redis,
    DynamoDB, …).
    """

    name: str = "base"  # human-friendly label, overridden by subclasses

    def __init__(self, store: BanditStateStore, **kwargs: Any) -> None:
        self.store = store

    # ---- abstract API -------------------------------------------------------

    @abstractmethod
    def select_arm(
        self,
        experiment_id: str,
        arm_ids: list[str],
        user_context: dict[str, Any] | None = None,
    ) -> str:
        """Choose the arm to present to a user.

        Parameters
        ----------
        experiment_id:
            Namespace that groups the arms together.
        arm_ids:
            The list of available arms / variants.
        user_context:
            Optional contextual features (reserved for contextual bandits).

        Returns
        -------
        str
            The chosen ``arm_id``.
        """

    @abstractmethod
    def update(
        self,
        experiment_id: str,
        arm_id: str,
        reward: float,
    ) -> None:
        """Record an observed reward for the given arm.

        Parameters
        ----------
        experiment_id:
            The experiment that the arm belongs to.
        arm_id:
            The arm that was shown.
        reward:
            Observed reward (typically 0 or 1 for Bernoulli trials).
        """

    @abstractmethod
    def default_arm_state(self) -> dict[str, float]:
        """Return the initial state dict for a brand-new arm.

        This is used by ``store.ensure_arms()`` when an experiment is first
        set up.
        """

    # ---- convenience --------------------------------------------------------

    def initialize_experiment(self, experiment_id: str, arm_ids: list[str]) -> None:
        """Ensure all arms exist in the store with proper defaults."""
        self.store.ensure_arms(experiment_id, arm_ids, self.default_arm_state())
