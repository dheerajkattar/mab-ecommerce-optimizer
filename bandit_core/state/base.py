"""Abstract interface for bandit arm state persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BanditStateStore(ABC):
    """Strategy-agnostic storage backend for per-arm state.

    Every arm in an experiment is represented by a flat ``{key: float}``
    dictionary.  Strategies decide *which* keys they use (e.g. ``alpha``,
    ``beta`` for Thompson Sampling; ``count``, ``value_sum`` for UCB1).
    """

    # ---- read ---------------------------------------------------------------

    @abstractmethod
    def get_arm_state(self, experiment_id: str, arm_id: str) -> dict[str, float]:
        """Return the full state dict for a single arm.

        Must return an empty dict (not raise) when the arm has no state yet.
        """

    @abstractmethod
    def get_experiment_state(
        self, experiment_id: str, arm_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        """Return ``{arm_id: state_dict}`` for all requested arms."""

    # ---- write --------------------------------------------------------------

    @abstractmethod
    def set_arm_state(self, experiment_id: str, arm_id: str, state: dict[str, float]) -> None:
        """Overwrite the full state dict for a single arm."""

    @abstractmethod
    def increment(self, experiment_id: str, arm_id: str, key: str, amount: float = 1.0) -> float:
        """Atomically increment a single key and return the new value."""

    # ---- lifecycle ----------------------------------------------------------

    @abstractmethod
    def initialize_arm(
        self,
        experiment_id: str,
        arm_id: str,
        default_state: dict[str, float],
    ) -> None:
        """Create the arm entry *only if it does not already exist*."""

    @abstractmethod
    def reset_experiment(self, experiment_id: str) -> None:
        """Delete all arm state for the given experiment."""

    # ---- optional helpers ---------------------------------------------------

    def ensure_arms(
        self,
        experiment_id: str,
        arm_ids: list[str],
        default_state: dict[str, float],
    ) -> None:
        """Convenience: call :meth:`initialize_arm` for each arm."""
        for arm_id in arm_ids:
            self.initialize_arm(experiment_id, arm_id, default_state)
