#!/usr/bin/env python3
"""Benchmark all three bandit strategies on the same synthetic dataset.

Usage:
    python benchmark_strategies.py          # defaults: 5000 rounds, seed=42
    python benchmark_strategies.py -n 10000 --seed 7

Produces ``benchmark_regret.png`` in the current directory.
"""

from __future__ import annotations

import argparse

import numpy as np

from bandit_core.sim.synthetic import BernoulliBanditEnv, run_simulation
from bandit_core.state.memory import InMemoryStateStore
from bandit_core.strategies.base import BaseBanditStrategy
from bandit_core.strategies.epsilon_greedy import EpsilonGreedyStrategy
from bandit_core.strategies.thompson import ThompsonSamplingStrategy
from bandit_core.strategies.ucb1 import UCB1Strategy

# ---- default problem setup --------------------------------------------------

DEFAULT_ARM_RATES: dict[str, float] = {
    "A": 0.10,  # 10 % conversion
    "B": 0.13,  # 13 %
    "C": 0.18,  # 18 % - the true best arm
    "D": 0.05,  # 5 %
}


def build_strategies(seed: int) -> list[tuple[str, BaseBanditStrategy]]:
    """Instantiate one of each strategy, each with its own store."""
    return [
        (
            "Thompson Sampling",
            ThompsonSamplingStrategy(InMemoryStateStore(), seed=seed),
        ),
        (
            "Epsilon-Greedy (eps=0.1)",
            EpsilonGreedyStrategy(InMemoryStateStore(), epsilon=0.1, seed=seed),
        ),
        (
            "UCB1",
            UCB1Strategy(InMemoryStateStore(), seed=seed),
        ),
    ]


def _plot_regret_curves(
    results: dict[str, list[float]],
    n_rounds: int,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate plots. Install with: pip install -e '.[bench]'"
        ) from exc

    # ---- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds_axis = np.arange(1, n_rounds + 1)

    for label, regrets in results.items():
        ax.plot(rounds_axis, regrets, label=label, linewidth=1.5)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title("Multi-Armed Bandit - Cumulative Regret Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bandit strategies")
    parser.add_argument("-n", "--rounds", type=int, default=5_000, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Run benchmark and print final regret table without saving a plot.",
    )
    args = parser.parse_args()

    n_rounds: int = args.rounds
    seed: int = args.seed

    strategies = build_strategies(seed)

    # We use the same seed for the environment across strategies so that
    # each strategy faces an *identical* reward sequence.
    results: dict[str, list[float]] = {}
    for label, strategy in strategies:
        env = BernoulliBanditEnv(DEFAULT_ARM_RATES, seed=seed)
        regrets, _rewards = run_simulation(
            strategy,
            env,
            experiment_id=f"bench_{strategy.name}",
            n_rounds=n_rounds,
        )
        results[label] = regrets

    out_path = "benchmark_regret.png"
    if args.skip_plot:
        print("[benchmark] Plot generation skipped (--skip-plot).")
    else:
        _plot_regret_curves(results, n_rounds, out_path)
        print(f"[benchmark] Saved regret chart -> {out_path}")

    # Also print final regrets for quick comparison
    print(f"\n{'Strategy':<30} {'Final Regret':>14}")
    print("-" * 46)
    for label, regrets in results.items():
        print(f"{label:<30} {regrets[-1]:>14.2f}")


if __name__ == "__main__":
    main()
