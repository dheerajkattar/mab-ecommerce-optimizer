"""Convergence tests: verify each strategy finds the best arm across 20 samples."""
from __future__ import annotations

import unittest
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from bandit_core.sim.synthetic import BernoulliBanditEnv
from bandit_core.state.memory import InMemoryStateStore
from bandit_core.strategies.base import BaseBanditStrategy
from bandit_core.strategies.epsilon_greedy import EpsilonGreedyStrategy
from bandit_core.strategies.thompson import ThompsonSamplingStrategy
from bandit_core.strategies.ucb1 import UCB1Strategy

# ---- Constants ---------------------------------------------------------------

ARM_IDS = ["A", "B", "C", "D"]
N_ROUNDS = 2000
N_SAMPLES = 20
BASE_SEED = 42
CONVERGENCE_WINDOW = 100
CONVERGENCE_THRESHOLD = 0.80
PROBE_COUNT = 200
PROBE_BEST_ARM_MIN = 0.70


# ---- Generate 20 synthetic samples ------------------------------------------

def _generate_samples(n: int, seed: int) -> List[Dict[str, float]]:
    """Generate *n* 4-arm rate configs, each with a clear best arm.

    For every sample the best arm's rate is at least 0.15 above the
    second-best, making convergence unambiguous.
    """
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, float]] = []
    for _ in range(n):
        # Pick 4 rates in [0.03, 0.40], then boost one to create a clear winner
        rates = rng.uniform(0.03, 0.40, size=4)
        winner_idx = int(rng.integers(0, 4))
        # Ensure winner is at least 0.15 above all others
        rates[winner_idx] = float(np.clip(max(rates) + rng.uniform(0.15, 0.30), 0.0, 0.95))
        samples.append({arm: float(round(rates[i], 3)) for i, arm in enumerate(ARM_IDS)})
    return samples


SAMPLES = _generate_samples(N_SAMPLES, seed=BASE_SEED)


# ---- Helpers -----------------------------------------------------------------

class SampleResult(NamedTuple):
    sample_idx: int
    best_arm: str
    choices: List[str]
    regrets: List[float]
    convergence_round: Optional[int]
    probe_best_pct: float


def _best_arm(rates: Dict[str, float]) -> str:
    return max(rates, key=rates.get)  # type: ignore[arg-type]


def _second_best_rate(rates: Dict[str, float]) -> float:
    sorted_rates = sorted(rates.values(), reverse=True)
    return sorted_rates[1]


def _run_simulation_with_choices(
    strategy: BaseBanditStrategy,
    env: BernoulliBanditEnv,
    experiment_id: str,
    n_rounds: int,
) -> Tuple[List[str], List[float]]:
    """Run a simulation and return per-round arm choices and cumulative regrets."""
    arm_ids = env.arm_ids
    strategy.initialize_experiment(experiment_id, arm_ids)

    choices: List[str] = []
    cumulative_regret = 0.0
    regrets: List[float] = []

    for _ in range(n_rounds):
        arm_id = strategy.select_arm(experiment_id, arm_ids)
        reward = env.pull(arm_id)
        strategy.update(experiment_id, arm_id, reward)

        choices.append(arm_id)
        cumulative_regret += env.regret(arm_id)
        regrets.append(cumulative_regret)

    return choices, regrets


def _convergence_round(
    choices: List[str],
    best_arm: str,
    window: int = CONVERGENCE_WINDOW,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> Optional[int]:
    """Return the first round where *best_arm* dominates the trailing window."""
    if len(choices) < window:
        return None
    best_count = sum(1 for c in choices[:window] if c == best_arm)
    for t in range(window, len(choices)):
        if choices[t] == best_arm:
            best_count += 1
        if choices[t - window] == best_arm:
            best_count -= 1
        if best_count / window >= threshold:
            return t
    return None


def _probe_best_pct(
    strategy: BaseBanditStrategy,
    experiment_id: str,
    arm_ids: List[str],
    best_arm: str,
    n_probes: int = PROBE_COUNT,
) -> float:
    """Run probe selections (no updates) and return fraction choosing best arm."""
    count = sum(
        1 for _ in range(n_probes)
        if strategy.select_arm(experiment_id, arm_ids) == best_arm
    )
    return count / n_probes


def _run_strategy_on_all_samples(
    strategy_name: str,
    build_fn,
) -> List[SampleResult]:
    """Run a strategy across all 20 samples, return per-sample results."""
    results: List[SampleResult] = []
    for i, rates in enumerate(SAMPLES):
        seed = BASE_SEED + i
        strategy = build_fn(seed)
        env = BernoulliBanditEnv(rates, seed=seed)
        exp_id = f"{strategy_name}_sample_{i}"
        best = _best_arm(rates)

        choices, regrets = _run_simulation_with_choices(strategy, env, exp_id, N_ROUNDS)
        conv = _convergence_round(choices, best)
        probe_pct = _probe_best_pct(strategy, exp_id, env.arm_ids, best)

        results.append(SampleResult(
            sample_idx=i,
            best_arm=best,
            choices=choices,
            regrets=regrets,
            convergence_round=conv,
            probe_best_pct=probe_pct,
        ))
    return results


# ---- Strategy builders -------------------------------------------------------

def _build_thompson(seed: int) -> BaseBanditStrategy:
    return ThompsonSamplingStrategy(InMemoryStateStore(), seed=seed)

def _build_epsilon_greedy(seed: int) -> BaseBanditStrategy:
    return EpsilonGreedyStrategy(InMemoryStateStore(), epsilon=0.1, seed=seed)

def _build_ucb1(seed: int) -> BaseBanditStrategy:
    return UCB1Strategy(InMemoryStateStore(), seed=seed)


# ---- Tests -------------------------------------------------------------------

class TestConvergence(unittest.TestCase):
    """Verify each strategy converges across 20 diverse synthetic samples."""

    def _assert_strategy_converges_all_samples(
        self,
        strategy_name: str,
        build_fn,
    ) -> List[SampleResult]:
        results = _run_strategy_on_all_samples(strategy_name, build_fn)

        converged_count = 0
        for r in results:
            rates = SAMPLES[r.sample_idx]
            gap = rates[r.best_arm] - _second_best_rate(rates)

            # Best arm should dominate the final portion of training.
            # We check actual training choices rather than post-training probes
            # because UCB1's exploration bonus causes it to keep exploring even
            # after it has effectively identified the best arm.
            tail = r.choices[-PROBE_COUNT:]
            tail_best_pct = sum(1 for c in tail if c == r.best_arm) / len(tail)
            self.assertGreater(
                tail_best_pct,
                PROBE_BEST_ARM_MIN,
                f"{strategy_name} sample {r.sample_idx}: best arm '{r.best_arm}' "
                f"selected only {tail_best_pct:.0%} in last {PROBE_COUNT} training "
                f"rounds (expected >{PROBE_BEST_ARM_MIN:.0%}). Rates: {rates}",
            )

            # Sublinear regret
            avg_regret = r.regrets[-1] / N_ROUNDS
            self.assertLess(
                avg_regret,
                gap,
                f"{strategy_name} sample {r.sample_idx}: avg regret/round "
                f"({avg_regret:.4f}) >= gap ({gap:.3f}). Rates: {rates}",
            )

            if r.convergence_round is not None:
                converged_count += 1

        # At least 90% of samples should converge within N_ROUNDS
        self.assertGreaterEqual(
            converged_count,
            int(N_SAMPLES * 0.9),
            f"{strategy_name}: only {converged_count}/{N_SAMPLES} samples converged "
            f"within {N_ROUNDS} rounds",
        )

        return results

    def test_thompson_converges_across_all_samples(self) -> None:
        self._assert_strategy_converges_all_samples("thompson", _build_thompson)

    def test_epsilon_greedy_converges_across_all_samples(self) -> None:
        self._assert_strategy_converges_all_samples("epsilon_greedy", _build_epsilon_greedy)

    def test_ucb1_converges_across_all_samples(self) -> None:
        self._assert_strategy_converges_all_samples("ucb1", _build_ucb1)

    def test_compare_convergence_speed_across_all_samples(self) -> None:
        """Run all strategies on all 20 samples, rank by median convergence round."""
        all_results: Dict[str, List[SampleResult]] = {}
        for name, build_fn in [
            ("Thompson Sampling", _build_thompson),
            ("Epsilon-Greedy (eps=0.1)", _build_epsilon_greedy),
            ("UCB1", _build_ucb1),
        ]:
            all_results[name] = _run_strategy_on_all_samples(name, build_fn)

        # Compute per-strategy aggregate stats
        summary: List[Tuple[str, float, float, int, int]] = []
        for name, results in all_results.items():
            conv_rounds = [r.convergence_round for r in results if r.convergence_round is not None]
            n_converged = len(conv_rounds)
            median_conv = float(np.median(conv_rounds)) if conv_rounds else float("inf")
            mean_regret = float(np.mean([r.regrets[-1] for r in results]))
            # Find how many samples this strategy converged fastest on
            summary.append((name, median_conv, mean_regret, n_converged, 0))

        # Count per-sample wins (fastest convergence round)
        strategy_names = list(all_results.keys())
        win_counts = {name: 0 for name in strategy_names}
        for i in range(N_SAMPLES):
            sample_conv = {}
            for name in strategy_names:
                cr = all_results[name][i].convergence_round
                sample_conv[name] = cr if cr is not None else float("inf")
            winner = min(sample_conv, key=sample_conv.get)  # type: ignore[arg-type]
            win_counts[winner] += 1

        # Rebuild summary with win counts
        summary = []
        for name in strategy_names:
            results = all_results[name]
            conv_rounds = [r.convergence_round for r in results if r.convergence_round is not None]
            n_converged = len(conv_rounds)
            median_conv = float(np.median(conv_rounds)) if conv_rounds else float("inf")
            mean_regret = float(np.mean([r.regrets[-1] for r in results]))
            summary.append((name, median_conv, mean_regret, n_converged, win_counts[name]))

        # Sort by median convergence round
        summary.sort(key=lambda s: s[1])

        # Print report
        print(f"\n{'='*80}")
        print(f"Convergence Speed Comparison — {N_SAMPLES} samples, "
              f"{N_ROUNDS} rounds each, 4 arms")
        print(f"{'='*80}")
        header = (
            f"{'Strategy':<30} {'Med. Conv':>10} {'Mean Regret':>12} "
            f"{'Converged':>10} {'Wins':>6}"
        )
        print(header)
        print("-" * 70)
        for name, med_conv, mean_reg, n_conv, wins in summary:
            med_str = f"{med_conv:.0f}" if med_conv < float("inf") else "N/A"
            print(f"{name:<30} {med_str:>10} {mean_reg:>12.2f} "
                  f"{n_conv:>6}/{N_SAMPLES:<3} {wins:>6}")
        print("-" * 70)
        print(f"Best overall: {summary[0][0]} "
              f"(median convergence round {summary[0][1]:.0f}, "
              f"{summary[0][4]} sample wins)")

        # Print per-sample detail
        print(f"\n{'Sample':>6}  {'Best Arm':>8}  {'Gap':>6}  ", end="")
        for name in strategy_names:
            print(f"{'Conv.' + name[:5]:>14}", end="")
        print(f"  {'Winner':<25}")
        print("-" * 100)
        for i in range(N_SAMPLES):
            rates = SAMPLES[i]
            best = _best_arm(rates)
            gap = rates[best] - _second_best_rate(rates)
            print(f"{i:>6}  {best:>8}  {gap:>6.3f}  ", end="")
            best_conv = float("inf")
            best_name = ""
            for name in strategy_names:
                cr = all_results[name][i].convergence_round
                cr_str = str(cr) if cr is not None else "N/A"
                print(f"{cr_str:>14}", end="")
                if cr is not None and cr < best_conv:
                    best_conv = cr
                    best_name = name
            print(f"  {best_name:<25}")
        print(f"{'='*80}\n")

        # Assert all strategies converge on at least 90% of samples
        for name, _, _, n_conv, _ in summary:
            self.assertGreaterEqual(
                n_conv,
                int(N_SAMPLES * 0.9),
                f"{name}: converged on only {n_conv}/{N_SAMPLES} samples",
            )


if __name__ == "__main__":
    unittest.main()
