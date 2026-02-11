# Convergence Test Results

## Test Configuration

- **Samples:** 20 synthetic 4-arm Bernoulli environments
- **Rounds per sample:** 2,000
- **Arms per sample:** 4 (A, B, C, D)
- **Best arm gap:** Each sample guarantees the best arm is at least 0.15 above the second-best
- **Convergence criteria:** Best arm selected in ≥80% of a trailing 100-round window
- **Seed:** 42 (deterministic, reproducible)

## Strategy Ranking

| Strategy | Median Convergence Round | Mean Final Regret | Samples Converged | Sample Wins |
|---|---|---|---|---|
| **Thompson Sampling** | 110 | 20.89 | 20/20 | 10 |
| Epsilon-Greedy (eps=0.1) | 110 | 62.93 | 20/20 | 10 |
| UCB1 | 331 | 93.68 | 20/20 | 0 |

## Per-Sample Results

| Sample | Best Arm | Gap | Thompson | Epsilon-Greedy | UCB1 | Fastest |
|---|---|---|---|---|---|---|
| 0 | A | 0.296 | 100 | 100 | 374 | Thompson Sampling |
| 1 | A | 0.205 | 103 | 100 | 218 | Epsilon-Greedy |
| 2 | B | 0.233 | 110 | 191 | 274 | Thompson Sampling |
| 3 | A | 0.203 | 100 | 314 | 596 | Thompson Sampling |
| 4 | B | 0.157 | 138 | 134 | 346 | Epsilon-Greedy |
| 5 | B | 0.199 | 100 | 108 | 375 | Thompson Sampling |
| 6 | C | 0.184 | 139 | 179 | 910 | Thompson Sampling |
| 7 | B | 0.197 | 320 | 101 | 372 | Epsilon-Greedy |
| 8 | A | 0.181 | 116 | 100 | 600 | Epsilon-Greedy |
| 9 | C | 0.301 | 100 | 137 | 227 | Thompson Sampling |
| 10 | A | 0.329 | 103 | 100 | 251 | Epsilon-Greedy |
| 11 | A | 0.233 | 109 | 170 | 429 | Thompson Sampling |
| 12 | D | 0.211 | 131 | 100 | 618 | Epsilon-Greedy |
| 13 | A | 0.406 | 102 | 100 | 220 | Epsilon-Greedy |
| 14 | B | 0.272 | 124 | 280 | 212 | Thompson Sampling |
| 15 | B | 0.220 | 231 | 184 | 316 | Epsilon-Greedy |
| 16 | B | 0.207 | 100 | 106 | 283 | Thompson Sampling |
| 17 | B | 0.267 | 151 | 135 | 225 | Epsilon-Greedy |
| 18 | D | 0.267 | 100 | 103 | 370 | Thompson Sampling |
| 19 | D | 0.286 | 139 | 112 | 192 | Epsilon-Greedy |

## Analysis

### Best Strategy: Thompson Sampling

Thompson Sampling and Epsilon-Greedy tie on median convergence speed (both at round 110), and each wins 10 out of 20 samples on fastest convergence. However, **Thompson Sampling is the clear winner** when considering total performance:

- **3x lower mean regret** (20.89 vs 62.93) -- Thompson wastes far fewer rounds on suboptimal arms after identifying the best one
- **Efficient exploration** -- Thompson's Bayesian posterior naturally balances exploration and exploitation without a fixed epsilon parameter
- **No tuning required** -- Unlike Epsilon-Greedy, Thompson has no hyperparameter to configure

### Epsilon-Greedy (eps=0.1)

Converges quickly but accumulates significantly more regret because the fixed 10% exploration rate continues pulling random arms even after the best arm is well-established. This wasted exploration accounts for the 3x regret gap versus Thompson.

### UCB1

Converges reliably on all 20 samples but is the slowest (median round 331). Its deterministic confidence-bound exploration is thorough but conservative, particularly on samples with smaller gaps between arm rates (e.g., sample 6 took 910 rounds). UCB1 never won a single sample on convergence speed. Its mean regret (93.68) is the highest of the three strategies.

## Recommendation

For production e-commerce optimization, **Thompson Sampling** is the recommended default strategy. It delivers the best balance of fast convergence and low cumulative regret, requiring no hyperparameter tuning.
