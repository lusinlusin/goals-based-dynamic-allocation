## Two market regimes — 100,000 forward paths

| Policy | Starts in good state | Starts in bad state | Start drawn from long-run mix |
|---|---|---|---|
| Regime-aware DP | 72.4% | 59.9% | 66.1% |
| Regime-blind DP (base case) | 69.8% (-2.7 pp) | 57.5% (-2.4 pp) | 63.5% (-2.6 pp) |
| Fixed 100% stock | 61.9% (-10.5 pp) | 50.8% (-9.1 pp) | 56.2% (-9.8 pp) |
| Fixed 60/40 | 47.9% (-24.6 pp) | 37.8% (-22.1 pp) | 42.8% (-23.3 pp) |

### Planning with the wrong persistence

| Planned with stay probability | Success (true stay = 0.9) | Cost vs the right probability | Avg. stock weight |
|---|---|---|---|
| 0.9 (right) | 66.1% | — | 74% |
| 0.5 | 65.4% | -0.70 ± 0.08 pp | 68% |
| 0.75 | 65.7% | -0.33 ± 0.07 pp | 70% |
| 0.95 | 65.9% | -0.13 ± 0.06 pp | 79% |
| 0.99 | 65.6% | -0.48 ± 0.08 pp | 81% |

Good state: stocks 10.0% expected, 12.0% volatility; bad state: 4.0% and 18.7%; bonds 3% and 5% in both. Each state persists with probability 0.9 a year, and the 50/50 long-run mix has the base-case 7% and 16%. The investor sees the current state. The regime-blind policy is the base-case DP. Brackets: paired difference against the regime-aware policy. The DP's own estimate for the regime-aware policy is 72.1% starting good and 59.7% starting bad.
