## Parameter uncertainty in the expected stock return — 100,000 forward paths

| Uncertainty s | Dynamic, learns μ | Dynamic, ignores it | Difference | Fixed 100% stock | Fixed 60/40 | DP's own estimate (learns / ignores) | Dynamic success, lowest / highest μ decile | Avg. stock (learns / ignores) |
|---|---|---|---|---|---|---|---|---|
| 0% | 62.2% | 62.2% | +0.00 ± 0.00 pp | 54.4% | 39.7% | 62.1% / 62.1% | 62% / 62% | 81% / 81% |
| 1% | 61.7% | 61.7% | +0.03 ± 0.02 pp | 54.1% | 39.9% | 61.6% / 62.1% | 44% / 78% | 81% / 81% |
| 2% | 60.7% | 60.8% | -0.03 ± 0.02 pp | 53.6% | 40.8% | 60.5% / 62.1% | 28% / 88% | 81% / 81% |
| 3% | 59.6% | 59.6% | -0.01 ± 0.03 pp | 53.2% | 41.7% | 59.2% / 62.1% | 15% / 94% | 81% / 80% |

### Share of the variance of the cumulative stock return due to not knowing μ

| s | 1 years | 5 years | 10 years | 20 years | 30 years |
|---|---|---|---|---|---|
| 1% | 0% | 2% | 4% | 7% | 11% |
| 2% | 2% | 7% | 14% | 24% | 32% |
| 3% | 3% | 15% | 26% | 42% | 52% |

Each path draws its own expected stock return, approximately N(7%, s²), and keeps it for 20 years. "Learns μ" updates a normal prior from each year's stock return and solves the DP on (year, wealth, belief); "ignores it" is the same solver with s = 0 (the base-case policy, 2,000 DP scenarios). Differences are paired. Deciles: the success probability among the 10% of paths with the lowest and the highest drawn μ.
