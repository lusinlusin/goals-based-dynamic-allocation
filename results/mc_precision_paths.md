## Forward-simulation precision (100,000 paths)

| Comparison | Difference in success | s.e. (same paths) | s.e. (separate paths) | Paths to tell apart: same / separate | Paths for ±0.5 pp: same / separate |
|---|---|---|---|---|---|
| DP · goal probability vs Fixed 60/40 | +22.5 pp | 0.13 pp | 0.22 pp | 14 / 36 | 27,279 / 72,906 |
| DP · goal probability vs Fixed 100% stock | +7.8 pp | 0.10 pp | 0.22 pp | 59 / 308 | 14,087 / 74,251 |
| DP · goal probability vs DP · goal, stock ≤ 80% | +7.7 pp | 0.09 pp | 0.22 pp | 49 / 311 | 11,675 / 74,245 |
| Fixed 70/30 vs Glide path 90→30 | +8.9 pp | 0.11 pp | 0.22 pp | 55 / 231 | 17,291 / 73,420 |
| Fixed 60/40 vs Glide path 90→30 | +3.7 pp | 0.09 pp | 0.22 pp | 258 / 1,345 | 13,824 / 72,173 |
| DP · power utility γ=5 vs Glide path 90→30 | +1.8 pp | 0.06 pp | 0.22 pp | 383 / 5,382 | 5,086 / 71,536 |

'Paths to tell apart' = paths at which the difference is 1.96 standard errors from zero; 'paths for ±0.5 pp' = paths at which the 95% interval on the difference is ±0.5 pp. A single success probability near 62% has a standard error of 0.48 pp at 10,000 paths and 0.15 pp at 100,000.
