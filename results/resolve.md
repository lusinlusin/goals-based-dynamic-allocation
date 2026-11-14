## Re-solving when the assumptions change — 100,000 forward paths

### Stock return 7% → 5% from year 10

| Approach | Success | ± s.e. | vs keeping the table | Avg. stock weight, years 10–19 | Table's own forecast when it acts |
|---|---|---|---|---|---|
| Keep the original table | 50.9% | 0.16% | — | 71% | 62.2% |
| Re-solve every year | 51.1% | 0.16% | +0.2 pp ± 0.05 | 67% | 51.2% |
| Re-solve, change noticed 3 years late | 51.0% | 0.16% | +0.1 pp ± 0.04 | 69% | 51.2% |
| Hindsight: changes known from the start | 51.1% | 0.16% | +0.3 pp ± 0.05 | 67% | 51.2% |
| Fixed 100% stock | 40.6% | 0.16% | -10.3 pp ± 0.10 | 100% | n/a |
| Fixed 60/40 | 28.0% | 0.14% | -22.8 pp ± 0.13 | 60% | n/a |

### Every change tested

| Change | Keep the table | Re-solve at once | Re-solve 3 years late | Hindsight | Original table's forecast at the change | Re-solved forecast |
|---|---|---|---|---|---|---|
| Stock return 7% → 5% from year 10 | 50.9% | 51.1% (+0.2 pp) | 51.0% (+0.1 pp) | 51.1% (+0.3 pp) | 62.2% | 51.2% |
| Stock return 7% → 3% from year 10 | 40.2% | 41.6% (+1.4 pp) | 41.2% (+1.0 pp) | 41.9% (+1.6 pp) | 62.2% | 41.5% |
| Stock return 7% → 5% from year 5 | 45.3% | 45.6% (+0.2 pp) | 45.5% (+0.2 pp) | 45.5% (+0.2 pp) | 62.3% | 45.5% |
| Stock return 7% → 9% from year 10 | 73.0% | 73.1% (+0.1 pp) | 73.1% (+0.1 pp) | 73.2% (+0.2 pp) | 62.2% | 73.0% |
| Stock volatility 16% → 24% from year 10 | 60.9% | 61.7% (+0.7 pp) | 61.5% (+0.5 pp) | 61.7% (+0.8 pp) | 62.2% | 61.7% |

All approaches run on the same paths, which follow the true (changing) assumptions. Differences against keeping the table are paired (same paths) with their standard error. "Forecast" is the success probability the table itself reports, averaged over paths, in the year the change is acted on (year 10, or 13 when noticed late); compare it with the success that then happens. The first change was re-solved every year; re-solves in years before the change reproduced the original table exactly. To get back to the original plan's 62.1% after the first change, saving has to rise from $10,000 to $14,100 a year for years 10–19 (re-solved with the new saving).
