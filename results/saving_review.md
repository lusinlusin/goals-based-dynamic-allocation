## Saving reviewed every year — 100,000 forward paths

### Average yearly saving needed for a 75% chance of reaching the goal (base case)

| Allocation | Fixed saving | Saving reviewed every year: average | 10th–90th pct across paths | Same in present value (3%) | Success (fixed / reviewed) | Years at the $40K cap | Avg. stock (fixed / reviewed) |
|---|---|---|---|---|---|---|---|
| Fixed 60/40 | $14,500 | $12,300 | $7,300–$17,700 | $14,500 / $12,400 | 75.2% / 75.0% | 1.8% | 60% / 60% |
| Dynamic allocation | $12,100 | $10,400 | $4,300–$17,400 | $12,100 / $10,300 | 75.3% / 75.0% | 3.1% | 73% / 87% |

### Stock return 5% from year 10, noticed at once

| Saving plan | Success | ± s.e. | Avg. saving, years 0–9 | Avg. saving, years 10–19 | 90th pct of avg. saving |
|---|---|---|---|---|---|
| Fixed $10,000 | 51.1% | 0.16% | $10,000 | $10,000 | $10,000 |
| Fixed $12,100 (the 75% plan) | 65.9% | 0.15% | $12,100 | $12,100 | $12,100 |
| Reviewed every year | 69.4% | 0.15% | $9,900 | $14,400 | $19,000 |

Review rule: each year save the smallest amount (between $0 and $40,000) whose reported success reaches a target, with the allocation re-optimised for that amount. The target is set so that the realised success is 75%: 63.7% for the dynamic allocation, 65.0% for 60/40 (lower than 75% because the rule steps in again in later years). Fixed saving amounts are the bisection results of the saving curve. Present value: the level yearly amount with the same value discounted at 3%. After the change, all plans re-solve the allocation; the review rule also re-solves the saving.
