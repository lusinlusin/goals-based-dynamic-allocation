## How much the DP scenario set matters

| Scenario set | Scenarios | DP's own estimate: range over 8 draws | Std. dev. | Same policies, forward-simulated: range |
|---|---|---|---|---|
| Plain random | 500 | 55.7% – 71.7% | 6.3 pp | 62.01% – 62.20% |
| Plain random | 2,000 | 59.6% – 67.5% | 2.6 pp | 62.16% – 62.21% |
| Plain random | 8,000 | 60.0% – 64.4% | 1.7 pp | 62.18% – 62.23% |
| Moment-matched | 500 | 62.0% – 62.5% | 0.2 pp | 62.08% – 62.21% |
| Moment-matched | 2,000 | 61.9% – 62.3% | 0.1 pp | 62.18% – 62.21% |
| Moment-matched | 8,000 | 62.1% – 62.2% | 0.0 pp | 62.16% – 62.20% |

Each row: 8 different scenario draws, each giving a policy and the DP's own estimate of its success probability; every policy is then scored on the same 100,000 forward paths.
