## Regimes that have to be inferred — 100,000 forward paths

| Policy | First state good (known) | First state bad (known) | First state unknown (long-run mix) |
|---|---|---|---|
| Sees the state | 72.4% (+2.7 pp) | 59.9% (+2.4 pp) | 66.1% (+2.6 pp) |
| Infers the state from returns | 70.2% (+0.4 pp) | 57.9% (+0.4 pp) | 63.9% (+0.4 pp) |
| Ignores the state (base case) | 69.8% | 57.5% | 63.5% |
| Fixed 100% stock | 61.9% (-7.8 pp) | 50.8% (-6.7 pp) | 56.2% (-7.2 pp) |
| Fixed 60/40 | 47.9% (-21.9 pp) | 37.8% (-19.7 pp) | 42.8% (-20.7 pp) |

Same two states as results/regime.md (stay probability 0.9). "Infers the state" updates P(bad state) from each year's stock return by Bayes' rule and solves the DP on (year, wealth, P(bad)) with 11 belief points; "sees the state" is told the state every year; "ignores the state" is the base-case policy. Brackets: paired difference against ignoring the state (s.e. 0.05 pp for inferring, 0.09 pp for seeing). In the first two columns the investor knows the first state; later states are hidden. The filter calls the state right in 68% of years; its average P(bad) is 0.58 in bad years and 0.42 in good ones. DP's own estimate, first state unknown: inferring 63.9% (actual 63.9%); ignoring 62.1% (actual 63.5%).
