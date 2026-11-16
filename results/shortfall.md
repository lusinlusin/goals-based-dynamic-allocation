## Penalising the size of a miss — 100,000 forward paths

### Chance of reaching the goal when the 10th-percentile outcome must be at least …

| Strategy family | 10th pct ≥ $305K | 10th pct ≥ $310K | 10th pct ≥ $315K | 10th pct ≥ $320K |
|---|---|---|---|---|
| Squared shortfall penalty | 60.3% | 58.7% | 56.8% | 54.6% |
| Linear shortfall penalty | not reached | not reached | not reached | not reached |
| Goal with a stock cap | 56.0% | 51.1% | 44.4% | 32.8% |
| Power utility | 51.7% | 50.7% | 46.3% | 41.3% |
| Fixed mix | 49.2% | 45.9% | 40.9% | 31.1% |

### Every strategy

| Strategy | Success | 5th pct | 10th pct | Median | Avg. wealth if missed | Expected shortfall | RMS shortfall | Avg. stock | Stock in last 5 years, paths that miss |
|---|---|---|---|---|---|---|---|---|---|
| Linear penalty, η = 0 | 62.2% | $251K | $296K | $543K | $357K | $54K | $103K | 81% | 95% |
| Linear penalty, η = 1 | 62.2% | $252K | $297K | $541K | $359K | $54K | $102K | 81% | 93% |
| Linear penalty, η = 2 | 62.0% | $252K | $298K | $539K | $360K | $53K | $102K | 80% | 93% |
| Linear penalty, η = 4 | 61.7% | $253K | $299K | $536K | $363K | $53K | $101K | 79% | 91% |
| Linear penalty, η = 8 | 60.7% | $254K | $301K | $530K | $369K | $52K | $100K | 78% | 87% |
| Linear penalty, η = 16 | 59.2% | $255K | $303K | $523K | $375K | $51K | $99K | 77% | 83% |
| Linear penalty, η = 32 | 57.2% | $257K | $305K | $517K | $382K | $50K | $98K | 75% | 79% |
| Squared penalty, η = 2 | 62.1% | $252K | $297K | $542K | $359K | $54K | $102K | 81% | 94% |
| Squared penalty, η = 4 | 62.0% | $253K | $298K | $540K | $360K | $53K | $102K | 80% | 93% |
| Squared penalty, η = 8 | 61.6% | $254K | $300K | $535K | $364K | $52K | $101K | 79% | 90% |
| Squared penalty, η = 16 | 59.5% | $258K | $308K | $526K | $372K | $52K | $99K | 76% | 84% |
| Squared penalty, η = 24 | 56.0% | $263K | $317K | $516K | $383K | $52K | $96K | 74% | 76% |
| Squared penalty, η = 32 | 53.1% | $267K | $323K | $509K | $389K | $52K | $95K | 73% | 70% |
| Squared penalty, η = 48 | 49.4% | $272K | $329K | $498K | $396K | $53K | $94K | 70% | 63% |
| Squared penalty, η = 64 | 46.9% | $274K | $332K | $492K | $400K | $53K | $93K | 69% | 59% |
| Goal, stock cap 90% | 58.7% | $260K | $302K | $532K | $365K | $56K | $102K | 76% | 86% |
| Goal, stock cap 80% | 54.5% | $270K | $307K | $517K | $373K | $58K | $101K | 70% | 77% |
| Goal, stock cap 70% | 49.1% | $278K | $312K | $497K | $380K | $61K | $100K | 63% | 68% |
| Goal, stock cap 60% | 42.6% | $286K | $316K | $470K | $386K | $65K | $101K | 56% | 59% |
| Goal, stock cap 50% | 34.6% | $293K | $320K | $446K | $391K | $71K | $102K | 48% | 49% |
| Goal, stock cap 40% | 25.3% | $298K | $322K | $427K | $394K | $79K | $106K | 39% | 40% |
| Power utility, γ = 2 | 54.3% | $249K | $292K | $526K | $365K | $62K | $108K | 100% | 99% |
| Power utility, γ = 3 | 50.6% | $271K | $311K | $503K | $379K | $60K | $100K | 86% | 69% |
| Power utility, γ = 5 | 37.8% | $294K | $323K | $457K | $394K | $66K | $98K | 63% | 41% |
| Power utility, γ = 7 | 26.7% | $300K | $325K | $432K | $397K | $76K | $102K | 49% | 30% |
| Power utility, γ = 10 | 16.2% | $303K | $324K | $412K | $395K | $88K | $110K | 36% | 21% |
| Power utility, γ = 15 | 10.1% | $301K | $320K | $398K | $390K | $99K | $117K | 27% | 20% |
| Fixed 30% stock | 15.4% | $300K | $321K | $409K | $392K | $91K | $112K | 30% | 30% |
| Fixed 40% stock | 24.4% | $298K | $321K | $426K | $394K | $80K | $106K | 40% | 40% |
| Fixed 50% stock | 32.8% | $293K | $320K | $444K | $393K | $72K | $103K | 50% | 50% |
| Fixed 60% stock | 39.7% | $286K | $316K | $461K | $389K | $67K | $101K | 60% | 60% |
| Fixed 70% stock | 44.9% | $278K | $311K | $479K | $384K | $64K | $102K | 70% | 70% |
| Fixed 80% stock | 49.0% | $269K | $306K | $495K | $378K | $62K | $103K | 80% | 80% |
| Fixed 90% stock | 52.1% | $259K | $299K | $511K | $371K | $62K | $105K | 90% | 90% |
| Fixed 100% stock | 54.4% | $249K | $292K | $527K | $364K | $62K | $108K | 100% | 100% |

Objective: maximise P(W ≥ goal) − η · E[((goal − W)⁺ / goal)^p], p = 1 (linear) or 2 (squared); η = 0 is the goal objective. Expected shortfall = E[(goal − W)⁺] over all paths; RMS shortfall = its root mean square. Matched table: success read off each family's efficient side, by linear interpolation between its points.
