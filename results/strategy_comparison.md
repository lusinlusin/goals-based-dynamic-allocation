## Strategy comparison — base case, 100,000 forward paths

| Strategy | Success | ± s.e. | 10th pct | Median | 90th pct | Avg. wealth if missed | CE (γ=2) | CE (γ=5) | Avg. stock |
|---|---|---|---|---|---|---|---|---|---|
| DP · goal probability | 62.2% | 0.15% | $296K | $543K | $677K | $357K | $460K | $359K | 81% |
| DP · log utility | 54.4% | 0.16% | $292K | $527K | $1,003K | $364K | $478K | $356K | 100% |
| DP · power utility γ=2 | 54.3% | 0.16% | $292K | $526K | $994K | $365K | $478K | $356K | 100% |
| DP · power utility γ=5 | 37.8% | 0.15% | $323K | $457K | $659K | $394K | $443K | $397K | 63% |
| DP · power utility γ=10 | 16.2% | 0.12% | $324K | $412K | $530K | $395K | $406K | $384K | 36% |
| Fixed 60/40 | 39.7% | 0.15% | $316K | $461K | $692K | $389K | $445K | $391K | 60% |
| Fixed 100% stock | 54.4% | 0.16% | $292K | $527K | $1,003K | $364K | $478K | $356K | 100% |
| Fixed 100% bond | 1.2% | 0.03% | $294K | $355K | $430K | $357K | $351K | $340K | 0% |
| Glide path 90→30 | 36.0% | 0.15% | $322K | $451K | $656K | $393K | $439K | $396K | 60% |

Goal $500,000 in 20 years from $50,000 plus $10,000 a year (needs 5.6% a year). DP in-sample estimate for the goal policy: 62.1%; forward simulation: 62.2% ± 0.15%. CE = certainty equivalent of terminal wealth.
