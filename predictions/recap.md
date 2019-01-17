# Read in sequence!

# Config 1
- baseline
kaggle: 0.11695
```
DEV ERROR ~ Stats over 50 random splits with 0.4 test
> mean: 0.11192275363532786
> variance: 1.5256138733187423e-05
> stdev: 0.0039059107431158
```
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11168221607798523
> variance: 1.0453698389570521e-05
> stdev: 0.0032332179619646
```


# Config 2
- baseline
- tolta normalizzazione
kaggle: 0.11739
```
> DEV ERROR ~ Stats over 50 random splits with 0.4 test
> mean: 0.11003740141784149
> variance: 1.7945361238117644e-05
> stdev: 0.004236196553291365
```
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.4 test
> mean: 0.11026295057955048
> variance: 2.5398713583021285e-05
> stdev: 0.0050397136409741855
```
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.4 test
> mean: 0.10963627411613559
> variance: 2.189215709043647e-05
> stdev: 0.004678905544081486
```
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11062294251785826
> variance: 2.073832210393077e-05
> stdev: 0.004553934793552799
```

---
Even if here it doesn't work, on kaggle the normalization seems to do its job.
I'm keeping it.

So the new baseline is: Config 1
```
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11168221607798523
> variance: 1.0453698389570521e-05
> stdev: 0.0032332179619646
```
---


# Config 1 (= baseline)
-baseline
-Use KNN(k=5) (instead of KNN)
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11194258545968322
> variance: 1.5495886742605704e-05
> stdev: 0.003936481518133383

```

# Config 3
-baseline
-Use KNN(k=3) (instead of KNN)
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11210913689290448
> variance: 1.2995477834848458e-05
> stdev: 0.0036049241094437004
```

# Config 4
-baseline
-Use KNN(k=10) (instead of KNN)
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11137587981953732
> variance: 2.2371766413271385e-05
> stdev: 0.004729880169018173
```
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11227932548849218
> variance: 1.3393336336836537e-05
> stdev: 0.003659690743332903

```

# Config 5
-baseline
-Use KNN(k=15) (instead of KNN)
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11211156457270563
> variance: 1.5851549918990178e-05
> stdev: 0.003981400497185655

```


---
k = 10 seems to be the best. I'm keeping it.
So the new baseline is: Config4

CHanged some stuff in features enginnering
like dropped something, skewd something, etc.
Score (by eye) more or less the same, now regression.
---

# Config 6
- LassoCV
kaggle:0.11637
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11035489457257136
> variance: 1.4245850502199812e-05
> stdev: 0.003774367563208413
```

# Config 7
- ElasticNet
kaggle: 0.11632
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11008167982447647
> variance: 1.8893235030635165e-05
> stdev: 0.004346634908827191

```

# Config 7
- XGBoost
```markdown
DEV ERROR ~ Stats over 50 random splits with 0.5 test
> mean: 0.11637396240252552
> variance: 1.9177524118993793e-05
> stdev: 0.0043792150117337005

```

# COnfig 8
- Mean with Lasso Ridge e Elastic
kaggle:0.11604