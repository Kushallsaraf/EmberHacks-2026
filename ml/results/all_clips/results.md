# Squat-form experiment

Higher mean accuracy than rules: mlp. This is not evidence of statistical significance.

| Method | Coverage | Accuracy mean ± SD | Bad precision | Bad recall | Bad F1 |
|---|---:|---:|---:|---:|---:|
| cnn | 100.0% | 0.620 ± 0.016 | 0.635 | 0.619 | 0.626 |
| mlp | 100.0% | 0.642 ± 0.010 | 0.661 | 0.626 | 0.643 |
| rules | 43.7% | 0.646 ± 0.000 | 0.613 | 0.772 | 0.683 |

Same predictions, scored only on clips where the rules completed a rep:

| Method | Accuracy mean ± SD | Bad precision | Bad recall | Bad F1 |
|---|---:|---:|---:|---:|
| cnn | 0.641 ± 0.027 | 0.633 | 0.660 | 0.646 |
| mlp | 0.667 ± 0.012 | 0.669 | 0.649 | 0.659 |
| rules | 0.646 ± 0.000 | 0.613 | 0.772 | 0.683 |

Split: subject. Evaluated clips by class: [629, 668].
Cohort: all clips with features; rules abstain on 730 (no completed rep) and are scored only where they answer.
Models and rules are scored on different clip sets in the first table; use the second to compare them.
Pooled matrices repeat each clip once per seed; these are not independent observations.
SD is population SD across seeds, not a confidence interval. No generalization claim.
