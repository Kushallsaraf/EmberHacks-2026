# Squat-form experiment

Neither neural model beats the rules on mean accuracy in this experiment.

| Method | Coverage | Accuracy mean ± SD | Bad precision | Bad recall | Bad F1 |
|---|---:|---:|---:|---:|---:|
| cnn | 100.0% | 0.593 ± 0.011 | 0.591 | 0.587 | 0.587 |
| mlp | 100.0% | 0.641 ± 0.014 | 0.647 | 0.611 | 0.628 |
| rules | 100.0% | 0.646 ± 0.000 | 0.613 | 0.772 | 0.683 |

Split: subject. Evaluated clips by class: [286, 281].
Rule-unavailable clips excluded from ALL methods: 730.
This common-cohort comparison can favor the rules by excluding difficult/incomplete reps.
Pooled matrices repeat each clip once per seed; these are not independent observations.
SD is population SD across seeds, not a confidence interval. No generalization claim.
