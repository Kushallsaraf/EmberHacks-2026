# Spotter: a reproducible PyTorch squat-form experiment

This optional CPU experiment compares a small temporal CNN and a summary-statistic
MLP with Spotter's existing browser squat rules. It is separate from Flask and the
browser application. Label **0 = good**, **1 = bad**. Video clips and trained
models are not tracked; result files are in `ml/results/`.

## Results (Mendeley squats, front + diagonal, 25 subjects)

Subject-grouped five-fold CV (no person in both train and test), three seeds, 60
epochs, 1,297 clips with features. On the 567 clips where the app rules complete a
rep, scoring the same predictions:

| Method | Clips assessed | Accuracy (mean ± SD over seeds) | Bad-class F1 |
|---|---:|---:|---:|
| App rules (`movement.js` port) | 43.7% | 0.646 | 0.683 |
| MLP, trained on all clips | 100% | 0.667 ± 0.012 | 0.659 |
| CNN, trained on all clips | 100% | 0.641 ± 0.027 | 0.646 |

Over all 1,297 clips the MLP reaches 0.642 ± 0.010 and the CNN 0.620 ± 0.016
(`ml/results/all_clips/`). Trained and tested only on the 567 rule-assessable clips,
the MLP scores 0.641 and the CNN 0.593 (`ml/results/`). Either way the gains over the
rules are small and not significance-tested; the clearer difference is coverage, since
the models score every clip with features while the rules abstain on 56%. Both models
fit their training folds to 85–100%, so the limit is generalizing to unseen people
under weak labels, not capacity. Hyperparameters were not tuned against these folds.

Data may come from licensed public footage or personal recordings. Each clip needs
a documented source and label. Synthetic arrays in `tests/ml/` are test fixtures
and are excluded from experiment results.

## Free data without recording yourself

Candidate: [A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad
Form Labels, version 3](https://data.mendeley.com/datasets/kgbb3yn47p/3), DOI
`10.17632/kgbb3yn47p.3`. The publisher lists **CC BY 4.0**, 26 participants,
bodyweight squats among seven exercises, and participant/form/view identifiers.
Attribution: Duddela Sai Prashanth, Priya Kamath, Rakshith Bhandary, S Ashlesh Pai,
Pratham PN, Advith P, and Abhishek Mashetty (2026). See the original record for
the license and download. No paid API, account subscription, or cloud GPU is needed.

What the squat footage actually contains (inspected 2026-11): 150 videos, one per
subject (001–024, 026) × good/bad × front/side/diagonal; 15 fps, ~30 s, with
~10–20 back-to-back repetitions each. Labels are per video and do not name the
fault; spot checks show shallow depth in some bad videos. Faces and often torsos
are covered by privacy blur boxes. Front and diagonal videos of the same subject
appear to be simultaneous recordings of the same reps.

The provider archive has five pairs of byte-identical squat videos: subject 003
diagonal and subject 023 side are each filed as both good and bad; subject 005/006
side (good and bad) and subject 026 bad side / subject 015 good side are shared
between subjects. No re-encoded near-duplicates were found. `segment_reps.py`
excludes every copy, which removes subject 003's diagonal pair from front + diagonal.

Pose coverage differs by view. Front/diagonal frames mostly pass the both-sides
visibility filter in `features.py`; side-view frames almost never do (the far leg is
occluded), so side view needs one-sided features and is not used yet.

### Automated preparation (front + diagonal)

Fetch only the squat videos (~0.5 GB) from the 3.1 GB archive with HTTP range
requests:

```bash
.venv-ml/bin/python -m ml.download_dataset
```

Split each video into one-repetition clips and write `ml/data/manifest.csv`:

```bash
.venv-ml/bin/python -m ml.segment_reps
```

`segment_reps.py` finds reps from hip height above the ankles, cuts at the middle of
the standing interval between bottoms, and drops reps with under 35% measured
frames. It replaces only its own `mendeley_*` clips and manifest rows, and writes
each clip's source frame range to ignored `ml/data/segments.csv`. With duplicates
excluded this gives 1,319 clips (good 634 / bad 685) from 98 videos.

Run the app rules over each whole source video, as the browser does during a set:

```bash
.venv-ml/bin/python -m ml.video_rules
```

The browser tracker calibrates once and stays armed between reps, so a one-rep clip
cut from a continuous set has no fresh standing hold; assessing clips separately
left only 13.5% assessable. `video_rules.py` streams each source video through
`SquatRules(target_reps=100)` with extraction's MediaPipe settings and assigns each
completed rule rep to the clip covering at least half of it. This assessed 568 of
1,319 clips (43%); the rest fall in tracking dropouts, which reset the app's
calibration too. `ml.extract_features` uses `ml/data/video_rules.csv` when present
(refusing it if older than `segments.csv`) and per-clip rules for other clips, and
records `rule_source` in provenance.

Limitations of this preparation: every rep inherits its video's label (weak labels;
a bad video can contain acceptable reps), labels are not independently reviewed,
and both views of one performance count as separate clips. Subject-grouped CV keeps
all of a subject's clips in one fold, so this does not leak across folds, but the
clip count overstates independent observations.

### Manual preparation (other sources)

1. Download just one of the source's folder organizations: the publisher says all
   three contain identical videos. Keep originals under ignored `ml/data/downloads/`.
2. Select bodyweight squat footage; inspect the provider's labels and keep only
   labels appropriate to this experiment: shallow depth, inward knee movement, or
   excessive forward lean. Do not label videos using the rules being evaluated.
   Record whether a label is provider-supplied or independently reviewed. If a bad
   label describes a different fault, exclude it or explicitly redefine the task.
3. Trim to one full repetition per clip with a free local video editor. Include a
   stable standing hold of at least 0.4 seconds before descent and 0.3 seconds after
   returning to standing so the actual app baseline can assess it. Keep the body in
   frame. Do not duplicate the same repetition across views or downloads.
4. Put clips in the folders below. Aim for 30–60+ clips with balanced labels and
   multiple participants. Preserve participant IDs across sessions and views.
5. Copy the header from `ml/data/manifest.example.csv` into `ml/data/manifest.csv`.
   Add one row per clip with these columns:

| Column | Meaning |
|---|---|
| `clip` | Relative path such as `good/participant01_rep01.mp4` |
| `subject_id` | Stable, dataset-prefixed participant ID, shared by all that person's clips |
| `source_url` | Original dataset/video URL; for personal clips, `self-recorded` |
| `license` | Original reuse license, e.g. `CC BY 4.0`; for your clips, `owned` |
| `label_source` | Provider label/annotation reference, or reviewer and decision; include source filename and trim interval |

The manifest template contains column headers only.
Keep attribution and record your trimming/relabeling changes when sharing results.
Do not put names, contact information, or private signed download URLs in a tracked
manifest. The extractor checks completeness and byte-identical duplicates, not
license validity, identity, semantic duplicates, or label correctness.

Optional personal recordings follow the same folder/manifest layout. Use a fixed
front or side camera with the whole body visible, one repetition per clip, balanced
labels, and consistent lighting. Side views help depth/lean; front views help knee
alignment, but neither view measures all faults reliably. Avoid intentionally
performing painful or unsafe movements just to collect bad examples.

```text
ml/data/
  manifest.csv
  clips/
    good/  # one full repetition per video
    bad/
  features.npz                 # generated, ignored
ml/models/                     # generated, ignored
ml/results/                    # created only after actual evaluation
  results.json
  confusion_matrix.png
  results.md
```

## Setup and exact commands

Run from the repository root. Use **CPython 3.11** (tested); the pinned MediaPipe
release also publishes 3.12 wheels. Python 3.13/3.14 are not supported by this pinned
stack even if used by the Flask environment. Do not change that environment.
Node >=20 is needed for tests that execute the existing JavaScript rule tracker.

Create a separate environment:

```bash
python3.11 -m venv .venv-ml
```

Install the pinned experiment dependencies:

```bash
.venv-ml/bin/python -m pip install -r requirements-ml.txt
```

Run the experiment tests:

```bash
.venv-ml/bin/python -m pytest tests/ml -q
```

After adding real clips and the manifest, extract landmarks and features:

```bash
.venv-ml/bin/python -m ml.extract_features
```

Train/evaluate both models and the rules on the same folds, then save inference
models fitted on the full evaluated cohort. The default cohort keeps only clips the
rules assess, for every method:

```bash
.venv-ml/bin/python -m ml.train_form_classifier --seeds 17 42 123 --epochs 60
```

`--cohort all` trains and scores the models on every clip, scores the rules only
where they complete a rep (with coverage), and also scores the models on those same
clips for the comparison:

```bash
.venv-ml/bin/python -m ml.train_form_classifier --cohort all --results ml/results/all_clips --model-dir ml/models/all_clips
```

Predict a separate uniformly sampled landmark sequence saved as a NumPy array:

```bash
.venv-ml/bin/python -m ml.predict path/to/landmarks.npy --model ml/models/cnn.pt
```

Or use `from ml.predict import predict_form`. Input is `(N, 33, 3)` with columns
**x, y, visibility**, not x/y/z. Keep missing frame positions as NaNs. Output is
`(label, probability_of_that_label)`; softmax confidence is not calibrated.

Existing suites still run in the original application environment:

```bash
.venv/bin/python -m pytest -q
```

```bash
npm test
```

ML tests skip missing optional packages in the application-only environment; run
the dedicated ML command to actually validate them. Training and extraction print
a clear error when inputs are absent; they do not manufacture result files. Old
results from a previous run are not silently deleted on a failed rerun; inspect
their archive SHA-256 before attributing them to current inputs.

## Features and MediaPipe choice

The isolated installation uses **mediapipe 0.10.21 `mp.solutions.pose.Pose`**.
Its CPU graph and bundled full pose model are used with detection/tracking
confidence 0.65. A test initializes the actual API and processes a blank image.
On macOS, MediaPipe initializes a graphics context even with this CPU graph; a
restricted sandbox can block that initialization. Run from a normal local terminal.
The browser uses the Tasks Lite model instead: model/decoding differences can
change landmarks, so reproducing thresholds does not promise identical camera
outputs. This experiment does not use the legacy Flask prototype's detector.

For every decoded frame, `features.py` computes 2D left/right knee angles,
left/right hip angles, torso angle to upward vertical (0° upright), and knee-width
divided by ankle-width. Required landmarks: shoulders 11/12, hips 23/24, knees
25/26, ankles 27/28. Frames are invalid when any required joint is nonfinite,
out of image bounds, below 0.7 visibility, or has degenerate feature geometry.
No zero-filled poses, artificial labels, or extrapolated missing endpoints are used.

At least eight valid frames are required. Valid rows are linearly interpolated to
32 frames over their original timestamps (configurable with `--frames`). Skips and
counts are logged and stored. Large internal gaps can still be interpolated; inspect
valid-frame counts and footage before interpreting a result. Timestamp conversion
uses decoded frame index / reported FPS, so use constant-frame-rate clips; variable
frame-rate timing is not supported. Normalized x/y features are affected by aspect
ratio and camera angle; close/overlapping ankles make the width ratio unstable.

## Faithful rules versus six-feature models

The active app is `app/static/js/movement.js`, not the unfinished
`app/exercises/squat.py` prototype. Source references in `rule_baseline.py` refer to
repository commit `ad5176d`:

- Squat geometry/thresholds: lines 7–19; raw 3D angles: 50–66, 177–187.
- Starting calibration and timing: 36–45, 195–234.
- Departure, reversal, minimum excursion and completion: 237–351, 399–402.
- Faults: shallow depth >105°, asymmetry >20° sustained for >=150 ms,
  and cadence <1400 ms: 277–282, 353–365.
- Image framing gate: `app/static/js/calibration.js`, lines 37–79.

The Python port runs on **raw world landmarks and original timing**, retaining
the complete state machine for one rep (`SquatRules()`) or a continuous set
(`SquatRules(target_reps=n)`: completion keeps the calibration armed; any dropout
forces a full re-arm). Hand-built cases are passed to both Python and the unchanged
JavaScript module, including missing/degenerate frames and back-to-back sets.
These test fixtures are never training/evaluation data.

The rules have no knee-valgus or torso-lean criterion. No threshold is invented for
them. A baseline bad label means any actual app fault, which also includes cadence
and asymmetry beyond the dataset's target faults; this mismatch is a limitation.
The neural models receive only the requested six 2D features, so this comparison
is between systems with different inputs, not a controlled architecture-only study.

If the rule state machine cannot complete a repetition, its output is unavailable
(`-1` in the archive). It is not silently labeled good or bad. Such clips are excluded
from **all three methods** for the common-cohort comparison. Results include the
excluded names and original/evaluated class counts. This selection can favor the
rules; report coverage alongside accuracy. Clips with valid features remain in the
archive even if the rules cannot assess them.

## Training and evaluation

- CNN: Conv1d(6→16) + ReLU, Conv1d(16→32) + ReLU, global average pooling,
  dropout 0.2, linear 32→2. Train-only normalized-feature noise (SD 0.015) and a
  monotonic smooth time warp of at most 4% of sequence duration.
- MLP: min/max/mean/range for each feature (24 inputs), linear 24→32 + ReLU,
  dropout 0.2, linear 32→2. No augmentation.
- Adam, learning rate 0.001, weight decay 0.0001; fixed 60 epochs; batch <=16.
  All operations stay on CPU with deterministic Torch algorithms and fixed Python,
  NumPy and Torch seeds. Direct dependency versions are pinned; transitive package
  resolver changes and hardware differences can still affect reproducibility.
- Normalization is fitted separately within each training fold, including MLP
  summary statistics. Test rows never select epochs, normalization, or models.
  There is no hyperparameter search or validation-driven early stopping.
- Default split is participant-grouped: **leave-one-subject-out below 40 evaluated
  clips**, otherwise stratified five-fold group CV, repeated over three seeds.
  This deliberately adapts the original clip-level brief to public multi-subject
  footage. Group/class constraints may prevent perfectly balanced folds. Both
  classes must remain in every training fold; insufficient data fails clearly.
- `--split clip` implements the original leave-one-clip-out below 40 / stratified
  five-fold otherwise. Use only for an explicitly disclosed within-recorder study
  (for example, all footage from one person), never to claim unseen-person accuracy.
  Leave-one-out folds do not change between seeds; only model randomness changes.

`results.json` contains held-out predictions, fold memberships, mean and population
SD across seeds for accuracy and bad-class precision/recall/F1, pooled confusion
matrices, seeds, class counts, provenance/video hashes, feature archive hash, package
versions and Git commit/dirty state. Matrices have true rows and predicted columns,
ordered good/bad. Pooled counts include each clip once per seed; they are correlated
repeated predictions, not additional examples. A zero-denominator metric is 0.

`results.md` plainly states when neither model beats rules on mean accuracy. Higher
accuracy is not a significance test. Full-data inference checkpoints are not used
to calculate reported CV scores. Checkpoints, video, and archives are ignored; the
three small result files remain trackable, and are created only by a successful run.

## Limitations

Small samples, uncertain labels, one recorder or few subjects, camera/view imbalance,
correlated repetitions, and source-domain differences all limit generalization.
Clip stratification cannot fix subject leakage. A single frame angle is not a
clinical form assessment; the baseline itself is a demo heuristic. No clinical,
injury-prevention, or universal coaching claim is supported.

The reported results use weak, provider-level labels that were not independently
reviewed, a single public dataset recorded in a few gyms, and front/diagonal views
that likely duplicate each performance. Test-suite results verify implementation
behavior, not squat-form classification accuracy.
