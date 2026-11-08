# Offline validation

Validation measures causal expert-log replay with cached BEV features. Logged
future observations are not changed by the model's actions, so these reports do
not establish closed-loop driving performance.

## Commands

Set the data root and checkpoint path in the matching YAML, then run from the
project root:

```bash
conda run -n FRESH-Drive python val.py --config config/val/pred.yaml
conda run -n FRESH-Drive python val.py --config config/val/plan1.yaml
conda run -n FRESH-Drive python val.py --config config/val/plan2.yaml
conda run -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/plan1_comparison.yaml
conda run -n FRESH-Drive python tools/evaluate_refresh.py --config config/val/plan2_comparison.yaml
```

Checkpoint and dataset modes must agree. The planning configurations point to
their respective `lightning_logs/cars_plan1` or `cars_plan2` checkpoints. Report
export uses one device for complete aggregation and fixes the seed to 1024.
`evaluation.max_samples` limits a regular validation run;
`evaluation.samples_per_event` limits the paired comparison. Output paths are
configured independently for each experiment.

Planning training selects the top three checkpoints by minimum
`val/next_step_error`. Filenames contain epoch and error; there is no automatic
`best.ckpt` alias. Replace the validation YAML checkpoint path with the actual
`Best checkpoint` path printed at training completion. This monitor evaluates
the adopted plan's next 0.1 s endpoint; fixed10 and common-future reports remain
necessary to assess longer-horizon revision and selection. Full settings are in
the [experiment summary](../docs/CARS-Drive_实验设置汇总_2026-11-09.md).

## Predictor evaluation

`ActionEvaluator` evaluates the existing `pred` checkpoint on 31 frames, with 20
history steps and 10 sequential next-action outputs. Each output sees only
arrived observations and the configured causal action context.

- `ade_m` averages the valid continuous prefix. `ade_full_horizon_m` and `fde_m`
  require a complete configured horizon and are appropriate for fixed 1 s tables.
- Position errors are metres; heading errors are wrapped angles in radians.
- JSON summaries deduplicate `(recording, anchor_timestamp_us)`. CSV retains
  manifest rows; optional prediction files retain batch order.
- Event summaries average supported event groups. Paired constant-action
  comparisons use the last completed historical action as their baseline.
- Log-level bootstrap confidence intervals require an authoritative
  `evaluation.log_map`; filenames are not used to infer independent logs.

The optional collision proxy uses verified
`evaluation.ego_footprint: [front, rear, left, right]` dimensions and discrete
annotated object footprints. Missing annotations are excluded and coverage is
reported. It is not an at-fault or closed-loop collision metric. Timing measures
cached-BEV batch forwards after configured warmup, excluding image encoding,
loading and transfers.

## Planning evaluation

`BinaryRefreshEvaluator` serves both planning modes and reports their distinct
objectives. The full 41-frame cache has event anchor 25; this is separate from the
predictor's 20-step history. Initialization is at slot 20, followed by ten
decisions at slots 21..30. Each candidate has ten future endpoints, with targets
`[B,10,10,3]`. Action histories contain only completed logged motion through the
current decision; candidate actions never update the real history.

Candidate revision uses full-horizon coordinate position L1 plus weighted
wrapped-heading L1. Adoption compares the current candidate with the actually
retained plan on common absolute future endpoints. Deadline replanning is
excluded from adoption supervision. Historical valid candidates supply auxiliary
comparisons, with the retained source and self-pairs excluded.

- `plan1` reports keep/replace CE, replacement probabilities and selection
  statistics, plus auxiliary historical CE.
- `plan2` reports old/new predicted errors, true errors, quality L1, signed gain
  L1 and historical equivalents. Its decision uses predicted old-minus-new
  error above the configured epsilon; it has no probability, AUROC or CE report.
- Both report fixed5 and fixed10 candidate ADE/FDE, actual retained-plan error,
  next-step error, realized replacement gain, harmful replacements and decision
  regret. Retained-plan FDE can have a shorter horizon and must not be presented
  as fixed10 FDE.
- Error/gain labels use coordinate L1 with weighted heading L1. ADE/FDE remain
  Euclidean displacement metrics; their values are not interchangeable.
- Candidate counts and all online decoder calls are reported, including
  rejected candidates. Initial planning and diagnostic calls are separate.

The regular audit exports `metrics.json`, `per_decision.csv` and, when supported,
`history_preferences.csv`. Historical losses average pairs within each decision
before averaging supported decisions, so later decisions do not receive extra
weight solely because more old candidates remain available.

The paired comparison shares one frozen predictor and one trained adapter across
adoption policies. The essential controls are `all_fresh_base` (adapter disabled,
always adopt), `all_fresh` (adapter enabled, always adopt) and `learned` (adapter
enabled, learned adoption). These distinguish candidate revision from adoption
quality. Oracle policies use ground truth as diagnostic decision references;
they are neither deployable policies nor global optima. Each policy advances its
own retained state and plan.

Planning reports do not claim fewer candidate decoder calls. Event-relative
timestamps refer to dataset scenario labels, not verified physical event onset.
Formal conclusions still require reviewing event coverage, log separation and
the paired evidence-response controls.

## Additional predictor diagnostics

`tools/evaluate_action_diagnostics.py` retains the separate predictor-only
interventions configured by `config/val/action_diagnostics.yaml`: delayed or
missing BEV, frozen-context five-step forecasts and diagnostic expert prefixes.
Those five-step diagnostics do not replace the ten-step planning audit.
`tools/analyze_action_results.py` summarizes existing action reports without
rerunning inference. An optional scratch directory copies referenced cache files
only; use a fresh directory after source caches change.
