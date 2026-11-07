# Offline validation

Run Python commands in the `FRESH-Drive` Conda environment. `val.py` evaluates
causal expert-log replay, not fixed-anchor forecasting or closed-loop driving.

Like Plan-R1, validation runs through `Trainer.validate()`. `val.py` only parses
arguments, loads the checkpoint/data and starts Lightning. `ActionEvaluator` in
`metrics/evaluator.py` wraps the checkpoint: `validation_step` evaluates each
batch and `on_validation_epoch_end` deduplicates anchors, aggregates and exports
the results, then publishes means with `self.log`. `metrics/action.py` contains
the per-window error calculations. Training hooks and checkpoint keys are
unchanged. Action reports use a single device so exact quantiles and log bootstrap
see the complete dataset. For action validation, labels/annotations stay on CPU;
only model inputs transfer to the selected device.

- `ade_m`: valid continuous-prefix ADE, retained for compatibility.
- `ade_full_horizon_m`: ADE restricted to complete configured horizons; use this
  with `fde_m` for fixed 1 s tables. Missing endpoints are not zero error.
- Heading errors use wrapped angles in radians. Position errors are metres.
- `quantiles` contains per-anchor P50/P90/P95. `event_macro_average` averages
  supported event means equally; `event_macro_support` records the group count.
- Global summaries and baseline comparisons count each `(recording, anchor)`
  once. Each label counts the anchor once in its own event group. CSV retains
  manifest rows, and prediction files retain batch order.
- `paired_baseline_comparison` reports baseline-minus-model improvements and
  win/tie fractions. Missing values are paired out. For paired log bootstrap CIs,
  set `evaluation.log_map: mapping.json`, mapping full recording/DB names to original
  log IDs obtained from authoritative metadata. No filename-based log inference
  or log-level CI is performed without this map. At least two logs are required.

## Collision proxy

Enable with `evaluation.ego_footprint: [front, rear, left, right]`. All four values are positive
metres from the **trajectory ego-pose origin**, not necessarily the vehicle
centre: +x is front, +y is left. Use verified vehicle dimensions/reference offsets;
no default dimensions are assumed. When this option is null, collision scores
are null, while annotation coverage is still reported.

The proxy uses a separating-axis test on oriented rectangles, including boundary
contact. Predictions are transformed from the trajectory annotation anchor frame
into each object's reference frame. It evaluates all annotated categories at
discrete target times; it does not perform swept collision detection, distinguish
fault, or simulate other vehicles' responses.

`collision_proxy_any` is the fraction of complete annotated windows with at least
one overlap. `collision_proxy_step_fraction` is the average overlap fraction in
those same complete windows. `collision_proxy_step_N` evaluates the Nth endpoint
on its own available support. Annotation and evaluated coverage are exported
separately. Invalid, missing, out-of-tolerance, malformed or non-finite annotation
frames are not treated as clear. Valid empty object lists are clear. The maximum
absolute annotation/target timestamp difference is set by
`evaluation.max_annotation_age_us` (default 100000). Offline labels may precede or follow
the target within this tolerance; this does not permit future policy inputs.

Model, constant-action baseline and expert collision proxies are exported; expert
overlaps do not cause hidden sample filtering. This is a geometric log-replay
proxy, not official nuScenes CR, NAVSIM NC, or a closed-loop collision rate.

## Timing

`evaluation.warmup_batches` controls extra unmeasured first-batch forwards (default 1).
Exported batch latency measures whole-window cached-BEV forward only, excluding
image encoding, transfers and loading. It is neither per-decision nor system
end-to-end latency. `evaluation.warmup_batches: 0` is useful for a functional smoke run.

## Example

Set `trainer.ckpt_path` in the appropriate YAML, then run:

```bash
conda run -n FRESH-Drive python val.py --config config/val/pred.yaml
conda run -n FRESH-Drive python val.py --config config/val/plan.yaml
```

The CLI accepts only `--config` (default `config/val/pred.yaml`). Configure
checkpoint/device under `trainer`, data/loading under `datamodule`, and report
options under `evaluation`. Paths are relative to the project root. Use
`evaluation.max_samples` for smoke runs and `evaluation.output_dir` for output.
Both checkpoint paths default to null to require an explicit checkpoint choice.
The entry point fixes the random seed to 1024; bootstrap and exported metadata
use that same seed. YAML does not override it.

Action validation retains the detailed JSON/CSV and optional predictions.
Both evaluators share the JSON export in `Evaluator`; `val.py` only selects
`ActionEvaluator` or `RefreshEvaluator` and calls `trainer.validate()`.

## Action checkpoint diagnostics

Run `conda run -n FRESH-Drive python tools/evaluate_action_diagnostics.py --config
config/val/action_diagnostics.yaml` for paired visual/stale-input interventions
and fixed-context five-step forecasts. This separate Lightning evaluator leaves
the checkpoint, training code, `val.py` and `evaluation/action` unchanged.

The default selects up to two anchors per event label before global deduplication
using seed 1024. Set `evaluation.samples_per_event: null` for all validation
anchors. Results and exact selected sample identities are saved separately in
`evaluation/action_diagnostics/metrics.json` and `per_sample.csv`. Optional
`scratch_dir` copies only referenced BEVs/events to local storage to reduce
repeated reads; it does not regenerate data. Use a new/empty scratch directory
after changing source caches.

Branches include fresh replay, no BEV evidence, historical evidence without new
frames, real delayed inputs (100/200/400/800/1000 ms), three missing observations
then recovery, historical-motion initialization, and teacher-forced diagnostics.
Delayed tokens are aligned from their source pose to the causal decision pose;
pre-window data is unavailable. Delay labels refer to cached BEV query times,
not exact per-camera exposure ages. This measures robustness, not completed stale
input fine-tuning. No-vision is an inference ablation, not a separately trained
state-only model. Teacher-forced results use expert prefixes and must never be
reported as deployable performance. Historical-motion initialization changes
the information condition and is also diagnostic only.

The short-history fresh branch controls for the delayed branch losing the first
history frames. A zero-age delayed branch deliberately supplies incorrect age
metadata to diagnose age-embedding sensitivity; it is not a legitimate policy.

Five-step forecasts freeze visual memory at the anchor, reset decoder hidden,
previous action and step as `RollingPlanner.decode` does, and predict five
0.1-second actions. Compare only against the five-step arrived-observation and
kinematic branches, not the one-second aggregate. This is anchor candidate
acceptance; it does not replace full rolling-refresh evaluation.

`tools/analyze_action_results.py` analyzes existing JSON/CSV by event/recording
without rerunning inference. Add `--motion-groups` to read cached supervision
and group by historical speed/turn rate. Default speed bins are 0–0.5, 0.5–2,
2–5 and >=5 m/s; turning uses |yaw rate| >=0.1 rad/s. These are descriptive
diagnostic bins, not benchmark categories or test-set selection criteria.
Refresh validation reuses `CARSDirve.refresh_validation_metrics` (also called
by the model's training validation hook) and exports Lightning's
`val/loss`, `val/ade`, `val/refresh_rate` and `val/replan_rate` to `metrics.json`,
with the configuration and evaluated sample count. Its ADE uses available rolling
plan endpoints (up to 0.5 s), not action's 1 s horizon. Refresh validation uses
the same event-centre filtering as refresh training. It does not enable action
collision/bootstrap options. DataModule validation loads only the val manifest.

Add verified footprint values and an original-log map to enable collision and
log-level statistical evaluation. Formal validation still requires inspecting
annotation coverage and train/val log separation. Refresh utility U/W and oracle
regret require a separate three-branch evaluation; this action evaluator does not
infer those values from action predictions.
