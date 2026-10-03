# Runnable workflows

## Structured intent

Generate a prompt containing the actual object/region catalog:

```bash
intent-handover demo --object hammer
intent-handover prompt --scene outputs/demo/hammer_scene.json \
  --instruction "I need to hammer a nail." > outputs/prompt.json
```

Send the `system` and `user` fields to your chosen model and save its JSON reply.
The repository supplies the prompt; no paid service or API key is required for
the bundled examples. Try the included response directly:

```bash
intent-handover select outputs/demo/hammer_scene.json \
  --intent examples/hammer_intent.json --output outputs/intent
```

`select --intent` and `pipeline --intent` validate the object, region names,
hand side, short Text2HOI prompt and `needs_clarification=false`. Ambiguous
responses are rejected. Selection uses the specified human region as its
exclusion region; `robot_region` records the requested counterpart but is not
an additional hard inclusion constraint. Changing a prompt with `select` alone
does not generate a new hand; use `pipeline` for neural prediction.

For local datasets, use `prompt --manifest outputs/han_dataset/dataset.json`.
Imported regions use generated demo labels that can be replaced through the
scene contract.

## Four method ablations

```bash
intent-handover ablate --output outputs/ablation
# Or use every object imported from the original dataset config:
intent-handover ablate --manifest outputs/han_dataset/dataset.json \
  --output outputs/han_ablation
```

This executes FS/A1/A2/A3 on the same input scene for each object, writes an
HTML and selection JSON per mode, and exports `ablation.csv` plus
`experiment.json`. `--scene` can supply one custom or predicted-hand scene.

In the separately installed benchmark, with its `[planning]` extra:

```bash
r2handoversim from-experiment \
  --manifest /path/to/intent-handover/outputs/ablation/experiment.json \
  --seed 0 --output outputs/ablation_trials
r2handoversim demo --trials outputs/ablation_trials/trials.json --headless \
  --screenshot --render-every 6 --output outputs/ablation_isaac
```

The converter holds the world-frame hand and object target fixed within each
object across modes, then plans each selected grasp separately. Results include
planning failures. `conversion.json` separately lists any modes without a
feasible grasp; those have no replay and are not included in simulator metric
denominators. Read that file together with the simulator report.

These are executable method ablations, separate from the benchmark's authored
offline variants and the paper's four baseline systems.

Exports replace completion indexes with `handover.incomplete.v1` while running
and on failure. The complete `dataset.json`, `experiment.json`, `replay.json`
or selection schema is published only after its referenced outputs have been
written. Rerun the command to recover; do not run multiple writers in one output
directory. `select`, `demo`, and `from-prediction` return exit code 2 if any
requested scene has no feasible grasp; the rejection report is still saved.

## Verify replay preserves the method

R2HandoverSim saves the **resolved** `*_trial.json` after replay. Use that file, rather than the pre-adaptation input, to check the
actual replay configuration:

```bash
intent-handover audit-replay \
  --scene outputs/demo/hammer_scene.json \
  --selection outputs/demo/hammer_FS.json \
  --trial /path/to/r2handoversim/outputs/replay/hammer_intent_aware_trial.json \
  --output outputs/hammer_replay_audit.json
```

Add `--delivery outputs/ergonomic_delivery.json` to require the original
world-frame delivery target as well. Exit code 0 means the checked grasp,
annotations and supplied receiver geometry agree. Trajectory and collision
evaluation run in the companion benchmark. Exit code 2 records the
differences. The audit recomputes selection to catch stale method outputs.

The audit verifies the explicit-aperture contract. Calibrated asset replay additionally requires
`asset_mesh_pad`, full preparation evidence and resolved frame/contact fields.
For mesh or fixed-receiver scenes it checks the original object mesh, the full
supplied hand geometry, receiver identity/pose and fixed object target. Saved
avoidance scores must agree with recomputation even if the winning ID is
unchanged. Trajectory, physics and benchmark metrics remain separate checks.

## Original candidates, real asset pads and fixed receivers

After [importing original local candidates](dataset.md#restore-locally-available-original-candidates),
use the companion with its local asset configuration. `prepare-candidates`
requires the companion's Isaac Sim environment; both tools leave source assets
unchanged and write derived data under `outputs/`.

```bash
r2handoversim prepare-candidates \
  --scene outputs/annotated_bottle/bottle_scene.json \
  --asset-config /path/to/local_assets.json \
  --output outputs/calibrated_bottle_scene.json
r2handoversim receiver-scenes --scene outputs/calibrated_bottle_scene.json \
  --receiver-config /path/to/receivers.json --samples 4 --seed 27 \
  --output outputs/receiver_scenes
```

For the benchmark paper's reachable-set condition, configure the companion's
receiver sampler with `sampling.require_reference_ik: true` (enabled in the
public configuration). It checks a shared reference grasp's full-pose IK
before any method is selected and logs all sampling proposals. The chosen
method's own Plan criterion is evaluated separately. This is an explicit
operational definition of the reachable set, with recorded bounds and seeds. Older bounded samples remain separately labelled validation runs.

For **each** entry in the resulting `scenes.json`, use its scene path and a
separate output directory. Receiver scene entries are not dataset manifest
entries: do not pass this manifest to `ablate --manifest`.

```bash
intent-handover ablate --scene /path/to/one_receiver_scene.json \
  --feasibility-width-policy object_projection \
  --output outputs/one_receiver_methods
r2handoversim from-experiment \
  --manifest outputs/one_receiver_methods/experiment.json \
  --output outputs/one_receiver_trials
```

The explicit feasibility option filters by the same whole-object width used
by the companion Stability metric while preserving actual pad aperture for
control. A scene can therefore have no feasible grasp; retain this outcome.
Omitting the option preserves the scene's existing policy (default `opening`).

Use the current R2HandoverSim workflow. All four modes must see the same prepared
candidates, hand and object target. Do not apply `sample-receivers` after method
selection. Run the companion's Isaac replay and `verify-output`, then audit each
resolved trial against its receiver-specific scene and selection. Keep selection,
stability and planning failures in the results. See [the width-convention and
paper-protocol conventions](geometry_audit.md).

## Neural prediction through simulation

See [the complete setup and one-command recipe](neural_pipeline.md) for original
Text2HOI weights, MANO decoding, optional skeletal delivery and Isaac Sim replay.
