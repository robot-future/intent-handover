# Text2HOI → MANO → grasp selection → Isaac Sim

The neural path is now connected to grasp selection. Use the same object cloud
and scene at every step. The bundled CPU demos provide a quick starting point.

## 1. Install optional decoding dependencies

Inside your neural Python environment:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[text2hoi,mano,download]'
python -m pip install --no-build-isolation --no-deps chumpy==0.70
python -m pip install 'git+https://github.com/openai/CLIP.git'
intent-handover download-weights
```

Chumpy 0.70 uses a legacy setup script that imports pip. Install it after the
extras with build isolation disabled, as above; the extras provide its runtime
dependencies. This avoids the missing-pip error in fresh environments.

Obtain MANO separately under its original terms. `--mano-models` must point to
the directory containing `MANO_RIGHT.pkl` and/or `MANO_LEFT.pkl`. These are
trusted local model files; no model or decoded mesh is bundled here. Legacy
Chumpy-based files are supported on Python 3.11 through temporary compatibility
aliases during loading; global NumPy/inspect names are restored afterwards.
The optional MANO extra pins NumPy below 2 for legacy compatibility.

## One-command workflow

After the installation above:

```bash
intent-handover pipeline --object bottle --checkpoints checkpoints/h2o \
  --mano-models /path/to/mano/models --device cuda --seed 0 \
  --output outputs/pipeline
```

Use `--scene outputs/han_dataset/bottle_scene.json` for an imported object
(check the exact path in `dataset.json`). `--intent response.json` accepts the
structured model response described in [workflows](workflows.md).
`--skeleton examples/seated_receiver.json` optionally adds ergonomic delivery;
use your calibrated keypoints for a real receiving target.

`pipeline.json` records the current stage, status and relative output paths.
Failures overwrite the previous run status, so an old successful run cannot be
mistaken for the current attempt. A completed prediction with no feasible grasp
returns a nonzero CLI exit code and records `selection_status` separately.
Standalone predictions also mark their metadata incomplete during reruns and
bind it to a SHA-256 of `prediction.npz`; the decoder rejects mismatched files.

In the benchmark environment:

```bash
r2handoversim from-pipeline \
  --pipeline /path/to/intent-handover/outputs/pipeline/pipeline.json \
  --output outputs/neural_trial.json
r2handoversim demo --trial outputs/neural_trial.json --headless \
  --hand-collision mesh --screenshot --animation --output outputs/neural_replay
```

If the pipeline includes a delivery target, install the benchmark's `[planning]`
extra: conversion then runs IK and trajectory planning. Otherwise it uses the
standard replay placement. The individual commands below remain available.

## 2. Generate an object and predict the receiving hand

```bash
intent-handover demo --object bottle
intent-handover text2hoi --checkpoints checkpoints/h2o \
  --point-cloud outputs/demo/bottle_points.npy \
  --prompt "Grasp a bottle with right hand." --hand right \
  --device cuda --frames 1 --seed 0 --output outputs/neural
```

## 3. Decode and select

```bash
intent-handover from-prediction \
  --scene outputs/demo/bottle_scene.json \
  --prediction outputs/neural/prediction.npz \
  --mano-models /path/to/mano/models \
  --frame 0 --mode FS --output outputs/predicted_hand
```

Outputs include `bottle_scene.json` with the predicted hand, `bottle_FS.json`
with ranked grasps and `bottle_FS.html` with a proxy visualization. The scene
contains MANO vertices/faces plus skeleton-based oriented boxes for collision
evaluation. Keep these generated assets out of a public repository unless their
redistribution terms permit it.

The bridge preserves Text2HOI's interleaved 6D rotation convention, converts to
MANO axis angles, decodes with zero shape coefficients and flat-hand mean, and
applies the inverse predicted object transform. Hand centre is the vertex mean;
direction is wrist-to-middle-fingertip. The palm normal comes from the
wrist/index/pinky-base triangle, with a left/right sign convention. Use
`--flip-palm-normal` if your input convention uses the opposite side of the palm.
The bridge uses the H2O checkpoint/object-transform convention.

## 4. Replay in the separate benchmark

In the benchmark/Isaac Sim environment (adjust paths to the method outputs):

```bash
r2handoversim from-intent \
  --scene /path/to/intent-handover/outputs/predicted_hand/bottle_scene.json \
  --selection /path/to/intent-handover/outputs/predicted_hand/bottle_FS.json \
  --output outputs/neural_trial.json
r2handoversim demo --trial outputs/neural_trial.json --headless \
  --screenshot --animation --output outputs/neural_replay
```

The simulator renders the decoded mesh and evaluates box proxies around the
predicted skeleton by default. Add `--hand-collision mesh` to evaluate safety
against static hand triangles in Isaac Sim. This procedural-scene workflow
uses box-based planning. For original-asset planning with PhysX queries, use
the [calibrated fixed-receiver workflow](workflows.md#original-candidates-real-asset-pads-and-fixed-receivers).

For method/replay contract checking, run [audit-replay](workflows.md#verify-replay-preserves-the-method)
on the benchmark's resolved trial. Use the current R2HandoverSim workflow and
prepare all asset candidates before selection to preserve the constraints
and ranking.
