# Original Text2HOI weights

The lightweight default demos need no checkpoints. This optional command runs
the original PointNet + contact CVAE + TextHOM DDPM with pretrained weights.
It exports coarse hand/object predictions and contact probabilities.

## Install

Use Python 3.10+ in a separate environment. Install a PyTorch build suitable for
your hardware, then install the adapter dependencies and official OpenAI CLIP:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[text2hoi,download]'
python -m pip install 'git+https://github.com/openai/CLIP.git'
intent-handover download-weights
```

The downloader fetches H2O `texthom.pth`, `pointfeat.pth`, and
`contact_estimator.pth` from the [original Text2HOI checkpoint folder](https://drive.google.com/drive/folders/1bfYF94-dVy-mA0n4cIRb_wI4ohPC6KK5).
If Google Drive throttles the download, obtain the three files manually and put
them in `checkpoints/h2o/`. Weights are not included in Git or relicensed here.
`intent-handover download-weights --verify-only` checks local files against the
bundled SHA-256 catalog. Downloads are checked before replacing any existing
file; a corrupt partial download is removed. Use `--force` to replace a corrupt
existing checkpoint. Both `text2hoi` and `pipeline` check all three hashes
before loading the models.
CLIP may download its own pretrained model on first use.

## Run

Supply a NumPy `.npy` array of shape N x 3, in metres, containing object surface
points in its canonical frame. Do not normalize it yourself: the adapter
preserves centre and scale for conditioning and samples 1024 points with FPS.

```bash
intent-handover demo --object bottle
intent-handover text2hoi --checkpoints checkpoints/h2o \
  --point-cloud outputs/demo/bottle_points.npy --prompt "Grasp a bottle with right hand." \
  --hand right --device cuda --frames 1 --seed 0 --output outputs/neural
```

Use `--device cpu` when CUDA is unavailable; the 1000 diffusion steps are slower.
`--frames` ranges from 1 to 150. One frame is a lightweight receiving-pose
example.
Prompts must fit 20 CLIP tokens (which can be more numerous than words);
overlong prompts are rejected rather than silently truncated.
The initial adapter targets H2O weights; other dataset architectures/configs
are not silently mixed. Checkpoint loading is strict and uses `weights_only=True`.

Outputs:

- `prediction.npz`: left/right hand parameters (1,T,99), object parameters
  (1,T,9), sampled object points, and contact probabilities (1,1024).
- `metadata.json`: prompt, hand, seed, device, representation and checkpoint
  SHA-256 hashes, run status, prediction hash and input-file hash.

Metadata is marked incomplete before each run and failed on errors. The MANO
bridge rejects unfinished runs and mismatched prediction hashes. Legacy
metadata without a status/hash remains accepted for older exports.

The 99D hand vector contains translation plus sixteen 6D rotations; it is not
the 48D MANO axis-angle vector. Use the optional `from-prediction` command to decode
MANO geometry, transform it into the canonical object frame, and run grasp
selection. Do not copy raw pose translations into `receiving_hand.center`: they
are expressed relative to the generated object pose. Follow the
[end-to-end recipe](neural_pipeline.md).

Architecture provenance and modifications are in [THIRD_PARTY.md](../THIRD_PARTY.md).
