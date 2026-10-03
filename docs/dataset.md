# Import the original local dataset

`Text2HOI/configs/config.yaml` selects `dataset: han`; the dataset YAML specifies:

```yaml
name: han
root: data/han
obj_root: data/han/object
data_obj_pc_path: data/han/obj.pkl
```

The importer resolves these paths relative to the original project root, not
your current working directory. It accepts either the main config or
`configs/dataset/han.yaml`. For configurations moved outside a `configs/` tree,
pass `--project-root /path/to/Text2HOI` explicitly. The original local pickle is
loaded like the project's `ObjectModel`; only use your trusted dataset cache.

## Available data verified on 2026-09-30

16 PLY files exist: binoculars, bottle, bowl, can, cup, dispenser, drill,
eyeglasses, gamecontroller, hammers, headphones, knife, pincer, scissors,
screwdriver and toothbrush. Names, including `hammers`, are retained verbatim.
Every PLY has 8192 vertices and **zero faces**: these are point clouds, not
triangle meshes. The cache contains 1024 points per object. Coordinates are used
in metres, consistent with the original han loader, without recentering or
rescaling. Use explicit `--scale-to-m` only for a differently scaled input.

The cache also lists `eyeglasses4`, but its PLY is missing. It is reported in
`dataset.json`, not silently substituted with `eyeglasses`. Source hashes and
bounds are recorded in [the inventory](dataset_inventory.json). Use the
matching project root and frame convention for each configured dataset.

## Import and view

```bash
python -m pip install -e '.[dataset]'
intent-handover dataset --config /path/to/Text2HOI/configs/config.yaml \
  --output outputs/han_dataset
```

For each available object the command writes:

- `*_points.npy`: cached 1024-point input for Text2HOI, preserving coordinates.
- `*_surface.npy`: complete original PLY point cloud.
- `*_scene.json`: source cloud, collision proxies and explicitly generated demo annotations.
- `*_FS.json` / `*_FS.html`: grasp-selection result and source-point-cloud view.
- `dataset.json`: config/cache/asset hashes, bounds, paths and missing entries.

The importer does not modify the source data or fix the stale pickle in place.
It retains each available file and records the missing one. Raw/derived data
goes under ignored `outputs/`; the repository distributes the loader, not the
third-party dataset. No download credentials or developer path is baked into
the runtime.

## What is original and what is generated

The source object clouds and cached neural inputs are original local data.
The importer adds a geometry-only bootstrap: 30 axis-based grasp proposals,
a hand proxy outside the long-axis end, and a receiving zone covering the
lower 35% of that axis. Generated annotations are labelled in each scene.

Collision geometry is a union of boxes fitted to occupied cells in a 3x3x3
grid of the original point cloud. Selection uses ray/box surface intersections
and the paper's width/avoidance score. Each approach proposal is
intersected with that proxy, inserted 18 mm along +Z, and centred across the
local pad section **before** any mode is selected. No lateral X search or
mode-specific repair is applied. Failed proposals remain visible and are
rejected by selection. Original transforms are retained in
`geometry_preparation`. The same 30 prepared candidates enter all modes;
region intersections and avoidance costs use their final transforms.

These scenes explicitly request `local_pad_proxy` width, using the release's
parallel-jaw finger footprint. This improves insertion and local aperture
semantics without introducing Robotiq link calibration into the method package.
The full source cloud is used for display. Contact points use the occupied-cell box geometry. For mesh-based contact
measurements, use the benchmark asset adapter for original OBJ fitting and
USD pad verification, following the [integration contract](geometry_audit.md).

`evaluation_split=S0` is the default **demo setting**. Supply your annotated
`usage_regions`, candidates and split through the scene contract for task-specific
evaluation.

## Use actual data with the original neural weights

The main config's example mentions binoculars. A verified left-hand example:

```bash
intent-handover text2hoi --checkpoints checkpoints/h2o \
  --point-cloud outputs/han_dataset/binoculars_points.npy \
  --prompt "Grasp binoculars with left hand." --hand left \
  --device cuda --frames 1 --seed 0 --output outputs/han_neural
intent-handover from-prediction \
  --scene outputs/han_dataset/binoculars_scene.json \
  --prediction outputs/han_neural/prediction.npz \
  --mano-models /path/to/mano/models --output outputs/han_predicted
```

See [neural setup](neural_pipeline.md) for optional dependencies/weights. This
uses original H2O Text2HOI weights on a local han object.
The generated hand replaces the proxy; object points remain in their original
frame.

## Restore locally available original candidates

A separate local source tree contains `panda_handover_obj.pkl` with 6,827
region-grouped Panda base poses across the same 16 objects, plus original OBJ
meshes. These are separate from the configured point-cloud cache above.
Import trusted local files (pickle inputs must be trusted):

```bash
intent-handover import-grasps --scene outputs/han_dataset/bottle_scene.json \
  --annotations /path/to/Text2HOI/panda_handover_obj.pkl \
  --control-points /path/to/Text2HOI/data/gripper/gripper_control_points/panda_gripper_coords.pickle \
  --mesh /path/to/Text2HOI/data/han/meshes/bottle.obj \
  --output outputs/annotated_bottle
```

The importer preserves all proposals in archive order, source hashes, raw
transforms, part names and frame conversion. The source control points define
+Y closure, +Z approach, fingertip origin and pad-centre score reference. Only
small numerical rotation errors are repaired. Default Y centring is performed
before selection; `--no-center` only converts the frame. No 18 mm insertion is
added to these existing annotated grasps. With `--mesh`, approach intersections
use triangles, while width remains explicitly `local_pad_proxy` until the
companion prepares the actual asset pads. Input mesh units/frame must match the
scene; no alignment or scaling is inferred.

Existing usage regions and receiving-hand geometry are retained. Candidate
part names are retained as source metadata; usage regions come from the input
scene. Source data and generated scenes remain local,
excluded from the package. See [asset integration](geometry_audit.md) for the
preparation → fixed receiver → four-mode selection → replay workflow.
