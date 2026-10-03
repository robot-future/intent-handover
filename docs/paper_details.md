# Paper implementation

Source: **Intent-Handover: Grounding Language in Human-Usage Regions for
Trustworthy Robot-to-Human Handovers**, supplied manuscript `IROS26_3319_FI.pdf`,
Sec. III-A/B and Fig. 3. The table maps the method to executable components.

| Paper detail | Code | Implementation |
|---|---|---|
| Width and human-usage constraints, Sec. III-A.3 | `core.select_grasp` | Original triangle meshes and explicit width conventions are supported; human-usage regions use supplied labels |
| Approach-axis intersection `x_int` | `geometry.approach_intersection`, `mesh_geometry.mesh_approach_intersection` | Exact ray/triangle or ray/OBB intersection; legacy supplied points still accepted |
| Minimize cosine minus hand/gripper distance | `core.select_grasp` | Metres, equal coefficients as written in the paper |
| FS/A1/A2/A3 | `core.MODES` | First feasible input candidate is the explicit A2/A3 tie policy |
| Shoulder/torso centre and comfortable reach radius | `execution.delivery_target` | Calibrated skeletal keypoints supplied through the input contract |
| 15° extension and minimum-angle direction alignment | `execution.axis_rotation`, `minimum_rotation` | Explicit axis/facing conventions below |
| World/robot/object/gripper transform composition | `execution.delivery_target` | Calibration must be supplied; identity world-to-robot is the default |

## Run the execution formulas

From this repository:

```bash
intent-handover demo --object hammer
intent-handover delivery \
  --scene outputs/demo/hammer_scene.json \
  --selection outputs/demo/hammer_FS.json \
  --skeleton examples/seated_receiver.json \
  --output outputs/ergonomic_delivery.json
```

The example contains authored seated-person keypoints.
Input `handover.skeleton.v1` uses metres, world +Z up, left/right shoulder,
elbow and wrist coordinates, desk height, a facing hint and an extension axis.
All points must already be calibrated into the same world frame.

The implementation follows the paper's formulas:

- `c_sh = (left_shoulder + right_shoulder) / 2`.
- `c_t = [c_sh.x, c_sh.y, (c_sh.z + z_desk) / 2]`.
- `l_ua = ||elbow - c_sh||`, `l_fa = ||wrist - elbow||`.
- `r_star = sqrt(l_ua**2 + l_fa**2)`, `P = c_t + r_star * e_f`.
- `e_h = R_extension(15 degrees) * normalize(wrist - elbow)`.
- Minimum-angle `R` aligns the stored predicted wrist-to-middle-finger direction
  with `e_h`.

In particular, the upper-arm length uses the **shoulder midpoint**, as printed
in the manuscript, not the ipsilateral shoulder.

## Explicit coordinate choices

The implementation uses the following coordinate and calibration conventions:

1. `facing_hint_world` selects the forward sign of the horizontal perpendicular
   to the shoulder axis. An ambiguous hint is rejected.
2. `extension_axis_world` is a signed world-frame unit axis perpendicular to the
   forearm. Positive extension uses the right-hand rule; default magnitude is
   15°. This specifies the wrist flexion/extension plane without inferring it
   from shoulder points alone.
3. The canonical hand direction is the stored hand direction in object
   coordinates. The minimum rotation rotates the entire hand/object/grasp
   configuration together. Rotation around that direction is resolved by the
   minimum-angle rule, with a deterministic perpendicular axis at the 180° tie.
4. `P` anchors the **object origin**. The predicted hand centre retains its
   offset from that origin; it is not independently snapped to `P`.
5. `T_a_b` maps frame b to a. Thus `T_world_gripper = T_world_object *
   T_object_gripper`, and `T_robot_gripper = T_robot_world * T_world_gripper`.

Output `handover.delivery.v1` includes these poses, intermediate quantities,
keypoints and target direction for diagnosis. The benchmark evaluates target
reachability with numerical IK and collision checks.

## Send the target to Isaac Sim

In the separate benchmark's environment, with its optional planning extra:

```bash
python -m pip install -e '.[planning]'
r2handoversim from-intent \
  --scene /path/to/intent-handover/outputs/demo/hammer_scene.json \
  --selection /path/to/intent-handover/outputs/demo/hammer_FS.json \
  --delivery /path/to/intent-handover/outputs/ergonomic_delivery.json \
  --seed 0 --output outputs/ergonomic_trial.json
r2handoversim demo --trial outputs/ergonomic_trial.json --headless \
  --screenshot --animation --output outputs/ergonomic_isaac
```

The receiving configuration stays fixed at the computed world-frame target.
The robot starts at its home configuration and plans toward the fixed target.
Supplied body keypoints and target direction appear as visual references in
Isaac Sim.

Sec. III-A defines N grasp candidates. The local Panda annotation archive
supplies 6,827 proposals for 16 objects. The neural route uses original
Text2HOI pretrained weights.

## Local data

The original han config's 16 available object point clouds and neural input
cache can be imported directly with their original coordinates and source hashes. See [dataset import and provenance](dataset.md).

## Geometry audit

New imports prepare inserted, centred candidates and measure local proxy pad
width before applying the paper's region constraint and avoidance score. Existing
scenes retain global projection unless they opt in. Separate original OBJ assets
were found outside the configured point-cloud layout; they remain local and are
handled by the companion asset adapter. See [geometry audit](geometry_audit.md)
for the frame contract and replay validation steps.

## Original candidates and calibrated assets

The annotation importer explicitly converts the Panda base to the method
finger-tip frame using source control points. Its optional Y centring occurs
before selection; it adds no proxy insertion. Tiny source rotation errors are
projected onto SO(3) with the raw values retained. The avoidance reference can
be the measured pad centre instead of an implicit CAD origin.

For real-asset replay, the companion prepares all candidates first. The method
independently checks the final mesh width, approach intersection, usage region
and cost before selecting. A sampled receiving hand must also be fixed before
selection; moving or replacing it afterwards invalidates the result. See
[the integration contract](geometry_audit.md). The randomized receiver protocol
is an explicit benchmark condition shared by all four method modes.

## Companion benchmark protocol differences

The separate benchmark manuscript `IROS26_3330_FI.pdf`, Sec. III-A, requires
receiver SE(3) poses sampled from a reachable set and then fixed in world
coordinates. The companion's reference-IK-conditioned sampler implements an
explicit preselection reachable set; it does not resample failures of FS or an
ablation. Sampling bounds, seeds and proposal decisions are recorded.

The manuscript uses numerical Jacobian IK and RRT-Connect within MoveIt. The
validated companion uses numerical pose IK and a local RRT-Connect planner
with original-collider PhysX queries. Planner timings and outcomes are recorded
for this implementation. The 32 bounded-receiver integration trials and the
additional conditioned-sampling checks have separate records.
