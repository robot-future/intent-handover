# Data contract v1

`handover.scene.v1` is a JSON object with `units: "m"`:

- `object.id`: string identifier.
- `object.boxes`: object geometry as a union of oriented boxes.
- `object.usage_regions`: mapping from semantic region id to a list of boxes.
- `intent.human_region`: a key in `usage_regions`; `object_id`, `robot_region`,
  `hand` and `text2hoi_prompt` document the structured intent.
- `receiving_hand.center`, `receiving_hand.direction`: hand reference point and
  nonzero wrist-to-finger direction, in the object frame.
- `receiving_hand.boxes`: optional-to-the-algorithm geometry used by the viewer
  and simulator conversion; required by the provided CLI viewer.
- `gripper.max_opening_m`: maximum jaw aperture.
- `candidates`: ordered objects containing `id`, `T_object_gripper` and
  `approach_point_object` (precomputed approach-axis surface intersection).
- `utterance`, `provenance`: display text and source description.

A box has `center` (3), `half_extents` (3 positive numbers), `rotation` (3x3
proper rotation), and an optional `label`. Geometry is expressed in the object
frame. Transforms are 4x4 row-major JSON arrays used with column vectors:
`p_object = T_object_gripper @ p_gripper`. Local gripper Y is the closing axis;
local +Z is the approach direction. TCP is at the finger tips.

`gripper.width_policy` defaults to `global_projection`: project all object
boxes onto the closing axis, preserving existing scenes. New dataset imports
use `local_pad_proxy`, described below. The usage check tests the approach
surface point; the separate benchmark affordance check tests finger-volume
intersection, so the two checks can differ.

`handover.selection.v1` contains `selected`, all evaluated `candidates`, `mode`,
`status`, `object_id`, and `provenance`. `selected` is null if no candidate is
feasible. Cost combines a unitless cosine and a distance in metres exactly as
in the paper; changing the geometry units changes ranking.

To replay in R2HandoverSim (installed separately):

```bash
r2handoversim from-intent --scene outputs/demo/hammer_scene.json \
  --selection outputs/demo/hammer_FS.json --output outputs/trial.json
r2handoversim demo --trial outputs/trial.json --headless
```

Conversion places the selected object/gripper relation at the demo's fixed UR5e
goal and positions the receiving hand relative to it. Supply `--delivery` for
a computed world-frame target.

## Optional predicted-hand geometry

`receiving_hand.palm_normal` is separate from the wrist-to-finger `direction`.
Both are in canonical object coordinates. `receiving_hand.mesh` may contain
MANO vertices/faces for the simulator viewer. The generated `boxes` approximate
the skeleton and remain the collision representation. The bridge also records
prediction frame and source checkpoint hashes.

## Computed approach intersections

A candidate may provide `approach_ray_origin_object` on its approach axis,
outside the object. The selector raycasts along gripper local +Z against the
union of oriented object boxes and computes the nearest forward surface hit.
This overrides `approach_point_object`; a miss rejects the candidate. The
selection records `approach_source`. The three bundled demos use this path.
Legacy candidates without the ray origin retain their supplied surface point.

`handover.skeleton.v1` and `handover.delivery.v1` are described in
[the execution recipe](paper_details.md).

## Configured dataset scenes

Optional `object.surface_points` contains original object-frame XYZ points.
The HTML report displays these points; selection width checks still use
`object.boxes`. `source_data` records paths, hashes, original counts and scale.
`annotation_status` identifies generated annotations. `evaluation_split` is
passed explicitly to the benchmark. Dataset import/export details and generated
annotation conventions are documented in [dataset.md](dataset.md).

## Local pad geometry

`gripper.grasp_frame` defaults to and currently only accepts
`parallel_jaw_tip`. The origin is the midpoint between the finger tips;
Y closes and +Z approaches. It is **not** a flange, wrist, Robotiq base, or
USD link origin. The avoidance distance uses `gripper.score_reference_point_gripper`
transformed into object coordinates as `p_g`. It defaults to [0,0,0] for legacy
scenes; annotated/calibrated scenes explicitly supply their pad centre.

With `width_policy: "local_pad_proxy"`, clip the object-box union to the
release proxy's entire pad footprint, X ∈ [-0.012, 0.012] m and
Z ∈ [-0.044, 0] m in the gripper frame. `width_m` is the span of all occupied
pieces along Y inside that window, including disconnected pieces. A broad
head outside the footprint no longer determines a narrow handle's aperture.
This policy measures the aperture of the clipped box geometry.

The section must be nonempty and centred about gripper Y=0 within 1 µm.
Empty/tangent sections (`empty_pad_window`) and offset sections
(`off_center_pad_section`) are rejected in **all four modes**. Missing section
width is `null`, not zero. An approach ray is mandatory for this policy.
For every policy, a supplied ray origin must be outside the object, behind
the TCP, and on its +Z axis. The region hit and avoidance cost are evaluated
using the final supplied pose; the selector never moves candidates.

Selection rows add `width_source`, `proxy_contact` (or null) and optional
`geometry_preparation` provenance. `proxy_contact.contact_points_gripper`
contains extrema of the clipped box union on the two inner pad planes. Its
`physical_grasp_verified` is false. Dataset candidates retain the original
pose and insertion in `geometry_preparation`; these are diagnostics, never
trusted as precomputed acceptance flags. The selector recomputes geometry.

Selection-level `grasp_contract` records `frame`, `closing_axis`,
`approach_axis`, `width_policy`, and `max_opening_m`. Consumers must preserve
the selected pose, aperture and contract or explicitly adapt and revalidate
them. Use the current R2HandoverSim integrated asset contract.
The integrated consumer carries explicit aperture and method evidence; use
`audit-replay` to verify the actual trial rather than inferring compatibility
from successful JSON loading.

## Triangle surfaces and asset preparation

Optional `object.mesh` contains finite `vertices` (N×3) and integer triangle
`faces` (M×3), in metres in the object frame. When present, an outside approach
ray is required and its nearest triangle intersection replaces the box hit.
Usage regions use the supplied box annotations. `receiving_hand.mesh` uses the same object frame.

`width_policy: "asset_mesh_pad"` requires the mesh and
`gripper.geometry_contract`. The companion's `handover.asset_gripper.v1`
contract contains the robot binding, canonical mesh digest,
`T_asset_tool_grasp_frame` (method fingertip coordinates → asset tool), and
`pad_window_gripper_m` X/Z bounds. The object/asset transform is therefore
`T_object_grasp @ inverse(T_asset_tool_grasp_frame)`.

Every candidate retains a `handover.asset_candidate.v1` preparation record,
including failures. Successful records bind the final pose, robot, mesh,
width, tool-frame transform, two contacts in both frames, and bilateral pad
distances. Method selection remeasures the full triangle section, verifies
its centring and contact positions, and rejects changed bindings or pad errors
over 0.2 mm. Selection never adjusts a calibrated candidate. `mesh_contact`
is distinct from `proxy_contact`; neither certifies force closure.

The selection contract also records `score_reference_point_gripper`,
`approach_surface`, and `geometry_contract`. The integrated trial preserves
these plus `method_selection` (selected row and mode), `gripper_opening_m`,
and `gripper_opening_source`. Aperture is in metres, not a linkage joint command.

For fixed receiver scenes, `receiver_protocol.policy` is `fixed_world`;
`receiver` records identity, side, seed, `T_world_hand` and `static_world`.
`target_T_world_object` fixes the shared object target. The hand geometry is
transformed into object coordinates before method selection. Replay auditing
checks this world target, receiver identity/pose, complete supplied hand
geometry, object mesh, selected grasp and aperture. Physics/planning are
separate benchmark validations.

## Width feasibility and actual aperture

`gripper.feasibility_width_policy` defaults to `opening` for compatibility:
`width_policy` determines both aperture and width feasibility. Set it explicitly
to `object_projection` when comparing with the companion's whole-object
Stability implementation. All four modes then also reject a full object span
over the aperture limit, using original mesh vertices when available and box
corners otherwise. The entire candidate set is filtered before ranking.

`selected.width_m` remains the actual pad aperture for control;
`selected.feasibility_width_m` records the separate width constraint. The
selection contract stores `feasibility_width_policy`. The audit compares a
resolved `stability_width_m` with the independently recomputed feasibility width
when object projection is requested. This preserves both physical contacts and
the shared evaluation rule without changing candidates in the simulator.

Replay audit also preserves an explicit `evaluation_split` and its optional
`evaluation_split_provenance`. Switching S1 to S0 would skip the Affordance
criterion and is a contract violation, even when all geometry stays identical.
