# Release validation — 2026-09-30

## 0.7.0 integration validation

Post-release checks with the companion's
[published 0.10.0](https://github.com/Hanxin-Zhang/r2handoversim/releases/tag/v0.10.0)
(commit `3eb1c9e007df1d9782c91580672c48f62220034e`) also cover the benchmark
reachable-set condition and a simulator lifecycle correction:

- A multi-object recording run exposed stale PhysX hand-collider state between
  trials. The companion now creates a fresh USD/PhysX scene per trial and checks
  eight hand-surface locations before evaluation. All 33 original inputs
  (16 can, 16 screwdriver and one neural example) were replanned from scratch.
  Geometry, joint trajectories, five metrics and first failures match the earlier
  runs. Independent method-side checks preserve every audited method/receiver
  field and validate all 264 surface probes. Export verification passes for
  33 trials and 4,101 frames. The interrupted recording batch keeps its failed
  status separately from the completed replacement batch.
- The replacement 33-trial recording batch also completes and passes export
  verification. Every exported trajectory array equals the fresh-evaluation
  array exactly across all 4,101 frames; method geometry and numeric results
  are unchanged. Outcomes remain 24 successes, seven Plan first failures and
  two Affordance first failures. The local captioned review lasts 169.011 s.
  The recording presents the same 33 evaluated trials.
- Two additional can receivers use `sampling.require_reference_ik: true`,
  defining a shared reference-grasp reachable set before four-mode selection.
  Both initial proposals pass, with no rejected proposals in this small sample.
  The eight resulting S0 trials have five successes and three Plan failures;
  no method failure triggers receiver resampling. All eight resolved method
  audits, 64 surface probes and 476-frame export checks pass. These samples
  are recorded separately from the original bounded-receiver batch. They
  check the configured sampling and evaluation protocol. See [protocol differences](paper_details.md#companion-benchmark-protocol-differences).

The original release checks were:

- 55 CPU tests cover annotation frame conversion, numerical rotation repair,
  triangle ray intersections and clipped pad sections, independent asset contact
  validation, fixed receiver replay audits and earlier workflows. Python 3.11 /
  NumPy 1.26 and Python 3.10 / NumPy 2.2 runs are recorded locally.
- Restored all 6,827 local annotated candidates across 16 original objects and
  exported all 64 four-mode settings. Original OBJ triangles drive approach hits;
  this batch still uses explicitly labelled proxy pad widths and authored usage
  regions. All 64 selections survived companion conversion and method audit.
  With the separate whole-object feasibility filter enabled, 60/64 settings
  have a feasible method candidate; all four bottle modes correctly have none.
  This broad check uses proxy pad geometry.
- Independently rechecked all 48 calibrated bottle candidates. Three final
  approach rays miss; FS/A2 reject another 13 for the supplied usage region.
  With the original bootstrap hand, FS/A1 choose candidate 42 (43.235 mm), while
  A2/A3 choose candidate 1 (30.958 mm). These are candidate-selection results.
- Generated four fixed random left/right MANO receiver scenes (seed 27) in the
  companion and reran all four modes before conversion. These receiver batches
  use supplied MANO templates as method hand inputs. All 16 inputs preserve
  selected pose/aperture, the original object mesh, full hand mesh/boxes and
  fixed receiver/object world poses in the method audit. FS/A1 now select
  candidate 4; A2/A3 select candidate 1. All 16 resolved Isaac trials also pass
  the same geometry audit. They
  fail the companion Stability stage: whole-mesh projected width exceeds
  85 mm, so evaluation stops at Stability. The results record local contact
  aperture and whole-object width separately.
- Prepared 224 original can candidates against the USD pads (220 fits, four
  retained failures). Four fixed receivers with `object_projection` feasibility
  select candidate 21 for FS, 66 for A1, and 12 for A2/A3. Their whole-object
  widths are 78.859, 82.240 and 74.917 mm, while actual apertures are 67.763,
  73.482 and 70.409 mm. Isaac PhysX S0 replay completes all 16 trials:
  FS succeeds in 3/4, A1 in 1/4, A2/A3 in 4/4 each; the four other trials fail
  Plan. Results belong to this fixed-receiver local integration batch.
  All 16 resolved trials pass method audit, including independently recomputed
  whole-object widths. Earlier local-aperture can runs retain all 16 Stability
  failures separately.
- Prepared 558 original screwdriver candidates (553 fitted, five retained
  failures) and ran 16 S1 trials using explicit authored usage boxes and the
  same four-receiver protocol. FS/A2/A3 succeed in 4/4 each; A1 has two Plan
  and two Affordance first failures. Actual USD finger geometry participates
  in the latter check. This local integration batch uses authored usage boxes
  and explicit S1 assignments.
  All 16 resolved trials pass method audit, including split/provenance and
  fixed hand/object geometry. The companion verifies 2,518 recorded frames.
- Ran 1,000 CPU diffusion steps with the original verified H2O Text2HOI
  weights on can (seed 27), decoded licensed MANO, and selected calibrated
  candidate 12 under whole-object feasibility. The predicted hand and all
  asset-candidate bindings survive the neural bridge. Generated the paper
  Sec. III-B delivery target from the supplied authored skeleton; this is
  separate from the random-template receiver experiment. The original asset
  replay preserves the prediction, grasp and delivery target (audit equivalent),
  but fails Plan with collision/search failure: eight IK solutions are found,
  and endpoint checks reject object/hand, robot self and environment contacts.
  The original target and receiver remain fixed throughout evaluation.
- Wheel/sdist build and pass `twine check`. The installed wheel imports original
  candidates, executes projection-feasibility ablations, audits resolved S0/S1
  trials, and runs all bundled demos outside the checkout. Package dependency
  checks pass; archives exclude local objects, checkpoints and MANO assets.
- Original candidates and mesh files, licensed MANO assets and derived replay
  records remain in ignored local outputs.

## 0.6.0 release validation

- 42 tests pass in Python 3.11 / NumPy 1.26.0 and in a fresh Python 3.10
  environment with NumPy 2.2.6 and the dataset extra. Coverage includes local
  pad geometry, final-pose constraints/ranking, replay changes, failed reruns,
  checkpoint/prediction integrity and meaningful CLI exit codes.
- A clean wheel installation ran all three CPU demos and four-mode ablations
  outside the checkout with no Torch, MANO, trimesh or simulator installed.
  After adding the dataset extra, it imported all 16 configured objects,
  reported the one missing cache entry and exported all 64 paired settings.
- Every imported setting has a feasible proxy candidate. Widths range from
  5.28 to 84.61 mm. All 128 selected proxy contacts lie on both an object-box
  surface and the corresponding inner finger plane, with residual below
  1e-12 m. FS/A2 selections satisfy the recomputed approach-point region test.
  These checks validate the proxy contact geometry.
- Real original-weight Text2HOI inference ran 1,000 DDPM steps on CPU for the
  imported bottle (seed 0, right hand), decoded the local licensed MANO model,
  selected a grasp and produced an ergonomic delivery target. The output
  arrays are finite and prediction/checkpoint hashes are recorded.
- Read-only integration with benchmark 0.8.0 converted and evaluated the three
  bundled proxy scenes. All three passed its offline criteria and the new
  method replay audit. An imported bottle was correctly rejected by the audit:
  method local width 63.764 mm became benchmark global width 67.686 mm.
- Wheel and source distributions build successfully and pass `twine check`.
  The source archive includes documentation, examples and tests. Both archives
  include the required code licenses and exclude datasets, checkpoints, MANO,
  generated geometry and local outputs. Package requirements pass `pip check`.
- The geometry findings informed the integration contract in
  [geometry_audit.md](geometry_audit.md). Generated validation artifacts remain
  in ignored local output directories.

## 0.1.0 initial release

- Seven CPU unit tests passed: usage ablation, aperture filtering, scoring,
  malformed inputs, rigid transforms, oriented-box intersections and neural
  point-cloud preprocessing.
- All three bundled grasp-selection demos ran and exported HTML, JSON and
  point-cloud files.
- Original Text2HOI H2O contact, PointNet and TextHOM checkpoints loaded with
  strict key matching. A real 1000-step, one-frame right-hand inference ran on
  CUDA with the bundled bottle proxy cloud and seed 0. All output arrays were
  finite. Verified file hashes are in `verified_weights.json`.
- Neural test environment: Python 3.11, PyTorch 2.7.0+cu126, NumPy 1.26.0,
  NVIDIA RTX 4070.
- Editable installation and wheel build succeeded. Installed demos ran from
  outside the source directory; wheel contents included demo assets and the
  upstream Text2HOI license.
- A generated scene/selection pair was converted by the companion benchmark
  and successfully replayed in Isaac Sim 5.0.

## 0.2.0 follow-up

- Ten CPU tests now pass, including the interleaved 6D rotation convention,
  object-frame invariance and palm-normal flipping.
- Decoded the actual H2O coarse output with locally supplied MANO using Python
  3.11, NumPy 1.26, SMPL-X 0.1.28 and Chumpy 0.70.
- Grasp selection on the decoded hand produced a valid result. The exported
  scene/selection pair ran in Isaac Sim, showing the decoded mesh and evaluating
  its skeletal collision proxies. The bottle example passed all active criteria.
- MANO models and generated hand meshes are excluded from the source release.
- Version 0.2.0 wheel installed independently; all three CPU demos ran from
  outside the source directory.

## 0.3.0 paper-detail follow-up

- Fifteen CPU tests pass, including ray/surface intersection, incorrect supplied
  surface annotations, comfortable radius/height, 15-degree extension, pose
  composition, forward sign and antipodal minimum-angle rotation.
- The authored seated-receiver skeleton produced a full delivery target. The
  companion benchmark solved this target with numerical pose IK, generated a
  collision-checked path, and replayed it successfully in Isaac Sim 5.0.
- The source PDF's Fig. 3 and Sec. III-B formulas were visually checked; explicit
  frame/axis/position-anchor conventions are recorded in `paper_details.md`.
- Version 0.3.0 wheels were built and the new delivery/planning integration was
  exercised outside both source trees using independently installed packages.

## 0.4.0 configured-data follow-up

- Eighteen CPU tests pass, including YAML default resolution, exact cached-point
  preservation, source hashing, missing-cache-entry reporting and input rejection.
- The original main config selected han. Imported all 16 existing PLY point
  clouds (8192 points each) plus their cached 1024-point inputs without rescaling.
  Reported the stale eyeglasses4 entry. Original files/cache were not modified.
- Ran a real 1000-step original-weight Text2HOI prediction on the imported
  binoculars cloud with a left-hand prompt; decoded with local MANO and selected
  a grasp. The full original object cloud survived conversion and Isaac replay.
- The 0.4.0 wheel's dataset import and cross-package conversion/evaluation ran
  outside both source directories. Data/derived geometry remain excluded from Git.

## 0.5.0 runnable-workflow follow-up

- Twenty-three tests pass, including structured-intent rejection, paired ablation
  exports, failed-run status replacement and corrupt-download preservation.
- The one-command pipeline verified all three original checkpoint hashes, ran
  1000 diffusion steps on CUDA, decoded local MANO and selected a bottle grasp.
  Its manifest converted directly to an Isaac Sim mesh-collision replay; all
  active demo criteria passed and screenshot/animated USD exports completed.
- Reimported all 16 configured objects with complete demo region catalogs and
  exported all 64 FS/A1/A2/A3 selections plus the experiment manifest.
- Reproduced Chumpy 0.70's isolated-build missing-pip failure with modern pip;
  verified the documented no-build-isolation/no-deps installation path.
- Built and independently installed the 0.5.0 wheel. Structured-intent selection,
  catalog prompts, checkpoint verification and paired-ablation exports ran
  outside the source directories and connected to the installed benchmark.
