# Start with a demo

[← Project overview](../README.md)

## 1. Install

Use Python 3.10 or newer. Run these commands in a terminal:

```bash
git clone https://github.com/robot-future/intent-handover.git
cd intent-handover
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, replace the activation command with `.venv\Scripts\activate`.
The CPU demos need NumPy only. Run subsequent commands from this repository
with the environment active.

## 2. Open the demos

```bash
intent-handover demo
```

Open these files directly in your browser:

| Object | File |
| :--- | :--- |
| Hammer | `outputs/demo/hammer_FS.html` |
| Screwdriver | `outputs/demo/screwdriver_FS.html` |
| Bottle | `outputs/demo/bottle_FS.html` |

Each report displays the object, receiving hand, selected gripper and candidate
scores. Green, orange and blue identify the object, hand and gripper. The
adjacent JSON files contain the scene and complete selection result.

To run one object:

```bash
intent-handover demo --object screwdriver --output outputs/screwdriver
```

## 3. Compare the four strategies

```bash
intent-handover ablate --output outputs/ablation
```

This creates 12 reports across the three bundled objects, `ablation.csv` and
`experiment.json` for paired benchmark conversion.

| Strategy | Human-usage filter | Avoidance ranking |
| :--- | :---: | :---: |
| **FS** | ✓ | ✓ |
| **A1** | — | ✓ |
| **A2** | ✓ | — |
| **A3** | — | — |

Width feasibility is checked in every strategy. A2 and A3 select the first
feasible candidate in input order.

## 4. Supply an intent

Try the included structured response against the hammer scene:

```bash
intent-handover select outputs/demo/hammer_scene.json \
  --intent examples/hammer_intent.json --output outputs/intent
```

For your own instruction, generate a prompt from the actual object catalog:

```bash
intent-handover prompt --scene outputs/demo/hammer_scene.json \
  --instruction "I need to hammer a nail." > outputs/prompt.json
```

Send the generated prompt to your chosen model, save its JSON response and pass
it through `--intent`. [Intent fields and validation →](workflows.md#structured-intent)

## 5. Connect your data and simulator

Choose the route that matches your inputs:

| Route | Required inputs | Guide |
| :--- | :--- | :--- |
| Local object data | Text2HOI dataset config and object files | [Import objects and grasp annotations](dataset.md) |
| Predicted receiving hand | Original H2O checkpoints and locally configured MANO | [Neural setup and pipeline](neural_pipeline.md) |
| Ergonomic target | Calibrated shoulder, elbow and wrist keypoints | [Execution formulas](paper_details.md#run-the-execution-formulas) |
| Original-asset replay | Isaac Sim, robot USD, object OBJ and receiving-hand configuration | [Prepare → select → replay → audit](workflows.md#original-candidates-real-asset-pads-and-fixed-receivers) |

For asset replay, prepare every candidate and establish each receiver **before**
running the four method modes. Keep the receiver and object target fixed through
conversion and replay.

## Useful checks

- If the console command is unavailable, use `python -m intent_handover` with
  the same arguments.
- A `no_feasible_grasp` result includes rejection reasons in JSON and HTML.
  Inspect the geometry, region and width settings there.
- Keep each run in a separate `--output` directory.

To run the CPU test suite:

```bash
python -m pip install -e '.[dataset]'
python -m unittest discover -s tests -v
```
