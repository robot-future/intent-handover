"""Complete user-facing workflows built on the same public method kernels."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np
from .core import MODES, select_grasp
from .demos import sample_surface
from .report import write_report
from .artifacts import export_run, write_json


def manifest_scenes(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != "handover.dataset.v1":
        raise ValueError("Expected handover.dataset.v1")
    scenes = []
    for row in data["objects"]:
        source = (path.parent/row["scene"]).resolve()
        if not source.is_relative_to(path.parent):
            raise ValueError("Scene path escapes dataset manifest directory")
        scenes.append(json.loads(source.read_text()))
    return scenes


def apply_intent(scene, intent):
    """Consume the documented VLM JSON output; never invoke a remote model."""
    if intent.get("needs_clarification") is not False:
        raise ValueError("Intent must explicitly resolve needs_clarification=false")
    if intent.get("object_id") != scene["object"]["id"]:
        raise ValueError("Intent object does not match the scene")
    regions = scene["object"]["usage_regions"]
    if intent.get("human_region") not in regions or intent.get("robot_region") not in regions:
        raise ValueError("Intent regions must exist in the scene catalog")
    if len(regions) > 1 and intent["human_region"] == intent["robot_region"]:
        raise ValueError("Human and robot regions must differ")
    if intent.get("hand") not in ("left", "right"):
        raise ValueError("Intent hand must be left or right")
    prompt = intent.get("text2hoi_prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.split()) > 20:
        raise ValueError("Intent needs a nonempty Text2HOI prompt of at most 20 words")
    result = deepcopy(scene)
    result["intent"] = deepcopy(intent)
    result["utterance"] = intent.get("instruction", prompt)
    return result


def export_selection(scene, output, mode="FS"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(scene["object"]["id"]))
    if mode not in MODES:
        raise ValueError("Mode must be FS, A1, A2 or A3")
    stem = output/f"{name}_{mode}"
    with export_run(output, [stem.name+".json"]):
        result = select_grasp(scene, mode)
        write_json(output/f"{name}_scene.json", scene)
        cloud = scene["object"].get("surface_points")
        np.save(output/f"{name}_points.npy", np.asarray(cloud) if cloud is not None else sample_surface(scene), allow_pickle=False)
        write_report(scene, result, stem.with_suffix(".html"))
        write_json(stem.with_suffix(".json"), result)
    return result, {"scene": f"{name}_scene.json", "selection": f"{name}_{mode}.json", "report": f"{name}_{mode}.html"}


def ablate(scenes, output):
    with export_run(output, ["experiment.json", "replay.json"]):
        return _ablate(scenes, output)


def _ablate(scenes, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    ids = [s["object"]["id"] for s in scenes]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", x) for x in ids):
        raise ValueError("Ablation requires nonempty, unique safe object ids")
    records, rows = [], []
    for scene in scenes:
        selections = {}
        for mode in MODES:
            result, files = export_selection(scene, output, mode)
            selections[mode] = files["selection"]
            chosen = result["selected"]
            rows.append({"object_id": scene["object"]["id"], "mode": mode, "status": result["status"],
                         "selected_id": chosen["id"] if chosen else None,
                         "valid_candidates": sum(r["valid"] for r in result["candidates"]),
                         "avoidance_cost": chosen["avoidance_cost"] if chosen else None})
        records.append({"object_id": scene["object"]["id"], "scene": files["scene"], "selections": selections})
    manifest = {"schema_version": "handover.experiment.v1", "objects": records, "modes": list(MODES),
                "scope": "Paired method ablations; all modes use the same object/hand inputs"}
    replay = {"schema_version": "handover.method_replay.v1", "record_kind": "settings_replay",
              "source": "Intent-Handover, method and FS/A1/A2/A3 ablation definitions",
              "experiment": "experiment.json", "paper_simulation_success_rate": None,
              "excluded": ["real-robot trial statistics", "participant questionnaires"],
              "settings": {mode: {"usage_constraint": flags[0], "avoidance_ranking": flags[1]}
                           for mode, flags in MODES.items()},
              "records": [{**row, "scene": f"{row['object_id']}_scene.json",
                           "selection": f"{row['object_id']}_{row['mode']}.json"} for row in rows]}
    with (output/"ablation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output/"replay.json", replay)
    write_json(output/"experiment.json", manifest)
    return manifest


def run_pipeline(scene, checkpoints, mano_models, output, device="cuda", seed=0, intent=None, skeleton=None):
    """One command: scene -> original-weight prediction -> MANO -> selection."""
    from .prediction_bridge import decode_prediction
    from .text2hoi import predict
    from .weights import verify_weights
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": "handover.pipeline.v1", "status": "running", "stage": "prepare", "seed": seed}
    def save(): write_json(output/"pipeline.json", state)
    save()
    try:
        scene = apply_intent(scene, intent) if intent is not None else deepcopy(scene)
        state["object_id"] = scene["object"]["id"]
        select_grasp(scene)
        hand = scene["intent"].get("hand", "right")
        if hand not in ("left", "right"):
            raise ValueError("Scene intent needs hand=left or right")
        model = Path(mano_models)/f"MANO_{hand.upper()}.pkl"
        if not model.is_file():
            raise ValueError(f"Missing MANO model: {model}")
        verify_weights(checkpoints)
        cloud = scene["object"].get("surface_points")
        np.save(output/"input_points.npy", np.asarray(cloud) if cloud is not None else sample_surface(scene), allow_pickle=False)
        state["stage"] = "text2hoi"; save()
        predict(SimpleNamespace(checkpoints=Path(checkpoints), point_cloud=output/"input_points.npy",
            prompt=scene["intent"]["text2hoi_prompt"], hand=hand, device=device, seed=seed, frames=1, output=output/"prediction"))
        state["stage"] = "mano"; save()
        scene = decode_prediction(scene, output/"prediction/prediction.npz", mano_models)
        result, files = export_selection(scene, output)
        state.update(files=files, selection_status=result["status"])
        if skeleton is not None and result["selected"]:
            from .execution import delivery_target
            target = delivery_target(scene, result, skeleton)
            (output/"delivery.json").write_text(json.dumps(target, indent=2, allow_nan=False))
            state["files"]["delivery"] = "delivery.json"
        state.update(status="succeeded", stage="complete"); save()
        return state
    except BaseException as exc:
        state.update(status="failed", error=str(exc)); save()
        raise
