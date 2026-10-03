"""Import the original local Text2HOI dataset config without redistributing data."""
import hashlib
import json
from pathlib import Path
import pickle
import re
import numpy as np
from .geometry import approach_intersection, box, pose
from .grasp_geometry import local_pad_geometry
from .artifacts import export_run, write_json


def resolve_config(config, project_root=None):
    import yaml
    path = Path(config).resolve()
    settings = yaml.safe_load(path.read_text())
    if "defaults" in settings:
        name = next((d["dataset"] for d in settings["defaults"] if isinstance(d, dict) and "dataset" in d), None)
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Main config needs a simple dataset name in defaults")
        path = path.parent/"dataset"/(name+".yaml")
        settings = yaml.safe_load(path.read_text())
    if not all(k in settings for k in ("name", "obj_root", "data_obj_pc_path")):
        raise ValueError("Dataset config requires name, obj_root and data_obj_pc_path")
    if project_root is None:
        configs = next((p for p in path.parents if p.name == "configs"), None)
        if configs is None:
            raise ValueError("Config outside a configs/ tree requires --project-root")
        project_root = configs.parent
    root = Path(project_root).resolve()
    resolved = {k: (root/str(settings[k])).resolve() for k in ("obj_root", "data_obj_pc_path")}
    return settings, path, resolved


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError("Point cloud must contain at least four finite XYZ points")
    if np.any(np.ptp(p, axis=0) <= 1e-7):
        raise ValueError("Degenerate object bounds")
    return p


def bootstrap_scene(name, cloud):
    """Authored geometry-only demo, never presented as recovered annotations."""
    p = finite_points(cloud)
    lo, hi = p.min(0), p.max(0)
    ext = hi-lo
    long = int(np.argmax(ext))
    center = (lo+hi)/2
    # A coarse occupied-cell union retains empty space better than one global box.
    cells = np.minimum(((p-lo)/ext*3).astype(int), 2)
    boxes = []
    for cell in np.unique(cells, axis=0):
        xyz = p[np.all(cells == cell, axis=1)]
        a, b = xyz.min(0), xyz.max(0)
        boxes.append(box((a+b)/2, np.maximum((b-a)/2, .0002), label="point_cloud_proxy"))
    region_hi = hi.copy(); region_hi[long] = lo[long]+.35*ext[long]
    receiving = box((lo+region_hi)/2, (region_hi-lo)/2, label="demo_receiving_zone")
    free_lo = lo.copy(); free_lo[long] = region_hi[long]
    unreserved = box((free_lo+hi)/2, (hi-free_lo)/2, label="unreserved_surface")
    direction = np.eye(3)[long]
    hand_center = center.copy(); hand_center[long] = lo[long]-.07
    side_axes = [i for i in range(3) if i != long]
    rotation = np.column_stack([np.eye(3)[side_axes[0]], np.eye(3)[side_axes[1]], direction])
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    hand = box(hand_center, [.025, .035, .015], rotation, "authored_hand_proxy")
    candidates = []
    for axis in range(3):
        close = min((i for i in range(3) if i != axis), key=lambda i: ext[i])
        slide = next(i for i in range(3) if i not in (axis, close))
        for sign in (-1, 1):
            z = -sign*np.eye(3)[axis]; y = np.eye(3)[close]
            r = np.column_stack([np.cross(y, z), y, z])
            for index, frac in enumerate(np.linspace(.1, .9, 5)):
                hit = center.copy()
                hit[axis] = lo[axis] if sign < 0 else hi[axis]
                hit[slide] = lo[slide]+frac*ext[slide]
                original = pose(hit, r)
                # Fit before selection: every mode scores the same corrected
                # candidates, and usage is raycast again on the corrected axis.
                clearance = float(np.linalg.norm(ext)) + .1
                origin = hit-z*clearance
                entry = approach_intersection(boxes, origin, z)
                fitted = original.copy()
                fit = {"status": "approach_miss", "original_T_object_gripper": original.tolist()}
                if entry is not None:
                    fitted[:3, 3] = entry + .018*z
                    section = local_pad_geometry(boxes, fitted)
                    fit["status"] = "empty_pad_window"
                    if section is not None:
                        fitted[:3, 3] += section["center_y_m"]*r[:, 1]
                        fit.update(status="proxy_pad_fit", insertion_m=.018,
                                   initial_approach_point_object=entry.tolist())
                candidates.append({"id": f"axis{axis}_{'plus' if sign > 0 else 'minus'}_{index}",
                    "T_object_gripper": fitted.tolist(),
                    "approach_ray_origin_object": (fitted[:3, 3]-z*clearance).tolist(),
                    "geometry_preparation": fit})
    return {"schema_version": "handover.scene.v1", "units": "m",
        "provenance": "Local configured object point cloud; generated proxy hand, 30 geometric candidates and receiving-zone annotation",
        "utterance": f"Pass the {name} to my right hand.",
        "intent": {"object_id": name, "human_region": "demo_receiving_zone", "robot_region": "unreserved_surface",
                   "hand": "right", "text2hoi_prompt": f"Grasp a {name} with right hand."},
        "object": {"id": name, "boxes": boxes, "surface_points": p.tolist(),
                   "usage_regions": {"demo_receiving_zone": [receiving], "unreserved_surface": [unreserved]}},
        "receiving_hand": {"center": hand_center.tolist(), "direction": direction.tolist(),
                           "palm_normal": direction.tolist(), "boxes": [hand]},
        "gripper": {"max_opening_m": .085, "geometry": "parallel-jaw box proxy",
                    "width_policy": "local_pad_proxy", "grasp_frame": "parallel_jaw_tip"},
        "candidates": candidates,
        "evaluation_split": "S0", "annotation_status": "generated geometry-only demo with S0 evaluation"}


def import_dataset(config, output, project_root=None, object_name="all", scale_to_m=1.):
    with export_run(output, ["dataset.json"]):
        return _import_dataset(config, output, project_root, object_name, scale_to_m)


def _import_dataset(config, output, project_root, object_name, scale_to_m):
    import trimesh
    settings, config_path, paths = resolve_config(config, project_root)
    if not np.isfinite(scale_to_m) or scale_to_m <= 0:
        raise ValueError("scale-to-m must be finite and positive")
    if settings["name"] != "han":
        raise ValueError("This importer supports the available han layout; other datasets need their own frame conventions")
    for path in paths.values():
        if not path.exists():
            raise ValueError(f"Configured dataset path is missing: {path}")
    # Original ObjectModel uses this trusted local pickle; no remote pickle loads.
    with paths["data_obj_pc_path"].open("rb") as stream:
        cache = pickle.load(stream)
    names = cache["object_name"]
    if object_name != "all" and object_name not in names:
        raise ValueError(f"Object {object_name!r} is absent from the configured cache")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    from .core import select_grasp
    from .report import write_report
    for name in names:
        if object_name != "all" and name != object_name:
            continue
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Unsafe object identifier in local cache")
        asset = (paths["obj_root"]/cache["obj_path"][name]).resolve()
        if not asset.is_relative_to(paths["obj_root"]):
            raise ValueError("Object path escapes configured obj_root")
        if not asset.is_file():
            missing.append({"object_id": name, "source_path": str(asset), "reason": "cached object file missing"})
            continue
        loaded = trimesh.load(asset, process=False)
        original = finite_points(loaded.vertices)*scale_to_m
        cached = finite_points(cache["obj_pcs"][name])*scale_to_m
        scene = bootstrap_scene(name, original)
        source = {"dataset": settings["name"], "config": str(config_path), "asset_path": str(asset),
                  "asset_sha256": digest(asset), "source_vertices": len(original),
                  "source_faces": len(loaded.faces) if hasattr(loaded, "faces") else 0,
                  "cached_points": len(cached), "scale_to_m": scale_to_m,
                  "bounds_m": [original.min(0).tolist(), original.max(0).tolist()]}
        scene["source_data"] = source
        # Exact cached cloud for Text2HOI; full-resolution source cloud for display.
        np.save(output/f"{name}_points.npy", cached, allow_pickle=False)
        np.save(output/f"{name}_surface.npy", original, allow_pickle=False)
        (output/f"{name}_scene.json").write_text(json.dumps(scene, allow_nan=False))
        selection = select_grasp(scene)
        (output/f"{name}_FS.json").write_text(json.dumps(selection, indent=2, allow_nan=False))
        write_report(scene, selection, output/f"{name}_FS.html")
        records.append({"object_id": name, **source, "scene": f"{name}_scene.json", "selection": f"{name}_FS.json",
                        "neural_point_cloud": f"{name}_points.npy", "selection_status": selection["status"]})
    manifest = {"schema_version": "handover.dataset.v1", "dataset": settings["name"], "units": "m",
                "config": str(config_path), "config_sha256": digest(config_path),
                "cache_sha256": digest(paths["data_obj_pc_path"]), "objects": records, "missing": missing,
                "annotation_status": "Original object point clouds with generated grasp/region annotations and S0 demo settings"}
    if not records:
        raise ValueError("No requested object files available")
    write_json(output/"dataset.json", manifest)
    return manifest
