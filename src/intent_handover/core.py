"""Intent-aware candidate filtering and avoidance-cost ranking."""
from copy import deepcopy
import numpy as np
from .geometry import approach_intersection, contains, points, projected_width, transform, unit, vector
from .grasp_geometry import GRASP_FRAME, WIDTH_POLICIES, local_pad_geometry
from .mesh_geometry import asset_candidate_geometry, mesh_approach_intersection, mesh_digest

MODES = {"FS": (True, True), "A1": (False, True),
         "A2": (True, False), "A3": (False, False)}


def select_grasp(scene, mode="FS"):
    """No learned models or hardware required. All geometry is in object frame.

    Usage constraint checks the candidate's object-surface approach intersection.
    In A2/A3 the first valid candidate is selected deterministically in input order.
    """
    if scene.get("schema_version") != "handover.scene.v1" or scene.get("units") != "m":
        raise ValueError("Expected handover.scene.v1 with units=m")
    if mode not in MODES:
        raise ValueError("Mode must be FS, A1, A2 or A3")
    use_region, use_avoidance = MODES[mode]
    if scene["intent"].get("needs_clarification", False):
        raise ValueError("Intent is ambiguous; resolve it before selecting a grasp")
    if scene["intent"]["object_id"] != scene["object"]["id"]:
        raise ValueError("Intent and scene object ids differ")
    max_width = float(scene["gripper"]["max_opening_m"])
    if not np.isfinite(max_width) or max_width <= 0:
        raise ValueError("Maximum gripper opening must be positive")
    width_policy = scene["gripper"].get("width_policy", "global_projection")
    if width_policy not in WIDTH_POLICIES:
        raise ValueError("Unknown gripper width_policy")
    if scene["gripper"].get("grasp_frame", GRASP_FRAME) != GRASP_FRAME:
        raise ValueError("Expected parallel_jaw_tip grasp frame; adapt other TCPs explicitly")
    score_reference = vector(scene["gripper"].get("score_reference_point_gripper", [0, 0, 0]))
    mesh = scene["object"].get("mesh")
    feasibility_policy = scene["gripper"].get("feasibility_width_policy", "opening")
    if feasibility_policy not in ("opening", "object_projection"):
        raise ValueError("Unknown feasibility_width_policy")
    if width_policy == "asset_mesh_pad" and (mesh is None or not scene["gripper"].get("geometry_contract")):
        raise ValueError("Asset mesh pad policy requires object.mesh and gripper.geometry_contract")
    digest = mesh_digest(mesh) if width_policy == "asset_mesh_pad" else None
    hand = scene["receiving_hand"]
    direction, center = unit(hand["direction"]), vector(hand["center"])
    regions = scene["object"]["usage_regions"]
    intended = scene["intent"]["human_region"]
    if intended not in regions:
        raise ValueError(f"Unknown human region: {intended}")
    rows, seen = [], set()
    for candidate in scene["candidates"]:
        cid = candidate["id"]
        if cid in seen:
            raise ValueError(f"Duplicate candidate id: {cid}")
        seen.add(cid)
        t = transform(candidate["T_object_gripper"])
        resolved = "approach_ray_origin_object" in candidate
        if (width_policy == "local_pad_proxy" or mesh is not None) and not resolved:
            raise ValueError("Local pad or mesh geometry requires an approach ray for usage revalidation")
        if resolved and np.linalg.norm(np.cross(vector(candidate["approach_ray_origin_object"])-t[:3, 3], t[:3, 2])) > 1e-6:
            raise ValueError("Approach ray origin must lie on the gripper approach axis")
        if resolved and (t[:3, 3]-vector(candidate["approach_ray_origin_object"])) @ t[:3, 2] <= 0:
            raise ValueError("Approach ray origin must lie behind the TCP")
        if mesh is not None:
            hit = mesh_approach_intersection(mesh, candidate["approach_ray_origin_object"], t[:3, 2])
            approach_source = "ray/triangle intersection"
        elif resolved:
            hit = approach_intersection(scene["object"]["boxes"], candidate["approach_ray_origin_object"], t[:3, 2])
            approach_source = "ray/OBB intersection"
        else:
            hit = vector(candidate["approach_point_object"])
            approach_source = "supplied surface annotation"
        section, mesh_section, preparation_errors = None, None, []
        if width_policy == "local_pad_proxy":
            section = local_pad_geometry(scene["object"]["boxes"], t)
            width = section["width_m"] if section else None
        elif width_policy == "asset_mesh_pad":
            mesh_section, preparation_errors = asset_candidate_geometry(scene, candidate, digest)
            width = mesh_section["width_m"] if mesh_section else None
        else:
            width = projected_width(scene["object"]["boxes"], t[:3, 1])
        feasibility_width = width
        if feasibility_policy == "object_projection":
            feasibility_width = (float(np.ptp(np.asarray(mesh["vertices"]) @ t[:3, 1])) if mesh is not None else
                                 projected_width(scene["object"]["boxes"], t[:3, 1]))
        in_usage = hit is not None and any(contains(b, hit) for b in regions[intended])
        on_object = hit is not None and (mesh is not None or any(contains(b, hit) for b in scene["object"]["boxes"]))
        score_point = points(t, score_reference)
        distance = float(np.linalg.norm(score_point - center))
        cosine = float(unit(t[:3, 2]) @ direction)
        cost = cosine - distance
        rejected = list(preparation_errors)
        if not on_object:
            rejected.append("approach_ray_misses_object" if resolved else "approach_point_outside_object")
        if width_policy == "local_pad_proxy":
            if section is None:
                rejected.append("empty_pad_window")
            elif abs(section["center_y_m"]) > 1e-6:
                rejected.append("off_center_pad_section")
        if mesh_section is not None and abs(mesh_section["center_y_m"]) > 1e-6:
            rejected.append("off_center_mesh_pad_section")
        if width is not None and width > max_width + 1e-9:
            rejected.append("width_exceeds_aperture")
        if feasibility_policy == "object_projection" and feasibility_width > max_width + 1e-9:
            rejected.append("object_projection_exceeds_aperture")
        if use_region and in_usage:
            rejected.append("human_usage_region")
        rows.append({"id": cid, "T_object_gripper": t.tolist(),
                     "approach_point_object": hit.tolist() if hit is not None else None, "width_m": width,
                     "feasibility_width_m": feasibility_width,
                     "width_source": width_policy, "proxy_contact": section, "mesh_contact": mesh_section,
                     "geometry_preparation": deepcopy(candidate.get("geometry_preparation")),
                     "approach_source": approach_source, "score_point_object": score_point.tolist(),
                     "cosine": cosine, "distance_m": distance, "avoidance_cost": cost,
                     "in_human_region": in_usage, "valid": not rejected,
                     "rejection_reasons": rejected})
    valid = [r for r in rows if r["valid"]]
    best = min(valid, key=lambda r: r["avoidance_cost"]) if use_avoidance and valid else (valid[0] if valid else None)
    return {"schema_version": "handover.selection.v1", "units": "m",
            "object_id": scene["object"]["id"], "mode": mode,
            "status": "ok" if best else "no_feasible_grasp", "selected": best,
            "grasp_contract": {"frame": GRASP_FRAME, "closing_axis": "+Y", "approach_axis": "+Z",
                               "score_reference_point_gripper": score_reference.tolist(),
                               "approach_surface": "triangle_mesh" if mesh is not None else "box_union",
                               "geometry_contract": deepcopy(scene["gripper"].get("geometry_contract")),
                               "feasibility_width_policy": feasibility_policy,
                               "width_policy": width_policy, "max_opening_m": max_width},
            "candidates": rows, "provenance": scene.get("provenance", "user input")}
