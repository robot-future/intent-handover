"""Read-only checks of legacy and explicit method-grasp replay contracts.

Compares grasp, annotation and receiver geometry using NumPy.
Use the resolved *_trial.json exported by the simulator for post-run checks.
"""
import numpy as np
from .core import select_grasp
from .geometry import inverse, points, projected_width, transform


def audit_replay(scene, selection, trial, delivery=None):
    if selection.get("schema_version") != "handover.selection.v1" or selection.get("units") != "m":
        raise ValueError("Expected handover.selection.v1 with units=m")
    if trial.get("schema_version") != "handover.trial.v1" or trial.get("units") != "m":
        raise ValueError("Expected handover.trial.v1 with units=m")
    current = select_grasp(scene, selection["mode"])
    expected = current["selected"]
    saved = selection.get("selected")
    if not expected or selection.get("status") != "ok" or not saved:
        raise ValueError("Replay audit requires a feasible method selection")
    issues = []
    def check(condition, reason):
        if not condition:
            issues.append(reason)
    def close(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=1e-8, rtol=0)
    check(selection["object_id"] == scene["object"]["id"] == trial.get("object_id"), "object_id_changed")
    if "evaluation_split" in scene:
        check(trial.get("split") == scene["evaluation_split"], "evaluation_split_changed")
    if "evaluation_split_provenance" in scene:
        check(trial.get("evaluation_split_provenance") == scene["evaluation_split_provenance"],
              "evaluation_split_provenance_changed")
    check(saved["id"] == expected["id"] and close(saved["T_object_gripper"], expected["T_object_gripper"])
          and close(saved["width_m"], expected["width_m"])
          and all(close(saved.get(k), expected[k]) for k in ("cosine", "distance_m", "avoidance_cost")),
          "selection_stale_or_modified")
    actual_pose = transform(trial["T_object_gripper"])
    check(close(actual_pose, expected["T_object_gripper"]), "grasp_pose_changed")
    check(close(trial["max_opening_m"], scene["gripper"]["max_opening_m"]), "maximum_aperture_changed")
    # Prefer the integrated consumer's explicit actuator aperture. Legacy
    # trials retain their documented global-projection/asset-fit behavior.
    fit = trial.get("asset_contact_fit", {})
    method = trial.get("method_selection")
    if "gripper_opening_m" in trial:
        actual_width = float(trial["gripper_opening_m"])
        width_source = trial.get("gripper_opening_source", "unknown")
        check(method is not None, "missing_method_selection_evidence")
    elif fit.get("status") == "bilateral_surface_fit":
        actual_width = float(fit["width_m"])
        width_source = "asset_contact_fit"
    else:
        actual_width = projected_width(trial["object_boxes"], actual_pose[:3, 1])
        width_source = "global_projection"
    check(np.isfinite(actual_width) and actual_width > 0 and close(actual_width, expected["width_m"]), "replay_width_changed")
    if current["grasp_contract"]["width_policy"] != width_source:
        issues.append("width_policy_not_preserved")
    if method is not None:
        check(method.get("id") == expected["id"] and method.get("mode") == selection["mode"]
              and close(method.get("T_object_gripper"), expected["T_object_gripper"])
              and close(method.get("width_m"), expected["width_m"]), "method_selection_evidence_changed")
        check(method.get("width_source") == expected["width_source"], "method_width_source_changed")
        check(all(close(method.get(k), expected[k]) for k in ("cosine", "distance_m", "avoidance_cost")),
              "method_score_evidence_changed")
        defaults = {"score_reference_point_gripper": [0, 0, 0], "approach_surface": "box_union", "geometry_contract": None,
                    "feasibility_width_policy": "opening"}
        supplied = trial.get("grasp_contract", {})
        check(all(supplied.get(k, defaults.get(k)) == v for k, v in current["grasp_contract"].items()), "grasp_contract_changed")
    if current["grasp_contract"]["feasibility_width_policy"] == "object_projection" and "stability_width_m" in trial:
        check(close(trial["stability_width_m"], expected["feasibility_width_m"]), "stability_width_changed")
    if current["grasp_contract"]["feasibility_width_policy"] == "object_projection":
        check(close(saved.get("feasibility_width_m"), expected["feasibility_width_m"]) and method is not None
              and close(method.get("feasibility_width_m"), expected["feasibility_width_m"]), "feasibility_width_evidence_changed")
    if "asset_robot" in trial or "T_tcp_asset_tool" in trial:
        geometry = current["grasp_contract"].get("geometry_contract")
        calibrated = (method is not None and expected["width_source"] == "asset_mesh_pad" and geometry
                      and method.get("geometry_preparation") == expected.get("geometry_preparation")
                      and trial.get("grasp_contract", {}).get("geometry_contract") == geometry)
        check(calibrated, "asset_tool_frame_requires_explicit_method_calibration")
        if calibrated and "T_tcp_asset_tool" in trial:
            check(close(trial.get("T_asset_tool_grasp_frame"), geometry["T_asset_tool_grasp_frame"]),
                  "resolved_asset_frame_changed")
            check(fit.get("status") == "bilateral_surface_fit" and close(fit.get("width_m"), expected["width_m"]),
                  "resolved_asset_contact_width_changed")
            errors = np.asarray(fit.get("bilateral_distance_m", []), dtype=float)
            check(errors.shape == (2,) and np.isfinite(errors).all() and np.all(errors >= 0) and np.all(errors <= .0002),
                  "resolved_asset_contact_verification_failed")
    if "mesh" in scene["object"]:
        from .mesh_geometry import mesh_digest
        actual_mesh = trial.get("object_mesh_object")
        check(actual_mesh is not None and mesh_digest(actual_mesh) == mesh_digest(scene["object"]["mesh"]),
              "object_mesh_changed")
    # Exact copied box annotations are part of this adapter contract. Compare
    # numeric geometry rather than optional labels or JSON formatting.
    def boxes_equal(left, right):
        if len(left) != len(right): return False
        return all(close(a["center"], b["center"]) and close(a["half_extents"], b["half_extents"])
                   and close(a.get("rotation", np.eye(3)), b.get("rotation", np.eye(3)))
                   for a, b in zip(left, right))
    check(boxes_equal(scene["object"]["boxes"], trial["object_boxes"]), "object_proxy_changed")
    region = scene["object"]["usage_regions"][scene["intent"]["human_region"]]
    check(boxes_equal(region, trial["usage_boxes"]), "usage_region_changed")
    world_object = transform(trial["target_T_world_gripper"]) @ inverse(actual_pose)
    if "target_T_world_object" in scene:
        check(close(world_object, scene["target_T_world_object"]) and
              close(trial.get("target_T_world_object"), scene["target_T_world_object"]), "fixed_object_target_changed")
    if scene.get("receiver_protocol", {}).get("policy") == "fixed_world":
        check(trial.get("receiver_protocol", {}).get("policy") == "fixed_world", "fixed_receiver_protocol_changed")
        receiver = scene.get("receiver", {})
        actual_receiver = trial.get("receiver", {})
        check(all(actual_receiver.get(k) == value for k, value in receiver.items() if k != "T_world_hand"),
              "fixed_receiver_identity_changed")
        if "T_world_hand" in receiver:
            check(close(actual_receiver.get("T_world_hand"), receiver["T_world_hand"]), "fixed_receiver_pose_changed")
    check(close(points(inverse(world_object), trial["palm_position_world"]), scene["receiving_hand"]["center"]),
          "receiver_object_relation_changed")
    # Preserve the complete receiving geometry when a scene supplies it, not
    # just a coincident palm centre with a rotated/replaced hand.
    if "boxes" in scene["receiving_hand"]:
        from .geometry import moved
        check(boxes_equal([moved(b, inverse(world_object)) for b in trial.get("hand_boxes_world", [])],
                          scene["receiving_hand"]["boxes"]), "receiver_proxy_geometry_changed")
    if "mesh" in scene["receiving_hand"]:
        expected_mesh = scene["receiving_hand"]["mesh"]
        actual_mesh = trial.get("hand_mesh_world")
        check(actual_mesh is not None and close(points(inverse(world_object), actual_mesh["vertices"]), expected_mesh["vertices"])
              and np.array_equal(actual_mesh["faces"], expected_mesh["faces"]), "receiver_mesh_geometry_changed")
    if trial.get("experiment", {}).get("paired_receiver_preserved") is False:
        issues.append("paired_receiver_not_preserved")
    if delivery is not None:
        if delivery.get("schema_version") != "handover.delivery.v1" or delivery.get("units") != "m":
            raise ValueError("Expected handover.delivery.v1 with units=m")
        check(delivery["object_id"] == current["object_id"] and delivery["grasp_id"] == expected["id"],
              "delivery_identity_changed")
        check(close(world_object, delivery["T_world_object"]) and
              close(trial["target_T_world_gripper"], delivery["T_world_gripper"]), "delivery_target_changed")
    return {"schema_version": "handover.replay_audit.v1", "units": "m",
            "status": "equivalent" if not issues else "not_equivalent",
            "scope": "Grasp, annotations and supplied receiver geometry",
            "object_id": current["object_id"], "mode": current["mode"], "issues": issues,
            "expected_width_m": expected["width_m"], "replay_width_m": actual_width,
            "expected_width_source": current["grasp_contract"]["width_policy"], "replay_width_source": width_source,
            "delivery_checked": delivery is not None, "physical_grasp_verified": False}
