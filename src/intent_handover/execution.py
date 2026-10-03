"""Paper Sec. III-B: ergonomic delivery from calibrated skeletal keypoints.

All input keypoints are world-frame metres, +Z up. The explicit extension axis
and facing hint resolve signs that the paper's equations leave unspecified.
"""
import numpy as np
from .geometry import points, pose, transform, unit, vector


def axis_rotation(axis, angle):
    x, y, z = unit(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)


def minimum_rotation(source, target):
    """Minimum-angle SO(3) alignment; deterministic axis at the antipodal tie."""
    a, b = unit(source), unit(target)
    cross = np.cross(a, b)
    cosine = float(np.clip(a@b, -1., 1.))
    if np.linalg.norm(cross) < 1e-10:
        if cosine > 0:
            return np.eye(3)
        helper = np.eye(3)[np.argmin(np.abs(a))]
        return axis_rotation(np.cross(a, helper), np.pi)
    return axis_rotation(cross, np.arctan2(np.linalg.norm(cross), cosine))


def delivery_target(scene, selection, skeleton):
    if skeleton.get("schema_version") != "handover.skeleton.v1" or skeleton.get("units") != "m":
        raise ValueError("Expected handover.skeleton.v1, calibrated world-frame metres")
    if (selection.get("status") != "ok" or not selection.get("selected")
            or selection["object_id"] != scene["object"]["id"]):
        raise ValueError("A feasible selection for this scene is required")
    left, right = vector(skeleton["left_shoulder"]), vector(skeleton["right_shoulder"])
    elbow, wrist = vector(skeleton["elbow"]), vector(skeleton["wrist"])
    shoulder = (left + right)/2
    desk = float(skeleton["desk_height_m"])
    extension = float(skeleton.get("wrist_extension_deg", 15.))
    if not np.isfinite([desk, extension]).all():
        raise ValueError("Desk height and wrist extension must be finite")
    torso = shoulder.copy()
    torso[2] = (shoulder[2]+desk)/2
    # Horizontal perpendicular to the shoulder axis, oriented by a facing hint.
    facing = unit(np.cross([0., 0., 1.], right-left))
    hint = vector(skeleton["facing_hint_world"])
    if abs(facing@hint) < 1e-8:
        raise ValueError("Facing hint must distinguish forward from backward")
    if facing@hint < 0:
        facing = -facing
    upper, forearm = np.linalg.norm(elbow-shoulder), np.linalg.norm(wrist-elbow)
    if min(upper, forearm) < 1e-6:
        raise ValueError("Degenerate arm keypoints")
    radius = float(np.hypot(upper, forearm))
    position = torso + radius*facing
    forearm_axis = unit(wrist-elbow)
    extension_axis = unit(skeleton["extension_axis_world"])
    if abs(extension_axis@forearm_axis) > 1e-5:
        raise ValueError("Wrist extension axis must be perpendicular to the forearm")
    hand_axis = axis_rotation(extension_axis, np.deg2rad(extension))@forearm_axis
    rotation = minimum_rotation(scene["receiving_hand"]["direction"], hand_axis)
    # P is the object-origin delivery anchor. Preserve the predicted hand/object
    # relation instead of independently translating the hand and the object.
    world_object = pose(position, rotation)
    world_gripper = world_object@transform(selection["selected"]["T_object_gripper"])
    robot_world = transform(skeleton.get("T_robot_world", np.eye(4)))
    return {"schema_version": "handover.delivery.v1", "units": "m",
            "object_id": scene["object"]["id"], "grasp_id": selection["selected"]["id"],
            "T_world_object": world_object.tolist(), "T_world_gripper": world_gripper.tolist(),
            "T_robot_gripper": (robot_world@world_gripper).tolist(),
            "hand_center_world": points(world_object, scene["receiving_hand"]["center"]).tolist(),
            "hand_direction_world": hand_axis.tolist(), "shoulder_midpoint": shoulder.tolist(),
            "torso_center": torso.tolist(), "upper_arm_length_m": float(upper),
            "forearm_length_m": float(forearm), "comfortable_radius_m": radius,
            "facing_direction_world": facing.tolist(), "wrist_extension_deg": extension,
            "keypoints_world": {key: vector(skeleton[key]).tolist() for key in
                                ("left_shoulder", "right_shoulder", "elbow", "wrist")},
            "position_anchor": "object origin", "orientation_rule": "minimum-angle alignment of predicted wrist-to-middle-finger direction",
            "provenance": "Intent-Handover Sec. III-B formulas; supplied keypoints, no live MediaPipe"}
