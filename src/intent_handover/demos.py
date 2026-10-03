"""Original procedural demo assets; no third-party meshes or MANO redistribution."""
from copy import deepcopy
import json
from importlib.resources import files
import numpy as np
from .geometry import box, box_pose, points, pose

NAMES = ("hammer", "screwdriver", "bottle")


def sample_surface(scene, count=2048, seed=0):
    """Deterministic surface cloud of the example's proxy boxes, in metres."""
    rng = np.random.default_rng(seed)
    result = []
    boxes = scene["object"]["boxes"]
    for i in range(count):
        b = boxes[i % len(boxes)]
        half = np.asarray(b["half_extents"])
        local = rng.uniform(-1, 1, 3) * half
        axis = int(rng.integers(3))
        local[axis] = half[axis] * rng.choice([-1, 1])
        result.append(points(box_pose(b), local))
    return np.asarray(result, dtype=np.float32)


def load_scene(name):
    if name not in NAMES:
        raise ValueError(f"Unknown demo {name}; choose from {', '.join(NAMES)}")
    return json.loads(files(__package__).joinpath("assets", name + ".json").read_text())


def build_scene(name):
    dimensions = {
        "hammer": [(0, 0, -.07), (.014, .014, .08), (0, 0, .035), (.07, .025, .025)],
        "screwdriver": [(0, 0, -.075), (.019, .019, .055), (0, 0, .04), (.004, .004, .065)],
        "bottle": [(0, 0, -.025), (.035, .035, .075), (0, 0, .075), (.014, .014, .025)],
    }
    c1, e1, c2, e2 = dimensions[name]
    region1, region2 = ("body", "neck") if name == "bottle" else ("handle", "head" if name == "hammer" else "shaft")
    first, second = box(c1, e1, label=region1), box(c2, e2, label=region2)
    # Gripper +Z approach is aligned with object +/-X; +Y closes along object Y.
    minus_x = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    plus_x = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    # Ordered deliberately so disabling usage awareness changes the selected region.
    candidates = []
    for cid, part, sign, rotation in [("usage_opposite", first, 1, minus_x),
                                      ("free_same", second, -1, plus_x),
                                      ("free_top", second, 1, np.diag([1., -1., -1.]))]:
        hit = list(part["center"])
        hit[0] += sign * part["half_extents"][0]
        if cid == "free_top":
            hit = list(part["center"])
            hit[2] += part["half_extents"][2]
        candidates.append({"id": cid, "T_object_gripper": pose(hit, rotation).tolist(),
                           "approach_point_object": hit,
                           "approach_ray_origin_object": (np.asarray(hit)-.2*rotation[:, 2]).tolist()})
    utterance = {"hammer": "I need to hammer a nail.",
                 "screwdriver": "I want to tighten this screw.",
                 "bottle": "Pass me the bottle so I can drink."}[name]
    hand_boxes = [box([-.105, 0, c1[2]], [.025, .038, .012], label="palm")]
    hand_boxes += [box([-.062, y, c1[2]], [.018, .006, .009], label="finger")
                   for y in (-.027, -.009, .009, .027)]
    hand_boxes += [box([-.085, -.047, c1[2]], [.018, .008, .01], label="thumb")]
    return {"schema_version": "handover.scene.v1", "units": "m",
            "provenance": "Procedural release demo with fixed receiving-hand geometry",
            "utterance": utterance,
            "intent": {"object_id": name, "human_region": region1, "robot_region": region2,
                       "hand": "right", "text2hoi_prompt": f"Grasp a {name} with right hand."},
            "object": {"id": name, "boxes": [first, second],
                       "usage_regions": {region1: [deepcopy(first)], region2: [deepcopy(second)]}},
            "receiving_hand": {"center": [-.075, 0, c1[2]], "direction": [1, 0, 0],
                               "boxes": hand_boxes},
            "gripper": {"max_opening_m": .085, "geometry": "parallel-jaw box proxy"},
            "candidates": candidates}
