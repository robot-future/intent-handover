"""Decode H2O Text2HOI poses with locally supplied MANO and update a scene.

MANO assets are never bundled. Output meshes/proxies are user-generated data.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .geometry import box, inverse, points, pose, unit
from .weights import sha256


def load_legacy_mano(path):
    """Normalize a locally trusted legacy MANO pickle without persistent patches.

    Original MANO distributions contain Chumpy arrays. Chumpy predates Python
    3.11/NumPy 1.24; install compatibility aliases only during loading/materialization.
    Never use this on untrusted pickle files.
    """
    import inspect
    import pickle
    from collections import namedtuple
    from smplx.utils import Struct
    aliases = {"bool": bool, "int": int, "float": float, "complex": complex,
               "object": object, "unicode": str, "str": str}
    added = []
    patched_inspect = not hasattr(inspect, "getargspec")
    try:
        for name, value in aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)
                added.append(name)
        if patched_inspect:
            arg_spec = namedtuple("ArgSpec", "args varargs keywords defaults")
            def getargspec(function):
                full = inspect.getfullargspec(function)
                return arg_spec(full.args, full.varargs, full.varkw, full.defaults)
            inspect.getargspec = getargspec
        try:
            with Path(path).open("rb") as stream:
                data = pickle.load(stream, encoding="latin1")
        except ModuleNotFoundError as exc:
            if exc.name == "chumpy":
                raise ImportError("Legacy MANO needs Chumpy. After installing '.[mano]', run: python -m pip install --no-build-isolation --no-deps chumpy==0.70") from exc
            raise
        data = {k: np.asarray(v.r) if hasattr(v, "r") else v for k, v in data.items()}
        return Struct(**data)
    finally:
        for name in added:
            delattr(np, name)
        if patched_inspect:
            delattr(inspect, "getargspec")


def rotation6d(values):
    """Text2HOI's INTERLEAVED first two columns, not two consecutive 3-vectors."""
    x = np.asarray(values, dtype=float)
    if x.shape[-1] != 6 or not np.isfinite(x).all():
        raise ValueError("Expected finite 6D rotations")
    pairs = x.reshape(*x.shape[:-1], 3, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate first rotation axis")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Collinear rotation axes")
    second /= norm
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def bone_box(a, b, radius):
    delta = np.asarray(b) - np.asarray(a)
    length = float(np.linalg.norm(delta))
    z = unit(delta)
    helper = [1, 0, 0] if abs(z[0]) < .8 else [0, 1, 0]
    x = unit(np.cross(helper, z))
    return box((np.asarray(a) + b) / 2, [radius, radius, max(length / 2, .002)],
               np.column_stack([x, np.cross(z, x), z]), "predicted_hand")


def geometry_to_scene(scene, vertices, joints, object_parameters, hand, flip_normal=False):
    """Vertices/joints share the predicted object frame's parent frame (H2O)."""
    vertices, joints = np.asarray(vertices), np.asarray(joints)
    if vertices.shape != (778, 3) or joints.ndim != 2 or joints.shape[0] < 16 or joints.shape[1] != 3:
        raise ValueError("Expected 778 MANO vertices and at least 16 joints")
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("Nonfinite decoded hand geometry")
    obj = np.asarray(object_parameters)
    if obj.shape != (9,):
        raise ValueError("H2O object pose must have nine parameters")
    t = pose(obj[:3], rotation6d(obj[3:]))
    vertices, joints = points(inverse(t), vertices), points(inverse(t), joints)
    direction = unit(vertices[445] - joints[0])
    normal = unit(np.cross(joints[1] - joints[0], joints[7] - joints[0]))
    if hand == "left":
        normal = -normal
    if flip_normal:
        normal = -normal
    proxies = []
    # MANO finger groups: index, middle, pinky, ring, thumb.
    for start, tip in [(1, 317), (4, 445), (7, 673), (10, 556), (13, 745)]:
        chain = [joints[0], *joints[start:start+3], vertices[tip]]
        for i, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
            proxies.append(bone_box(a, b, .009 if i == 0 else .006))
    result = deepcopy(scene)
    result["intent"]["hand"] = hand
    result["receiving_hand"] = {"center": vertices.mean(0).tolist(),
        "direction": direction.tolist(), "palm_normal": normal.tolist(), "boxes": proxies,
        "mesh": {"vertices": vertices.tolist()},
        "provenance": "Original H2O Text2HOI coarse prediction decoded with user-provided MANO"}
    result["provenance"] = "Procedural object/candidates with a Text2HOI-predicted MANO receiving hand"
    if "source_data" in scene:
        result["provenance"] = "Configured local object point cloud, generated candidates/region and Text2HOI-predicted MANO hand"
    return result


def decode_prediction(scene, prediction, mano_models, frame=0, flip_normal=False):
    prediction = Path(prediction)
    metadata = json.loads(prediction.with_name("metadata.json").read_text())
    if metadata.get("status", "succeeded") != "succeeded":
        raise ValueError("Prediction is incomplete or failed; rerun text2hoi before decoding")
    if "prediction_sha256" in metadata and sha256(prediction) != metadata["prediction_sha256"]:
        raise ValueError("Prediction checksum mismatch; metadata and prediction must come from the same run")
    if metadata.get("dataset", "h2o") != "h2o":
        raise ValueError("The bridge currently supports H2O object-frame conventions only")
    hand = metadata["hand"]
    if hand not in ("left", "right"):
        raise ValueError("Hand must be left or right")
    with np.load(prediction, allow_pickle=False) as data:
        hands, objects = data[hand + "_hand"], data["object_pose"]
        if hands.ndim != 3 or hands.shape[0] != 1 or hands.shape[2] != 99:
            raise ValueError("Expected hand predictions of shape (1,T,99)")
        if objects.shape != (1, hands.shape[1], 9) or not 0 <= frame < hands.shape[1]:
            raise ValueError("Mismatched object predictions or frame out of range")
        parameters, obj = hands[0, frame], objects[0, frame]
    model_file = Path(mano_models) / f"MANO_{hand.upper()}.pkl"
    if not model_file.is_file():
        raise ValueError(f"Missing locally licensed MANO model: {model_file}")
    import torch
    import smplx
    from scipy.spatial.transform import Rotation
    # SMPL-X expects axis angles. Convert every joint from the original 6D layout.
    axis_angles = Rotation.from_matrix(rotation6d(parameters[3:].reshape(16, 6))).as_rotvec().astype(np.float32)
    layer = smplx.MANO(str(model_file), data_struct=load_legacy_mano(model_file), use_pca=False, flat_hand_mean=True,
                       is_rhand=hand == "right", batch_size=1)
    with torch.no_grad():
        output = layer(global_orient=torch.from_numpy(axis_angles[:1]),
                       hand_pose=torch.from_numpy(axis_angles[1:].reshape(1,45)),
                       betas=torch.zeros((1,10)))
    vertices = output.vertices[0].numpy() + parameters[:3]
    joints = output.joints[0].numpy() + parameters[:3]
    result = geometry_to_scene(scene, vertices, joints, obj, hand, flip_normal)
    result["receiving_hand"]["mesh"]["faces"] = layer.faces.tolist()
    result["receiving_hand"]["prediction_frame"] = frame
    result["receiving_hand"]["checkpoint_hashes"] = metadata.get("checkpoints_sha256", {})
    return result
