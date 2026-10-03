"""Import locally trusted legacy Panda proposals without inventing paper labels."""
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
from .geometry import corners, inverse, points, pose, transform, unit
from .grasp_geometry import local_pad_geometry
from .mesh_geometry import mesh_arrays
from .weights import sha256


def panda_frame(control_points):
    """Canonical +Y = left-to-right closing direction; +Z = base +Z.

    Return the base-to-tip frame and the actual pad-centre score reference.
    All offsets come from the supplied original control points, not constants.
    """
    left, right, left_tip, right_tip = [np.asarray(control_points[k], dtype=float) for k in
        ("gripper_left_center_flat", "gripper_right_center_flat", "gripper_left_tip_flat", "gripper_right_tip_flat")]
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (left, right, left_tip, right_tip)):
        raise ValueError("Panda control points must be finite XYZ vectors in metres")
    y = unit(right-left); z = np.array([0., 0., 1.])
    if abs(y@z) > 1e-6:
        raise ValueError("Panda closing axis must be perpendicular to base +Z")
    frame = pose((left_tip+right_tip)/2, np.column_stack([np.cross(y, z), y, z]))
    return frame, points(inverse(frame), (left+right)/2)


def repair_annotation_pose(raw):
    """Remove only small rotation serialization error; retain every raw value."""
    t = np.asarray(raw, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Malformed source grasp transform")
    r = t[:3, :3]
    error = float(np.max(np.abs(r.T@r-np.eye(3))))
    if error > 1e-3 or np.linalg.det(r) <= 0:
        raise ValueError("Source rotation is not a small numerical perturbation of SO(3)")
    u, _, vt = np.linalg.svd(r)
    repaired = t.copy(); repaired[:3, :3] = u@vt; repaired[3] = [0, 0, 0, 1]
    return transform(repaired), error


def import_panda_candidates(scene, annotations, control_points, mesh=None, center=True):
    """Convert every annotated proposal in input order, before mode selection.

    Source part names are provenance only. Existing human usage geometry and
    intent remain authoritative; no point/face segmentation is inferred here.
    """
    annotations, control_points = Path(annotations), Path(control_points)
    with annotations.open("rb") as stream:
        archive = pickle.load(stream, encoding="latin1")
    with control_points.open("rb") as stream:
        controls = pickle.load(stream, encoding="latin1")
    object_id = scene["object"]["id"]
    if object_id not in archive or not isinstance(archive[object_id], dict) or not archive[object_id]:
        raise ValueError("No region-grouped candidate annotations for this object")
    base_tip, score_reference = panda_frame(controls)
    result = deepcopy(scene)
    if mesh is not None:
        vertices, faces = mesh_arrays(mesh)
        result["object"]["mesh"] = {"vertices": vertices.tolist(), "faces": faces.tolist()}
    extents = (np.asarray(result["object"]["mesh"]["vertices"]) if "mesh" in result["object"] else
               np.concatenate([corners(b) for b in scene["object"]["boxes"]]))
    midpoint = (extents.min(0)+extents.max(0))/2
    radius = np.linalg.norm(extents.max(0)-extents.min(0))/2
    candidates, region_counts = [], {}
    for region, proposals in archive[object_id].items():
        if not isinstance(region, str) or not region or not len(proposals):
            raise ValueError("Annotation region names/proposals must be nonempty")
        region_counts[region] = len(proposals)
        for index, raw in enumerate(proposals):
            source, error = repair_annotation_pose(raw)
            mapped = source@base_tip
            final = mapped.copy()
            section = local_pad_geometry(result["object"]["boxes"], final) if center else None
            if section is not None:
                final[:3, 3] += section["center_y_m"]*final[:3, 1]
            clearance = float(np.linalg.norm(final[:3, 3]-midpoint)+radius+.1)
            candidates.append({"id": f"annotated_{len(candidates):05d}", "T_object_gripper": final.tolist(),
                "approach_ray_origin_object": (final[:3, 3]-clearance*final[:3, 2]).tolist(),
                "geometry_preparation": {"status": "annotation_proxy_centered" if section is not None else "annotation_frame_converted",
                    "source_region": region, "source_region_index": index,
                    "source_T_object_panda": np.asarray(raw).tolist(),
                    "source_rotation_orthogonality_error": error,
                    "original_T_object_gripper": mapped.tolist(),
                    "translation_in_original_tool_m": (mapped[:3, :3].T@(final[:3, 3]-mapped[:3, 3])).tolist(),
                    "insertion_added_m": 0.}})
    if not candidates:
        raise ValueError("No annotated candidates to import")
    result["candidates"] = candidates
    result["gripper"].pop("geometry_contract", None)
    result["gripper"].update(width_policy="local_pad_proxy", grasp_frame="parallel_jaw_tip",
                            score_reference_point_gripper=score_reference.tolist())
    result["candidate_source"] = {"kind": "local_region_grouped_panda_annotations",
        "annotations_sha256": sha256(annotations), "control_points_sha256": sha256(control_points),
        "count": len(candidates), "regions": region_counts, "T_panda_gripper": base_tip.tolist(),
        "region_labels_used_as_surface_masks": False, "paper_candidate_subset_verified": False}
    result["provenance"] = "Local annotated Panda proposals converted before ranking; original object inputs, supplied usage geometry and hand"
    return result
