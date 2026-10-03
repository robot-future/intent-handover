"""Local aperture estimates for the release's parallel-jaw box proxy.

This is deliberately independent of Robotiq linkage/CAD calibration. Contacts
are on proxy boxes, not measured object triangles or force-closure evidence.
"""
from itertools import combinations
import numpy as np
from .geometry import box_pose, transform, vector


WIDTH_POLICIES = ("global_projection", "local_pad_proxy", "asset_mesh_pad")
GRASP_FRAME = "parallel_jaw_tip"
# Same inner finger faces as geometry.gripper_boxes: Y closes, +Z approaches.
PAD_X = (-.012, .012)
PAD_Z = (-.044, 0.)
_TRIPLES = np.array(list(combinations(range(10), 3)))


def local_pad_geometry(boxes, grasp):
    """Clip every OBB to the pad's X/Z window, then measure its Y envelope.

    Vertices of a clipped convex box are intersections of three boundary
    planes. Include all occupied pieces in the window, including disconnected
    ones; never choose a narrow piece and silently ignore another obstruction.
    Returns None for no volumetric section. The pose is never changed here.
    """
    t = transform(grasp)
    sections = []
    for b in boxes:
        bt = box_pose(b)
        axes = t[:3, :3].T @ bt[:3, :3]
        center = (bt[:3, 3] - t[:3, 3]) @ t[:3, :3]
        normals = np.concatenate((axes.T, -axes.T,
                                  [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]))
        limits = np.r_[vector(b["half_extents"]) + axes.T @ center,
                       vector(b["half_extents"]) - axes.T @ center,
                       PAD_X[1], -PAD_X[0], PAD_Z[1], -PAD_Z[0]]
        systems = normals[_TRIPLES]
        valid = np.abs(np.linalg.det(systems)) > 1e-10
        vertices = np.linalg.solve(systems[valid], limits[_TRIPLES[valid]][..., None])[..., 0]
        vertices = vertices[np.all(vertices @ normals.T <= limits + 1e-9, axis=1)]
        # Disconnected tangent faces must not jointly look like an occupied
        # volume merely because their combined XYZ extents are all nonzero.
        if len(vertices) >= 4 and np.linalg.matrix_rank(vertices-vertices[0], tol=1e-8) == 3:
            sections.append(vertices)
    if not sections:
        return None
    vertices = np.concatenate(sections)
    # A face/edge touching the tip is not an inserted grasp.
    if np.any(np.ptp(vertices, axis=0) <= 1e-8):
        return None
    contacts = vertices[[np.argmin(vertices[:, 1]), np.argmax(vertices[:, 1])]]
    low, high = contacts[:, 1]
    return {"width_m": float(high-low), "center_y_m": float((low+high)/2),
            "contact_points_gripper": contacts.tolist(),
            "pad_window_gripper_m": {"x": list(PAD_X), "z": list(PAD_Z)},
            "source": "clipped_object_box_union", "physical_grasp_verified": False}
