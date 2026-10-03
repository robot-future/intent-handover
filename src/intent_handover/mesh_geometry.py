"""Triangle-surface approach intersections, independent of robot calibration."""
import numpy as np
import hashlib
from .geometry import unit, vector


def mesh_arrays(mesh):
    vertices = np.asarray(mesh["vertices"], dtype=float)
    faces = np.asarray(mesh["faces"])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Expected finite vertices and integer triangle indices")
    return vertices, faces


def mesh_digest(mesh):
    """Canonical digest shared with the companion's asset_preparation contract."""
    vertices, faces = mesh_arrays(mesh)
    digest = hashlib.sha256()
    for array, dtype in ((vertices, "<f8"), (faces, "<i8")):
        array = np.asarray(array, dtype=dtype)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def point_mesh_distance(mesh, point):
    """Closest distance to triangles, including all edge and vertex regions."""
    vertices, faces = mesh_arrays(mesh)
    tri = vertices[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b-a, c-a, vector(point)-a
    d00 = np.einsum("ij,ij->i", ab, ab); d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab); d21 = np.einsum("ij,ij->i", ap, ac)
    den = d00*d11-d01*d01
    active = den > 1e-24
    u = np.zeros(len(tri)); v = np.zeros(len(tri))
    u[active] = (d11[active]*d20[active]-d01[active]*d21[active])/den[active]
    v[active] = (d00[active]*d21[active]-d01[active]*d20[active])/den[active]
    inside = active & (u >= 0) & (v >= 0) & (u+v <= 1)
    closest = a+u[:, None]*ab+v[:, None]*ac
    distances = [np.linalg.norm(closest[inside]-point, axis=1)]
    for start, end in ((a, b), (b, c), (c, a)):
        delta = end-start
        denom = np.einsum("ij,ij->i", delta, delta)
        parameter = np.divide(np.einsum("ij,ij->i", point-start, delta), denom,
                              out=np.zeros(len(tri)), where=denom > 1e-24)
        closest = start+np.clip(parameter, 0, 1)[:, None]*delta
        distances.append(np.linalg.norm(closest-point, axis=1))
    return float(np.concatenate(distances).min())


def mesh_pad_geometry(mesh, grasp, window):
    """Measure a fixed pose's full triangle surface inside a declared pad window.

    Clip against X/Z bounds, including intersections at prism corners. This
    kernel never searches, inserts or translates a grasp or calibrates a robot.
    """
    from .geometry import transform
    t = transform(grasp)
    vertices, faces = mesh_arrays(mesh)
    bounds = []
    for axis, key in ((0, "x"), (2, "z")):
        pair = np.asarray(window[key], dtype=float)
        if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
            raise ValueError("Pad bounds must be two ordered finite metre coordinates")
        bounds.append((axis, *pair))
    triangles = ((vertices-t[:3, 3])@t[:3, :3])[faces]
    keep = np.ones(len(triangles), dtype=bool)
    for axis, lo, hi in bounds:
        keep &= (triangles[:, :, axis].max(1) >= lo) & (triangles[:, :, axis].min(1) <= hi)
    triangles = triangles[keep]
    if not len(triangles): return None
    a, b = triangles.reshape(-1, 3), np.roll(triangles, -1, axis=1).reshape(-1, 3)
    candidates = [a]
    for axis, lo, hi in bounds:
        delta = b[:, axis]-a[:, axis]
        active = np.abs(delta) > 1e-12
        for boundary in (lo, hi):
            u = (boundary-a[active, axis])/delta[active]
            valid = (u >= 0) & (u <= 1)
            candidates.append(a[active][valid]+u[valid, None]*(b-a)[active][valid])
    v0 = triangles[:, 0]; e1 = triangles[:, 1]-v0; e2 = triangles[:, 2]-v0
    det = e1[:, 0]*e2[:, 2]-e1[:, 2]*e2[:, 0]
    active = np.abs(det) > 1e-12
    for x in bounds[0][1:]:
        for z in bounds[1][1:]:
            dx, dz = x-v0[active, 0], z-v0[active, 2]
            u = (dx*e2[active, 2]-dz*e2[active, 0])/det[active]
            v = (e1[active, 0]*dz-e1[active, 2]*dx)/det[active]
            valid = (u >= 0) & (v >= 0) & (u+v <= 1)
            candidates.append(v0[active][valid]+u[valid, None]*e1[active][valid]+v[valid, None]*e2[active][valid])
    clipped = np.concatenate(candidates)
    keep = np.ones(len(clipped), dtype=bool)
    for axis, lo, hi in bounds:
        keep &= (clipped[:, axis] >= lo-1e-10) & (clipped[:, axis] <= hi+1e-10)
    clipped = clipped[keep]
    if len(clipped) < 4 or np.any(np.ptp(clipped, axis=0) < 1e-8): return None
    contacts = clipped[[np.argmin(clipped[:, 1]), np.argmax(clipped[:, 1])]]
    low, high = contacts[:, 1]
    return {"width_m": float(high-low), "center_y_m": float((low+high)/2),
            "contact_points_gripper": contacts.tolist(), "pad_window_gripper_m": window,
            "source": "clipped_triangle_mesh", "physical_grasp_verified": False}


def asset_candidate_geometry(scene, candidate, digest):
    """Check immutable asset preparation evidence and independently remeasure."""
    from .geometry import points, transform
    prep = candidate.get("geometry_preparation", {})
    if prep.get("schema_version") != "handover.asset_candidate.v1":
        return None, ["missing_asset_candidate_preparation"]
    if prep.get("status") != "bilateral_surface_fit":
        return None, ["asset_candidate_preparation_failed"]
    contract = scene["gripper"]["geometry_contract"]
    if contract.get("schema_version") != "handover.asset_gripper.v1":
        raise ValueError("Expected handover.asset_gripper.v1 geometry contract")
    mesh = scene["object"]["mesh"]
    t = transform(candidate["T_object_gripper"])
    offset = transform(contract["T_asset_tool_grasp_frame"])
    problems = []
    def check(ok, reason):
        if not ok: problems.append(reason)
    check(digest == contract["mesh_sha256"] == prep["mesh_sha256"], "asset_mesh_binding_changed")
    check(prep["robot"] == contract["robot"], "asset_robot_binding_changed")
    check(prep.get("candidate_id") == candidate["id"], "asset_candidate_identity_changed")
    check(np.allclose(transform(prep["T_object_gripper"]), t, atol=1e-8, rtol=0), "asset_prepared_pose_changed")
    check(np.allclose(transform(prep["T_asset_tool_grasp_frame"]), offset, atol=1e-8, rtol=0), "asset_frame_binding_changed")
    contacts = np.asarray(prep["contact_points_gripper"], dtype=float)
    asset_contacts = np.asarray(prep["contact_points_asset_tool"], dtype=float)
    distances = np.asarray(prep["bilateral_distance_m"], dtype=float)
    if contacts.shape != (2, 3) or asset_contacts.shape != (2, 3) or not np.isfinite(contacts).all() or not np.isfinite(asset_contacts).all():
        return None, problems+["invalid_asset_contact_points"]
    check(np.allclose(points(offset, contacts), asset_contacts, atol=1e-8, rtol=0), "asset_contact_frames_inconsistent")
    check(all(point_mesh_distance(mesh, p) <= 1e-6 for p in points(t, contacts)), "asset_contacts_off_object_surface")
    check(distances.shape == (2,) and np.isfinite(distances).all() and np.all(distances >= 0)
          and np.all(distances <= .0002), "asset_pad_contact_distance_exceeded")
    section = mesh_pad_geometry(mesh, t, contract["pad_window_gripper_m"])
    if section is None: return None, problems+["empty_mesh_pad_window"]
    width = float(prep["width_m"])
    check(np.isfinite(width) and abs(width-section["width_m"]) <= 1e-6, "asset_width_binding_changed")
    check(np.allclose(contacts[:, 1], [-width/2, width/2], atol=1e-6, rtol=0), "asset_contacts_not_on_pad_planes")
    for axis, name in ((0, "x"), (2, "z")):
        lo, hi = contract["pad_window_gripper_m"][name]
        check(np.all(contacts[:, axis] >= lo-1e-8) and np.all(contacts[:, axis] <= hi+1e-8), "asset_contact_outside_pad_window")
    return section, problems


def mesh_approach_intersection(mesh, origin, direction):
    """Nearest positive two-sided Moller-Trumbore hit; no watertight assumption.

    Require the origin outside the mesh AABB, so entry rays cannot start inside
    a solid. Imported rays use a bounding-sphere clearance to meet this rule.
    """
    vertices, faces = mesh_arrays(mesh)
    origin, direction = vector(origin), unit(direction)
    if np.all(origin >= vertices.min(0)-1e-9) and np.all(origin <= vertices.max(0)+1e-9):
        raise ValueError("Mesh approach origin must be outside its bounding box")
    triangles = vertices[faces]
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    active = np.abs(det) > 1e-12
    if not np.any(active): return None
    e1, e2, h, det = e1[active], e2[active], h[active], det[active]
    delta = origin-triangles[active, 0]
    u = np.einsum("ij,ij->i", delta, h)/det
    q = np.cross(delta, e1)
    v = q @ direction/det
    distance = np.einsum("ij,ij->i", e2, q)/det
    hit = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (distance >= 0)
    return origin + distance[hit].min()*direction if np.any(hit) else None
