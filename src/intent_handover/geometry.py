"""Small NumPy geometry kernel. Column-vector transforms, metres, right handed.

The demo uses unions of oriented boxes as explicit proxy geometries.
Touching counts as intersection.
"""
import itertools
import numpy as np


def vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Expected a finite 3-vector")
    return result


def unit(value):
    value = vector(value)
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError("Direction must be nonzero")
    return value / norm


def transform(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Expected a finite 4x4 transform")
    r = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6)):
        raise ValueError("Transform must belong to SE(3)")
    return value


def pose(position=(0, 0, 0), rotation=None):
    result = np.eye(4)
    result[:3, 3] = vector(position)
    if rotation is not None:
        result[:3, :3] = rotation
    return transform(result)


def inverse(value):
    value = transform(value)
    return pose(-value[:3, :3].T @ value[:3, 3], value[:3, :3].T)


def points(value, xyz):
    value = transform(value)
    return np.asarray(xyz) @ value[:3, :3].T + value[:3, 3]


def box(center, half_extents, rotation=None, label=""):
    extents = vector(half_extents)
    if np.any(extents <= 0):
        raise ValueError("Box half extents must be positive")
    return {"center": vector(center).tolist(), "half_extents": extents.tolist(),
            "rotation": pose(rotation=rotation)[:3, :3].tolist(), "label": label}


def box_pose(b):
    vector(b["center"])
    if np.any(vector(b["half_extents"]) <= 0):
        raise ValueError("Box half extents must be positive")
    return pose(b["center"], b.get("rotation", np.eye(3)))


def corners(b):
    return points(box_pose(b), np.array(list(itertools.product([-1, 1], repeat=3)))
                  * vector(b["half_extents"]))


def moved(b, t):
    combined = transform(t) @ box_pose(b)
    return box(combined[:3, 3], b["half_extents"], combined[:3, :3], b.get("label", ""))


def contains(b, p):
    local = points(inverse(box_pose(b)), vector(p))
    return bool(np.all(np.abs(local) <= vector(b["half_extents"]) + 1e-9))


def intersects(a, b):
    """Exact oriented-box SAT, including edge cross-product separating axes."""
    ra, rb = box_pose(a)[:3, :3], box_pose(b)[:3, :3]
    delta = vector(b["center"]) - vector(a["center"])
    axes = [*ra.T, *rb.T] + [np.cross(x, y) for x in ra.T for y in rb.T]
    ea, eb = vector(a["half_extents"]), vector(b["half_extents"])
    for axis in axes:
        if np.linalg.norm(axis) < 1e-10:
            continue
        radius = np.sum(ea * np.abs(ra.T @ axis)) + np.sum(eb * np.abs(rb.T @ axis))
        if abs(delta @ axis) > radius + 1e-9:
            return False
    return True


def sphere_intersects(b, center, radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Sphere radius must be positive")
    local = points(inverse(box_pose(b)), vector(center))
    closest = np.clip(local, -vector(b["half_extents"]), vector(b["half_extents"]))
    return bool(np.linalg.norm(local - closest) <= radius + 1e-9)


def projected_width(boxes, closing_axis):
    axis = unit(closing_axis)
    if not boxes:
        raise ValueError("Object requires at least one box")
    coordinates = np.concatenate([corners(b) @ axis for b in boxes])
    return float(coordinates.max() - coordinates.min())


def approach_intersection(boxes, origin, direction):
    """First ray/surface intersection for an outside origin and a union of OBBs."""
    origin, direction = vector(origin), unit(direction)
    if any(contains(b, origin) for b in boxes):
        raise ValueError("Approach ray origin must lie outside the object")
    first = np.inf
    for b in boxes:
        rotation = box_pose(b)[:3, :3]
        local = rotation.T@(origin-vector(b["center"]))
        ray = rotation.T@direction
        lower, upper = -np.inf, np.inf
        for i, extent in enumerate(b["half_extents"]):
            if abs(ray[i]) < 1e-12:
                if abs(local[i]) > extent:
                    lower, upper = np.inf, -np.inf
                    break
            else:
                pair = sorted(((-extent-local[i])/ray[i], (extent-local[i])/ray[i]))
                lower, upper = max(lower, pair[0]), min(upper, pair[1])
        if upper >= max(lower, 0.):
            first = min(first, max(lower, 0.))
    return None if not np.isfinite(first) else origin+first*direction


def gripper_boxes(opening):
    """Simplified parallel gripper, TCP at finger tips, closing axis +Y, approach +Z."""
    if not np.isfinite(opening) or opening < 0:
        raise ValueError("Opening must be finite and nonnegative")
    return [box([0, sign * (opening / 2 + .006), -.022], [.012, .006, .022], label="finger")
            for sign in (-1, 1)] + [box([0, 0, -.052], [.02, opening / 2 + .012, .008], label="palm")]
