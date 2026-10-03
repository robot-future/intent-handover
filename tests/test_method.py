import copy
import unittest
import numpy as np
from intent_handover.core import select_grasp
from intent_handover.demos import load_scene
from intent_handover.geometry import box, contains, intersects, inverse, moved, points, pose, sphere_intersects, transform
from intent_handover.text2hoi import prepare_points


class MethodTests(unittest.TestCase):
    def test_usage_awareness_changes_choice(self):
        scene = load_scene("hammer")
        self.assertEqual(select_grasp(scene)["selected"]["id"], "free_top")
        self.assertEqual(select_grasp(scene, "A1")["selected"]["id"], "usage_opposite")

    def test_width_constraint_survives_all_ablations(self):
        scene = load_scene("hammer")
        scene["gripper"]["max_opening_m"] = .001
        for mode in ("FS", "A1", "A2", "A3"):
            result = select_grasp(scene, mode)
            self.assertIsNone(result["selected"])
            self.assertEqual(result["status"], "no_feasible_grasp")

    def test_score_prefers_opposite_direction(self):
        scene = load_scene("hammer")
        rows = {r["id"]: r for r in select_grasp(scene)["candidates"]}
        self.assertLess(rows["usage_opposite"]["cosine"], rows["free_same"]["cosine"])

    def test_invalid_inputs_are_rejected(self):
        for edit in (lambda s: s.update(units="mm"),
                     lambda s: s["intent"].update(object_id="different"),
                     lambda s: s["intent"].update(needs_clarification=True),
                     lambda s: s["receiving_hand"].update(direction=[0, 0, 0]),
                     lambda s: s["intent"].update(human_region="missing"),
                     lambda s: s["candidates"].append(copy.deepcopy(s["candidates"][0]))):
            scene = load_scene("hammer")
            edit(scene)
            with self.assertRaises(ValueError):
                select_grasp(scene)

    def test_se3_roundtrip(self):
        t = pose([2, 3, 4], [[0,-1,0],[1,0,0],[0,0,1]])
        p = np.array([[1,2,3],[-2,0,1]])
        np.testing.assert_allclose(points(inverse(t), points(t,p)), p, atol=1e-10)
        with self.assertRaises(ValueError):
            transform(np.zeros((4,4)))

    def test_rotated_boxes_and_sphere(self):
        a = box([0,0,0],[1,.1,.1])
        b = box([0,.5,0],[1,.1,.1])
        self.assertFalse(intersects(a,b))
        b = moved(b, pose(rotation=[[0,-1,0],[1,0,0],[0,0,1]]))
        self.assertTrue(intersects(a,b))
        self.assertTrue(sphere_intersects(a,[0,0,0],.01))
        self.assertFalse(sphere_intersects(a,[0,3,0],.1))

    def test_point_cloud_preparation(self):
        raw = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
        sampled, normalized, center, scale = prepare_points(raw)
        self.assertEqual(sampled.shape,(1024,3))
        self.assertAlmostEqual(float(np.linalg.norm(normalized,axis=1).max()),1.,places=6)
        with self.assertRaises(ValueError):
            prepare_points(np.zeros((4,3)))


if __name__ == "__main__":
    unittest.main()
