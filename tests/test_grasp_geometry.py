import copy
import unittest
import numpy as np
from intent_handover.core import MODES, select_grasp
from intent_handover.dataset import bootstrap_scene
from intent_handover.geometry import box, contains, moved, points, pose
from intent_handover.grasp_geometry import local_pad_geometry


class GraspGeometryTests(unittest.TestCase):
    def scene(self):
        # Narrow handle inside the pads, broad head beyond their X footprint.
        return {"schema_version": "handover.scene.v1", "units": "m",
                "intent": {"object_id": "tool", "human_region": "head"},
                "object": {"id": "tool", "boxes": [box([0, 0, -.01], [.04, .01, .01]),
                            box([.07, 0, -.01], [.02, .06, .01])],
                           "usage_regions": {"head": [box([.07, 0, -.01], [.02, .06, .01])]}},
                "receiving_hand": {"direction": [0, 0, 1], "center": [.15, 0, 0]},
                "gripper": {"max_opening_m": .085, "width_policy": "local_pad_proxy"},
                "candidates": [{"id": "handle", "T_object_gripper": pose().tolist(),
                                "approach_ray_origin_object": [0, 0, -.2]}]}

    def test_local_width_and_contacts_ignore_remote_head(self):
        scene = self.scene()
        result = select_grasp(scene)
        row = result["selected"]
        self.assertAlmostEqual(row["width_m"], .02)
        contacts = np.array(row["proxy_contact"]["contact_points_gripper"])
        np.testing.assert_allclose(contacts[:, 1], [-.01, .01])
        for p in contacts:
            self.assertTrue(any(contains(b, p) for b in scene["object"]["boxes"]))
        scene["gripper"]["width_policy"] = "global_projection"
        self.assertIsNone(select_grasp(scene)["selected"])

    def test_clipped_section_is_invariant_under_rigid_frame_change(self):
        scene = self.scene()
        angle = .71
        rot = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
               [-np.sin(angle), 0, np.cos(angle)]]
        t = pose([.4, -.3, .2], rot)
        section = local_pad_geometry([moved(b, t) for b in scene["object"]["boxes"]], t)
        self.assertAlmostEqual(section["width_m"], .02)
        self.assertAlmostEqual(section["center_y_m"], 0.)

    def test_oblique_box_is_clipped_at_pad_boundaries(self):
        angle = np.pi/4
        rot = [[np.cos(angle), -np.sin(angle), 0],
               [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        b = box([0, 0, -.02], [.1, .005, .01], rot)
        section = local_pad_geometry([b], pose())
        self.assertAlmostEqual(section["width_m"], 2*(.012+.005*np.sqrt(2)))
        for p in section["contact_points_gripper"]:
            self.assertTrue(contains(b, p))
            self.assertLessEqual(abs(p[0]), .012+1e-9)

    def test_empty_tangent_and_off_center_sections_are_rejected(self):
        for position, reason in [([0, 0, -.1], "empty_pad_window"),
                                 ([0, 0, -.02], "empty_pad_window"),
                                 ([0, .003, 0], "off_center_pad_section")]:
            scene = self.scene()
            scene["candidates"][0]["T_object_gripper"] = pose(position).tolist()
            scene["candidates"][0]["approach_ray_origin_object"] = [position[0], position[1], -.2]
            for mode in MODES:
                result = select_grasp(scene, mode)
                self.assertIsNone(result["selected"])
                self.assertIn(reason, result["candidates"][0]["rejection_reasons"])

    def test_all_occupied_sections_contribute_to_aperture(self):
        scene = self.scene()
        scene["object"]["boxes"].append(box([0, .1, -.01], [.01, .01, .01]))
        for mode in MODES:
            row = select_grasp(scene, mode)["candidates"][0]
            self.assertAlmostEqual(row["width_m"], .12)
            self.assertIn("width_exceeds_aperture", row["rejection_reasons"])

    def test_disconnected_tangent_faces_are_not_an_inserted_section(self):
        boxes = [box([sign*.022, 0, -.02], [.01, .02, .01]) for sign in (-1, 1)]
        self.assertIsNone(local_pad_geometry(boxes, pose()))

    def test_corrected_axis_and_pose_drive_usage_and_ranking(self):
        scene = self.scene()
        scene["object"]["usage_regions"]["head"] = [box([.025, 0, -.01], [.005, .02, .02])]
        candidate = copy.deepcopy(scene["candidates"][0])
        candidate.update(id="shifted", T_object_gripper=pose([.025, 0, 0]).tolist(),
                         approach_ray_origin_object=[.025, 0, -.2],
                         approach_point_object=[0, 0, -.02],
                         geometry_preparation={"original_T_object_gripper": pose().tolist()})
        scene["candidates"].append(candidate)
        for mode in ("FS", "A2"):
            self.assertEqual(select_grasp(scene, mode)["selected"]["id"], "handle")
        result = select_grasp(scene, "A1")
        shifted = result["candidates"][1]
        self.assertTrue(shifted["in_human_region"])
        self.assertAlmostEqual(shifted["distance_m"], .125)
        np.testing.assert_allclose(shifted["approach_point_object"], [.025, 0, -.02])
        self.assertEqual(result["selected"]["id"], "handle")

    def test_imported_candidates_insert_before_all_mode_selection(self):
        p = np.random.default_rng(7).uniform(-.03, .03, (1000, 3))
        scene = bootstrap_scene("test", p)
        before = copy.deepcopy(scene)
        for mode in MODES:
            row = select_grasp(scene, mode)["selected"]
            self.assertIsNotNone(row)
            t = np.array(row["T_object_gripper"])
            raw = np.array(row["geometry_preparation"]["original_T_object_gripper"])
            self.assertGreater(np.linalg.norm(t[:3, 3]-raw[:3, 3]), .001)
            self.assertAlmostEqual(row["proxy_contact"]["center_y_m"], 0.)
            for c in points(t, row["proxy_contact"]["contact_points_gripper"]):
                self.assertTrue(any(contains(b, c) for b in scene["object"]["boxes"]))
        self.assertEqual(scene, before)

    def test_invalid_policy_frame_and_forward_origin_fail_closed(self):
        for key, value in [("width_policy", "unknown"), ("grasp_frame", "robotiq_base")]:
            scene = self.scene(); scene["gripper"][key] = value
            with self.assertRaises(ValueError): select_grasp(scene)
        scene = self.scene(); scene["candidates"][0]["approach_ray_origin_object"] = [0, 0, .2]
        with self.assertRaisesRegex(ValueError, "behind"): select_grasp(scene)
        scene = self.scene(); del scene["candidates"][0]["approach_ray_origin_object"]
        with self.assertRaisesRegex(ValueError, "requires an approach ray"): select_grasp(scene)
