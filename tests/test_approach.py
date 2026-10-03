import unittest
import numpy as np
from intent_handover.geometry import approach_intersection, box
from intent_handover.demos import build_scene
from intent_handover.core import select_grasp


class ApproachTests(unittest.TestCase):
    def test_nearest_surface_and_parallel_miss(self):
        boxes=[box([0,0,0],[1,1,1]), box([4,0,0],[1,1,1])]
        np.testing.assert_allclose(approach_intersection(boxes,[8,0,0],[-1,0,0]),[5,0,0])
        self.assertIsNone(approach_intersection(boxes,[8,2,0],[-1,0,0]))
        with self.assertRaisesRegex(ValueError,'outside'):
            approach_intersection(boxes,[0,0,0],[1,0,0])

    def test_ray_overrides_incorrect_surface_annotation(self):
        scene=build_scene('hammer')
        scene['candidates'][0]['approach_point_object']=[0,0,.06]
        result=select_grasp(scene)
        self.assertTrue(result['candidates'][0]['in_human_region'])
        self.assertIn('human_usage_region',result['candidates'][0]['rejection_reasons'])
        self.assertEqual(result['selected']['id'],'free_top')
