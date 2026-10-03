import copy
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
from intent_handover.annotations import import_panda_candidates, panda_frame, repair_annotation_pose
from intent_handover.core import select_grasp
from intent_handover.demos import load_scene
from intent_handover.geometry import box, inverse, points, pose
from intent_handover.mesh_geometry import mesh_approach_intersection


class AnnotationTests(unittest.TestCase):
    def controls(self):
        return {'gripper_left_center_flat':[.0399,0,.1034],
                'gripper_right_center_flat':[-.0399,0,.1034],
                'gripper_left_tip_flat':[.0399,0,.112204],
                'gripper_right_tip_flat':[-.0399,0,.112204]}

    def test_panda_control_points_define_explicit_tip_and_score_frames(self):
        frame, score = panda_frame(self.controls())
        np.testing.assert_allclose(frame, pose([0,0,.112204],[[0,-1,0],[1,0,0],[0,0,1]]))
        np.testing.assert_allclose(score,[0,0,-.008804])
        np.testing.assert_allclose(points(frame,score),[0,0,.1034])

    def test_only_small_rotation_serialization_errors_are_repaired(self):
        raw=pose();raw[0,0]+=1e-5
        fixed,error=repair_annotation_pose(raw)
        np.testing.assert_allclose(fixed,pose())
        self.assertGreater(error,0)
        for bad in (np.diag([2,1,1,1]),np.diag([-1,1,1,1]),np.full((4,4),np.nan)):
            with self.assertRaises(ValueError): repair_annotation_pose(bad)

    def test_import_preserves_source_region_and_does_not_invent_masks(self):
        scene=load_scene('hammer'); original=copy.deepcopy(scene)
        frame,_=panda_frame(self.controls())
        desired=pose([.02,0,.01]); raw=desired@inverse(frame)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with (root/'grasps.pkl').open('wb') as f:pickle.dump({'hammer':{'original head':[raw,raw]}},f)
            with (root/'controls.pkl').open('wb') as f:pickle.dump(self.controls(),f)
            result=import_panda_candidates(scene,root/'grasps.pkl',root/'controls.pkl',center=False)
        self.assertEqual(scene,original)
        self.assertEqual(result['object']['usage_regions'],scene['object']['usage_regions'])
        self.assertEqual(result['intent'],scene['intent'])
        self.assertEqual(len(result['candidates']),2)
        np.testing.assert_allclose(result['candidates'][0]['T_object_gripper'],desired,atol=1e-15)
        self.assertEqual(result['candidates'][0]['geometry_preparation']['insertion_added_m'],0)
        self.assertEqual(result['candidates'][0]['geometry_preparation']['source_region'],'original head')
        self.assertFalse(result['candidate_source']['paper_candidate_subset_verified'])

    def test_triangle_entry_is_actual_surface_not_proxy_surface(self):
        mesh={'vertices':[[-1,-1,0],[1,-1,0],[0,1,0],[-1,-1,.5],[1,-1,.5],[0,1,.5]],
              'faces':[[0,1,2],[3,4,5]]}
        np.testing.assert_allclose(mesh_approach_intersection(mesh,[0,0,-2],[0,0,1]),[0,0,0])
        np.testing.assert_allclose(mesh_approach_intersection(mesh,[0,0,2],[0,0,-1]),[0,0,.5])
        self.assertIsNone(mesh_approach_intersection(mesh,[2,0,-2],[0,0,1]))
        with self.assertRaisesRegex(ValueError,'outside'):
            mesh_approach_intersection(mesh,[0,0,.2],[0,0,1])
        scene=load_scene('hammer')
        scene['object']['mesh']=mesh
        scene['object']['usage_regions'][scene['intent']['human_region']]=[box([0,0,0],[.01,.01,.01])]
        scene['candidates']=[{'id':'mesh_hit','T_object_gripper':pose([0,0,.01]).tolist(),
                              'approach_ray_origin_object':[0,0,-2]}]
        result=select_grasp(scene)
        self.assertEqual(result['candidates'][0]['approach_source'],'ray/triangle intersection')
        self.assertIn('human_usage_region',result['candidates'][0]['rejection_reasons'])
        np.testing.assert_allclose(result['candidates'][0]['approach_point_object'],[0,0,0])

    def test_score_reference_is_transformed_before_paper_distance(self):
        scene=load_scene('hammer');scene['gripper']['score_reference_point_gripper']=[0,0,-.008804]
        result=select_grasp(scene)
        for row in result['candidates']:
            score=points(row['T_object_gripper'],[0,0,-.008804])
            np.testing.assert_allclose(row['score_point_object'],score)
            distance=np.linalg.norm(score-scene['receiving_hand']['center'])
            self.assertAlmostEqual(row['distance_m'],distance)
            self.assertAlmostEqual(row['avoidance_cost'],row['cosine']-distance)
