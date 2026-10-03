import unittest
import numpy as np
from intent_handover.core import select_grasp
from intent_handover.demos import load_scene
from intent_handover.execution import delivery_target, minimum_rotation
from intent_handover.geometry import pose


class ExecutionTests(unittest.TestCase):
    def skeleton(self):
        return {"schema_version":"handover.skeleton.v1", "units":"m",
            "left_shoulder":[-.2,0,1.3], "right_shoulder":[.2,0,1.3],
            "elbow":[0,0,1.0], "wrist":[0,.4,1.0], "desk_height_m":.7,
            "facing_hint_world":[0,1,0], "extension_axis_world":[1,0,0]}

    def test_paper_radius_height_extension_and_pose_chain(self):
        scene = load_scene('hammer'); selection = select_grasp(scene)
        skeleton = self.skeleton()
        skeleton['T_robot_world'] = pose([1,2,3]).tolist()
        result = delivery_target(scene, selection, skeleton)
        self.assertAlmostEqual(result['comfortable_radius_m'],.5)
        np.testing.assert_allclose(result['torso_center'],[0,0,1])
        np.testing.assert_allclose(np.array(result['T_world_object'])[:3,3],[0,.5,1])
        np.testing.assert_allclose(result['hand_direction_world'],[0,np.cos(np.pi/12),np.sin(np.pi/12)],atol=1e-9)
        np.testing.assert_allclose(np.array(result['T_world_object'])@selection['selected']['T_object_gripper'],result['T_world_gripper'])
        np.testing.assert_allclose(np.array(skeleton['T_robot_world'])@result['T_world_gripper'],result['T_robot_gripper'])

    def test_minimum_rotation_parallel_antipodal_and_general(self):
        for source,target in [([1,0,0],[1,0,0]),([1,0,0],[-1,0,0]),([1,0,0],[0,1,0])]:
            r=minimum_rotation(source,target)
            np.testing.assert_allclose(r@source,target,atol=1e-9)
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(r),1.)

    def test_facing_hint_and_invalid_extension_plane(self):
        scene=load_scene('hammer'); selection=select_grasp(scene)
        skeleton=self.skeleton(); skeleton['facing_hint_world']=[0,-1,0]
        self.assertLess(delivery_target(scene,selection,skeleton)['T_world_object'][1][3],0)
        skeleton['extension_axis_world']=[0,1,0]
        with self.assertRaisesRegex(ValueError,'perpendicular'):
            delivery_target(scene,selection,skeleton)
