import unittest
import numpy as np
from intent_handover.demos import load_scene
from intent_handover.geometry import points, pose
from intent_handover.prediction_bridge import geometry_to_scene, rotation6d


class BridgeTests(unittest.TestCase):
    def test_interleaved_rotation_layout(self):
        r = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        np.testing.assert_allclose(rotation6d(r[:,:2].reshape(6)), r)
        with self.assertRaises(ValueError):
            rotation6d([1,1,0,0,0,0])

    def test_hand_is_restored_to_canonical_object_frame(self):
        rng = np.random.default_rng(10)
        vertices = rng.normal(0,.03,(778,3))
        joints = rng.normal(0,.03,(16,3))
        scene = load_scene("bottle")
        identity = np.array([0,0,0,1,0,0,1,0,0.])
        original = geometry_to_scene(scene,vertices,joints,identity,"right")
        t = pose([.2,.5,-.1],[[0,-1,0],[1,0,0],[0,0,1]])
        obj = np.r_[t[:3,3],t[:3,:2].reshape(6)]
        transformed = geometry_to_scene(scene,points(t,vertices),points(t,joints),obj,"right")
        for key in ("center","direction","palm_normal"):
            np.testing.assert_allclose(original["receiving_hand"][key],transformed["receiving_hand"][key],atol=1e-9)
        np.testing.assert_allclose(transformed["receiving_hand"]["mesh"]["vertices"],vertices,atol=1e-9)
        self.assertNotIn("mesh",scene["receiving_hand"])

    def test_flip_normal_does_not_change_hand_direction(self):
        rng = np.random.default_rng(11)
        v,j = rng.normal(size=(778,3)),rng.normal(size=(16,3))
        obj = [0,0,0,1,0,0,1,0,0]
        a = geometry_to_scene(load_scene("bottle"),v,j,obj,"right")
        b = geometry_to_scene(load_scene("bottle"),v,j,obj,"right",True)
        np.testing.assert_allclose(a["receiving_hand"]["palm_normal"],-np.array(b["receiving_hand"]["palm_normal"]))
        np.testing.assert_allclose(a["receiving_hand"]["direction"],b["receiving_hand"]["direction"])


if __name__ == "__main__":
    unittest.main()
