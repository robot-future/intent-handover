import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
from intent_handover.dataset import bootstrap_scene, import_dataset, resolve_config


class DatasetTests(unittest.TestCase):
    def test_main_config_preserves_points_and_reports_missing_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs/dataset').mkdir(parents=True)
            (root/'data/han/object').mkdir(parents=True)
            (root/'configs/config.yaml').write_text('defaults:\n  - _self_\n  - dataset: han\n')
            (root/'configs/dataset/han.yaml').write_text('name: han\nobj_root: data/han/object\ndata_obj_pc_path: data/han/obj.pkl\n')
            p=np.random.default_rng(4).uniform(-.025,.025,(100,3))
            asset=root/'data/han/object/test.ply'
            asset.write_text('ply\nformat ascii 1.0\nelement vertex 100\nproperty double x\nproperty double y\nproperty double z\nend_header\n'+'\n'.join(' '.join(map(str,row)) for row in p))
            cache={'object_name':['test','missing'], 'obj_path':{'test':'test.ply','missing':'missing.ply'},'obj_pcs':{'test':p[::2]}}
            with (root/'data/han/obj.pkl').open('wb') as f: pickle.dump(cache,f)
            manifest=import_dataset(root/'configs/config.yaml',root/'out')
            self.assertEqual(len(manifest['objects']),1)
            self.assertEqual(manifest['missing'][0]['object_id'],'missing')
            record=manifest['objects'][0]
            self.assertEqual(record['asset_sha256'],hashlib.sha256(asset.read_bytes()).hexdigest())
            self.assertEqual(record['source_faces'],0)
            np.testing.assert_array_equal(np.load(root/'out/test_points.npy'),p[::2])
            scene=json.loads((root/'out/test_scene.json').read_text())
            np.testing.assert_allclose(scene['object']['surface_points'],p)
            self.assertIn('generated',scene['annotation_status'])
            self.assertEqual(scene['evaluation_split'],'S0')

    def test_degenerate_geometry_and_unconfigured_root_are_rejected(self):
        with self.assertRaises(ValueError): bootstrap_scene('bad',np.zeros((100,3)))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'han.yaml';p.write_text('name: han\nobj_root: data/object\ndata_obj_pc_path: data/obj.pkl\n')
            with self.assertRaisesRegex(ValueError,'project-root'): resolve_config(p)

    def test_original_coordinates_and_names_are_preserved(self):
        p=np.random.default_rng(9).uniform(-.03,.03,(100,3))+[.4,-.2,.1]
        scene=bootstrap_scene('hammers',p)
        self.assertEqual(scene['object']['id'],'hammers')
        np.testing.assert_array_equal(scene['object']['surface_points'],p)
        self.assertEqual(len(scene['candidates']),30)
