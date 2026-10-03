import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from intent_handover.core import select_grasp
from intent_handover.dataset import import_dataset
from intent_handover.demos import load_scene
from intent_handover.geometry import inverse, moved, points, pose
from intent_handover.prediction_bridge import decode_prediction
from intent_handover.replay_audit import audit_replay
from intent_handover.text2hoi import predict
from intent_handover.workflows import ablate, export_selection, manifest_scenes
from intent_handover.cli import main


class ReleaseIntegrityTests(unittest.TestCase):
    def trial(self, scene, selection):
        t = np.array(selection['selected']['T_object_gripper'])
        world_object = pose([.5, 0, .6])
        return {'schema_version':'handover.trial.v1','units':'m', 'object_id':scene['object']['id'],
                'T_object_gripper':t.tolist(), 'target_T_world_gripper':(world_object@t).tolist(),
                'max_opening_m':scene['gripper']['max_opening_m'],
                'object_boxes':copy.deepcopy(scene['object']['boxes']),
                'usage_boxes':copy.deepcopy(scene['object']['usage_regions'][scene['intent']['human_region']]),
                'hand_boxes_world':[moved(b,world_object) for b in scene['receiving_hand']['boxes']],
                'palm_position_world':points(world_object,scene['receiving_hand']['center']).tolist()}

    def test_fixed_receiver_cannot_move_with_object_or_lose_geometry(self):
        scene=load_scene('hammer')
        scene['target_T_world_object']=pose([.5,0,.6]).tolist()
        scene['receiver_protocol']={'policy':'fixed_world'}
        scene['receiver']={'id':'fixed','seed':7,'side':'left','static_world':True,
                           'T_world_hand':pose([.3,0,.7]).tolist()}
        scene['evaluation_split']='S1'
        scene['evaluation_split_provenance']='Authored regions for local integration'
        selection=select_grasp(scene);trial=self.trial(scene,selection)
        trial['receiver_protocol']=copy.deepcopy(scene['receiver_protocol'])
        trial['receiver']=copy.deepcopy(scene['receiver'])
        trial['target_T_world_object']=copy.deepcopy(scene['target_T_world_object'])
        trial['split']='S1'
        trial['evaluation_split_provenance']=scene['evaluation_split_provenance']
        self.assertEqual(audit_replay(scene,selection,trial)['status'],'equivalent')
        trial['target_T_world_gripper'][0][3]+=.1
        trial['receiver']['T_world_hand'][0][3]+=.1
        del trial['hand_boxes_world']
        trial['split']='S0'
        del trial['evaluation_split_provenance']
        issues=audit_replay(scene,selection,trial)['issues']
        for reason in ('fixed_object_target_changed','fixed_receiver_pose_changed','receiver_proxy_geometry_changed',
                       'evaluation_split_changed','evaluation_split_provenance_changed'):
            self.assertIn(reason,issues)

    def test_unmodified_proxy_replay_is_equivalent_but_not_physics_certified(self):
        scene=load_scene('hammer'); selection=select_grasp(scene)
        trial=self.trial(scene,selection)
        audit=audit_replay(scene,selection,trial)
        self.assertEqual(audit['status'],'equivalent')
        self.assertFalse(audit['physical_grasp_verified'])

    def test_post_selection_mesh_fit_and_receiver_retarget_are_detected(self):
        scene=load_scene('hammer'); selection=select_grasp(scene)
        trial=self.trial(scene,selection)
        trial['T_object_gripper'][0][3]+=.01
        trial['asset_contact_fit']={'status':'bilateral_surface_fit','width_m':.003}
        trial['T_tcp_asset_tool']=pose().tolist()
        trial['experiment']={'paired_receiver_preserved':False}
        trial['max_opening_m']=.09
        result=audit_replay(scene,selection,trial)
        self.assertEqual(result['status'],'not_equivalent')
        for issue in ('grasp_pose_changed','replay_width_changed','maximum_aperture_changed',
                      'receiver_object_relation_changed','paired_receiver_not_preserved',
                      'asset_tool_frame_requires_explicit_method_calibration'):
            self.assertIn(issue,result['issues'])

    def test_stale_selection_and_changed_delivery_are_detected(self):
        scene=load_scene('hammer'); selection=select_grasp(scene)
        trial=self.trial(scene,selection)
        world_object=np.array(trial['target_T_world_gripper'])@inverse(trial['T_object_gripper'])
        delivery={'schema_version':'handover.delivery.v1','units':'m', 'object_id':scene['object']['id'],
                  'grasp_id':selection['selected']['id'], 'T_world_object':world_object.tolist(),
                  'T_world_gripper':trial['target_T_world_gripper']}
        self.assertEqual(audit_replay(scene,selection,trial,delivery)['status'],'equivalent')
        delivery['T_world_object'][0][3]+=.1
        selection['selected']['width_m']+=.01
        result=audit_replay(scene,selection,trial,delivery)
        self.assertIn('selection_stale_or_modified',result['issues'])
        self.assertIn('delivery_target_changed',result['issues'])

    def test_local_policy_is_not_silently_treated_as_global_replay(self):
        from intent_handover.dataset import bootstrap_scene
        cloud=np.random.default_rng(11).uniform(-.03,.03,(200,3))
        scene=bootstrap_scene('test',cloud);selection=select_grasp(scene)
        result=audit_replay(scene,selection,self.trial(scene,selection))
        self.assertEqual(result['status'],'not_equivalent')
        self.assertIn('width_policy_not_preserved',result['issues'])

    def test_failed_ablation_invalidates_old_completion_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ablate([load_scene('hammer')],root)
            with patch('intent_handover.workflows.export_selection',side_effect=ValueError('interrupted export')):
                with self.assertRaisesRegex(ValueError,'interrupted'): ablate([load_scene('hammer')],root)
            for name in ('experiment.json','replay.json'):
                self.assertEqual(json.loads((root/name).read_text())['status'],'failed')
                self.assertEqual(json.loads((root/name).read_text())['schema_version'],'handover.incomplete.v1')

    def test_failed_dataset_and_selection_cannot_reuse_old_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'dataset.json').write_text('{"schema_version":"handover.dataset.v1","objects":[]}')
            with self.assertRaises(OSError): import_dataset(root/'missing.yaml',root)
            with self.assertRaises(ValueError): manifest_scenes(root/'dataset.json')
            export_selection(load_scene('hammer'),root)
            with patch('intent_handover.workflows.write_report',side_effect=OSError('write failed')):
                with self.assertRaises(OSError): export_selection(load_scene('hammer'),root)
            self.assertEqual(json.loads((root/'hammer_FS.json').read_text())['status'],'failed')

    def test_prediction_checks_hashes_and_invalidates_old_metadata_before_loading_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'metadata.json').write_text('{"status":"succeeded"}')
            args=SimpleNamespace(output=root,frames=1,hand='right',prompt='Grasp a bottle.',checkpoints=root)
            with patch('intent_handover.text2hoi.verify_weights',side_effect=ValueError('Checksum mismatch')):
                with self.assertRaisesRegex(ValueError,'Checksum'): predict(args)
            self.assertEqual(json.loads((root/'metadata.json').read_text())['status'],'failed')
            with self.assertRaisesRegex(ValueError,'incomplete or failed'):
                decode_prediction(load_scene('bottle'),root/'prediction.npz',root)

    def test_prediction_and_metadata_from_different_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'prediction.npz').write_bytes(b'changed')
            (root/'metadata.json').write_text(json.dumps({'status':'succeeded','prediction_sha256':'bad'}))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                decode_prediction(load_scene('bottle'),root/'prediction.npz',root)

    def test_cli_infeasible_selection_returns_nonzero_with_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=load_scene('hammer');scene['gripper']['max_opening_m']=.001
            (root/'input.json').write_text(json.dumps(scene))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(['select',str(root/'input.json'),'--output',str(root/'out')])
            self.assertEqual(caught.exception.code,2)
            self.assertEqual(json.loads((root/'out/hammer_FS.json').read_text())['status'],'no_feasible_grasp')
            self.assertTrue((root/'out/hammer_FS.html').is_file())

    def test_cli_audit_writes_differences_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=load_scene('hammer');selection=select_grasp(scene)
            trial=self.trial(scene,selection);trial['max_opening_m']=.01
            for name,data in [('scene',scene),('selection',selection),('trial',trial)]:
                (root/f'{name}.json').write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(['audit-replay','--scene',str(root/'scene.json'),'--selection',str(root/'selection.json'),
                          '--trial',str(root/'trial.json'),'--output',str(root/'audit.json')])
            self.assertEqual(caught.exception.code,2)
            self.assertIn('maximum_aperture_changed',json.loads((root/'audit.json').read_text())['issues'])
