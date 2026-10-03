import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from intent_handover.demos import load_scene
from intent_handover.workflows import ablate, apply_intent, run_pipeline
from intent_handover.weights import download_weights, verify_weights


class WorkflowTests(unittest.TestCase):
    def intent(self):
        return {'object_id':'hammer','human_region':'head','robot_region':'handle','hand':'left',
                'text2hoi_prompt':'Grasp a hammer with left hand.','needs_clarification':False}

    def test_structured_intent_is_applied_without_mutating_scene(self):
        scene=load_scene('hammer'); changed=apply_intent(scene,self.intent())
        self.assertEqual(scene['intent']['human_region'],'handle')
        self.assertEqual(changed['intent']['human_region'],'head')
        self.assertEqual(changed['intent']['hand'],'left')

    def test_ambiguous_and_invented_intents_are_rejected(self):
        for edit in ({'needs_clarification':True},{'human_region':'unknown'},{'hand':'either'}, {'object_id':'drill'}):
            intent=self.intent();intent.update(edit)
            with self.assertRaises(ValueError): apply_intent(load_scene('hammer'),intent)

    def test_ablation_exports_four_modes_and_selection_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=ablate([load_scene('hammer')],root)
            self.assertEqual(manifest['modes'],['FS','A1','A2','A3'])
            fs=json.loads((root/'hammer_FS.json').read_text())
            a1=json.loads((root/'hammer_A1.json').read_text())
            self.assertNotEqual(fs['selected']['id'],a1['selected']['id'])
            self.assertEqual(len((root/'ablation.csv').read_text().splitlines()),5)
            replay=json.loads((root/'replay.json').read_text())
            self.assertEqual(len(replay['records']),4)
            self.assertIsNone(replay['paper_simulation_success_rate'])
            self.assertEqual(replay['settings']['A1'],{'usage_constraint':False,'avoidance_ranking':True})

    def test_failed_pipeline_replaces_old_success_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pipeline.json').write_text('{"status":"succeeded"}')
            with self.assertRaisesRegex(ValueError,'Missing MANO'):
                run_pipeline(load_scene('hammer'),root,root,root)
            self.assertEqual(json.loads((root/'pipeline.json').read_text())['status'],'failed')

    def test_download_validates_before_replacing_an_existing_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.pth';path.write_bytes(b'original')
            catalog={'test':{'google_drive_id':'test-id','sha256':hashlib.sha256(b'expected').hexdigest()}}
            def fake_download(**kwargs):
                Path(kwargs['output']).write_bytes(b'bad-download');return kwargs['output']
            with patch('intent_handover.weights.catalog',return_value=catalog), patch.dict('sys.modules',{'gdown':SimpleNamespace(download=fake_download)}):
                with self.assertRaisesRegex(ValueError,'Checksum'): verify_weights(tmp)
                with self.assertRaisesRegex(ValueError,'Download/checksum'): download_weights(tmp,force=True)
            self.assertEqual(path.read_bytes(),b'original')
            self.assertFalse(path.with_suffix('.download').exists())
