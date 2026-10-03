import copy
import unittest
import numpy as np
from intent_handover.core import MODES, select_grasp
from intent_handover.geometry import box, points, pose
from intent_handover.mesh_geometry import mesh_digest, mesh_pad_geometry, point_mesh_distance
from intent_handover.replay_audit import audit_replay


class AssetContractTests(unittest.TestCase):
    def scene(self):
        mesh={'vertices':[[x,y,z] for x in (-.005,.005) for y in (-.02,.02) for z in (-.024,-.004)],
              'faces':[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                       [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]}
        window={'x':[-.01,.01],'z':[-.032,0.]}
        offset=pose([0,0,.01]);section=mesh_pad_geometry(mesh,pose(),window)
        contract={'schema_version':'handover.asset_gripper.v1','pad_window_gripper_m':window,'T_asset_tool_grasp_frame':offset.tolist(),
                  'robot':{'usd':'test.usd'},'mesh_sha256':mesh_digest(mesh)}
        prep={'schema_version':'handover.asset_candidate.v1','status':'bilateral_surface_fit','candidate_id':'fit',
              'original_T_object_gripper':pose().tolist(),'T_object_gripper':pose().tolist(),
              'width_m':.04,'contact_points_gripper':section['contact_points_gripper'],
              'contact_points_asset_tool':points(offset,section['contact_points_gripper']).tolist(),
              'bilateral_distance_m':[0,0], **{k:contract[k] for k in ('robot','mesh_sha256','T_asset_tool_grasp_frame')}}
        return {'schema_version':'handover.scene.v1','units':'m',
                'object':{'id':'test','mesh':mesh,'boxes':[box([0,0,-.014],[.005,.02,.01])],
                          'usage_regions':{'human':[box([1,0,0],[.1,.1,.1])]}},
                'intent':{'object_id':'test','human_region':'human'},
                'receiving_hand':{'center':[.2,0,0],'direction':[0,0,1]},
                'gripper':{'max_opening_m':.085,'width_policy':'asset_mesh_pad',
                           'geometry_contract':contract,'score_reference_point_gripper':[0,0,-.016]},
                'candidates':[{'id':'fit','T_object_gripper':pose().tolist(),'approach_ray_origin_object':[0,0,-.2],
                               'geometry_preparation':prep}]}

    def test_mesh_pad_is_independently_measured_and_selected(self):
        scene=self.scene();before=copy.deepcopy(scene)
        for mode in MODES:
            row=select_grasp(scene,mode)['selected']
            self.assertIsNotNone(row)
            self.assertAlmostEqual(row['width_m'],.04)
            self.assertEqual(row['width_source'],'asset_mesh_pad')
            self.assertIsNone(row['proxy_contact'])
            self.assertEqual(row['mesh_contact']['source'],'clipped_triangle_mesh')
        self.assertEqual(scene,before)

    def test_invalidated_evidence_rejects_candidate_in_every_mode(self):
        edits=[('asset_width_binding_changed',lambda s:s['candidates'][0]['geometry_preparation'].update(width_m=.03)),
               ('asset_candidate_identity_changed',lambda s:s['candidates'][0]['geometry_preparation'].update(candidate_id='other')),
               ('asset_mesh_binding_changed',lambda s:s['gripper']['geometry_contract'].update(mesh_sha256='changed')),
               ('asset_prepared_pose_changed',lambda s:s['candidates'][0]['geometry_preparation'].update(T_object_gripper=pose([.01,0,0]).tolist())),
               ('asset_pad_contact_distance_exceeded',lambda s:s['candidates'][0]['geometry_preparation'].update(bilateral_distance_m=[.001,0])),
               ('asset_candidate_preparation_failed',lambda s:s['candidates'][0]['geometry_preparation'].update(status='failed'))]
        for reason, edit in edits:
            scene=self.scene();edit(scene)
            for mode in MODES:
                result=select_grasp(scene,mode)
                self.assertIsNone(result['selected'])
                self.assertIn(reason,result['candidates'][0]['rejection_reasons'])

    def test_contact_on_pad_plane_but_not_object_is_rejected(self):
        scene=self.scene();prep=scene['candidates'][0]['geometry_preparation']
        prep['contact_points_gripper'][0][0]=.009
        prep['contact_points_asset_tool']=points(prep['T_asset_tool_grasp_frame'],prep['contact_points_gripper']).tolist()
        result=select_grasp(scene)
        self.assertIn('asset_contacts_off_object_surface',result['candidates'][0]['rejection_reasons'])

    def test_projection_feasibility_is_separate_from_local_actuator_aperture(self):
        scene=self.scene();mesh=scene['object']['mesh']
        remote=np.asarray(mesh['vertices'])*[1,10,1]+[1,0,1]
        mesh['vertices']+=remote.tolist()
        mesh['faces']+=(np.asarray(mesh['faces'])+8).tolist()
        digest=mesh_digest(mesh)
        scene['gripper']['geometry_contract']['mesh_sha256']=digest
        scene['candidates'][0]['geometry_preparation']['mesh_sha256']=digest
        self.assertAlmostEqual(select_grasp(scene)['selected']['width_m'],.04)
        scene['gripper']['feasibility_width_policy']='object_projection'
        for mode in MODES:
            result=select_grasp(scene,mode)
            self.assertIsNone(result['selected'])
            row=result['candidates'][0]
            self.assertAlmostEqual(row['width_m'],.04)
            self.assertAlmostEqual(row['feasibility_width_m'],.4)
            self.assertIn('object_projection_exceeds_aperture',row['rejection_reasons'])

    def test_large_triangle_corners_inside_pad_are_not_lost(self):
        mesh={'vertices':[[-1,-.02,-1],[1,-.02,-1],[0,-.02,1],[-1,.02,-1],[1,.02,-1],[0,.02,1]],
              'faces':[[0,1,2],[3,4,5]]}
        section=mesh_pad_geometry(mesh,pose(),{'x':[-.01,.01],'z':[-.032,0]})
        self.assertAlmostEqual(section['width_m'],.04)
        self.assertAlmostEqual(point_mesh_distance(mesh,[0,0,0]),.02)
        self.assertAlmostEqual(point_mesh_distance(mesh,[0,.02,0]),0)

    def test_explicit_replay_aperture_and_preparation_contract_match(self):
        scene=self.scene();selection=select_grasp(scene)
        trial={'schema_version':'handover.trial.v1','units':'m','object_id':'test',
               'T_object_gripper':pose().tolist(),'target_T_world_gripper':pose().tolist(),
               'max_opening_m':.085,'object_boxes':scene['object']['boxes'],
               'usage_boxes':scene['object']['usage_regions']['human'],'palm_position_world':[.2,0,0],
               'gripper_opening_m':.04,'gripper_opening_source':'asset_mesh_pad',
               'grasp_contract':selection['grasp_contract'],
               'object_mesh_object':scene['object']['mesh'],
               'method_selection':{**selection['selected'],'mode':'FS'},'asset_robot':{'usd':'test.usd'}}
        self.assertEqual(audit_replay(scene,selection,trial)['status'],'equivalent')
        scene['gripper']['feasibility_width_policy']='object_projection'
        selection=select_grasp(scene)
        trial['grasp_contract']=selection['grasp_contract']
        trial['method_selection']={**selection['selected'],'mode':'FS'}
        trial['stability_width_m']=.04
        self.assertEqual(audit_replay(scene,selection,trial)['status'],'equivalent')
        trial['stability_width_m']=.05
        self.assertIn('stability_width_changed',audit_replay(scene,selection,trial)['issues'])
        trial['stability_width_m']=.04
        trial['T_tcp_asset_tool']=pose([0,0,.1]).tolist()
        trial['T_asset_tool_grasp_frame']=copy.deepcopy(scene['gripper']['geometry_contract']['T_asset_tool_grasp_frame'])
        trial['asset_contact_fit']=copy.deepcopy(selection['selected']['geometry_preparation'])
        scene['receiving_hand']['mesh']=copy.deepcopy(scene['object']['mesh'])
        trial['hand_mesh_world']=copy.deepcopy(scene['receiving_hand']['mesh'])
        self.assertEqual(audit_replay(scene,selection,trial)['status'],'equivalent')
        trial['hand_mesh_world']['vertices'][0][0]+=.001
        trial['object_mesh_object']=copy.deepcopy(trial['object_mesh_object'])
        trial['object_mesh_object']['vertices'][0][0]+=.001
        trial['asset_contact_fit']['bilateral_distance_m']=[.001,0]
        issues=audit_replay(scene,selection,trial)['issues']
        for reason in ('receiver_mesh_geometry_changed','object_mesh_changed','resolved_asset_contact_verification_failed'):
            self.assertIn(reason,issues)
        trial['gripper_opening_m']=.03
        self.assertIn('replay_width_changed',audit_replay(scene,selection,trial)['issues'])

    def test_unchanged_winner_does_not_conceal_stale_avoidance_score(self):
        scene=self.scene();selection=select_grasp(scene)
        trial={'schema_version':'handover.trial.v1','units':'m','object_id':'test',
               'T_object_gripper':pose().tolist(),'target_T_world_gripper':pose().tolist(),
               'max_opening_m':.085,'object_boxes':scene['object']['boxes'],
               'usage_boxes':scene['object']['usage_regions']['human'],'palm_position_world':[.2,0,0],
               'gripper_opening_m':.04,'gripper_opening_source':'asset_mesh_pad',
               'grasp_contract':selection['grasp_contract'],'object_mesh_object':scene['object']['mesh'],
               'method_selection':{**selection['selected'],'mode':'FS'}}
        scene['receiving_hand']['center'][0]+=.01
        self.assertEqual(select_grasp(scene)['selected']['id'],selection['selected']['id'])
        issues=audit_replay(scene,selection,trial)['issues']
        self.assertIn('selection_stale_or_modified',issues)
        self.assertIn('method_score_evidence_changed',issues)
