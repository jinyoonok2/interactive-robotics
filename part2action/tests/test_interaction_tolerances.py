import numpy as np
from scipy.spatial.transform import Rotation
from diagnostics.interaction_tolerances import (
    conditions, controller_gains, grasp_window, perturb_actions, replay_fidelity_passes, tracking_metrics,
)


def sample():
    a = np.zeros((35, 7)); a[:, 6] = .04; a[5:, 0] = .3; a[25:, 6] = 0.
    return a, grasp_window(a, ['Grasp the mug at its handle'] * len(a))


def test_identity_and_temporal_interventions_preserve_arm_and_do_not_wrap():
    a, w = sample(); c = conditions()
    np.testing.assert_array_equal(perturb_actions(a, w, c[0]), a)
    for row in c[-2:]:
        changed = perturb_actions(a, w, row)
        np.testing.assert_array_equal(changed[:, :6], a[:, :6])
        assert np.flatnonzero(changed[:, 6] <= .02)[0] == w['close'] + row['close_offset_steps']
        assert changed[0, 6] == .04 and changed[-1, 6] == 0.


def test_position_ramp_and_local_rotation_are_realized_as_composition():
    a, w = sample(); a[:, 3:6] = [.4, .3, .2]
    c = dict(conditions()[0], translation_m=[.01, 0, 0], rotation_deg=[0, 10, 0])
    b = perturb_actions(a, w, c)
    np.testing.assert_array_equal(b[:w['start'], :3], a[:w['start'], :3])
    np.testing.assert_allclose(b[-1, :3] - a[-1, :3], [.01, 0, 0])
    expected = Rotation.from_euler('xyz', a[-1, 3:6]) * Rotation.from_rotvec(np.deg2rad([0, 10, 0]))
    np.testing.assert_allclose(Rotation.from_euler('xyz', b[-1, 3:6]).as_matrix(), expected.as_matrix())
    assert len(conditions()) == 15
    np.testing.assert_array_equal(a[:, 6], sample()[0][:, 6])


def test_bad_replay_cannot_become_a_robustness_failure_label():
    tcp = np.tile([0., 0., 0., 1., 0., 0., 0.], (5, 1))
    changed = tcp.copy(); changed[:, 4] += .03
    m = tracking_metrics(changed, tcp, np.zeros(5), np.zeros(5))
    assert not replay_fidelity_passes(m, {'tcp_rmse_m': .01})
    m['tcp_rmse_m'] = float('nan')
    assert not replay_fidelity_passes(m, {'tcp_rmse_m': .01})
    for mode in ['closed', 'touch', 'no_close']:
        a, _ = sample(); text = ['Grasp the mug at its handle'] * len(a)
        if mode == 'closed': a[0, 6] = 0
        if mode == 'touch': text = ['Touch the mug at its handle'] * len(a)
        if mode == 'no_close': a[:, 6] = .04
        try: grasp_window(a, text)
        except ValueError: pass
        else: raise AssertionError(mode)


def test_control_schedule_uses_source_event_not_intervened_closure():
    a, w = sample()
    gains = controller_gains(a, w)
    assert gains[w['close']-1][0] == .005
    assert gains[w['close']][0] == .0001
    changed = perturb_actions(a, w, conditions()[-2])
    assert changed[w['close']-2, 6] == 0.
    # The runner computes this once from original actions; gains cannot receive
    # the perturbed event as an implicit second intervention.
    assert gains[w['close']-2][0] == .005


def test_selection_skips_missing_metadata_and_never_uses_test_episodes():
    import json
    import tempfile
    from pathlib import Path
    import h5py
    from run_interaction_tolerance_pilot import select_cases
    a, _ = sample()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); (root/'demos').mkdir()
        (root/'episodes_meta_train.json').write_text(json.dumps([
            {'ep_id':'train1', 'chain_params':[{'skill_name':'grasp_obj','params':{'part_grasp':'handle'}}]},
        ]))
        with h5py.File(root/'demos/mug.hdf5','w') as f:
            for i, ep in enumerate([None, 'test_only', 'train1']):
                d=f.create_group(f'data/demo_{i}');d.create_dataset('actions',data=a)
                d.create_dataset('skill_instructions',data=['Grasp the mug at its handle']*len(a),dtype=h5py.string_dtype())
                if ep:
                    d.create_dataset('metadata/ep_id',data=ep)
                    d.create_dataset('metadata/obj_id',data='1')
        selected, skipped = select_cases(root,['mug'],1,42)
        assert selected[0]['episode_id']=='train1'
        assert skipped['missing episode metadata']==1


def test_manifest_subset_preserves_original_scene_seed_and_rejects_invalid_indices():
    from run_interaction_tolerance_pilot import manifest_cases
    previous = {'seed':42, 'cases':[{'demo':f'demo_{i}'} for i in range(20)]}
    subset = manifest_cases(previous, [8])
    assert subset == [{'demo':'demo_8', 'construction_seed':50}]
    assert 'construction_seed' not in previous['cases'][8]
    # Subsetting an already filtered manifest must not re-index its scene seed.
    assert manifest_cases({'seed':42, 'cases':subset}, [0]) == subset
    assert [c['construction_seed'] for c in manifest_cases(previous, [15, 8])] == [57, 50]
    for indices in [[], [-1], [20], [8, 8]]:
        try: manifest_cases(previous, indices)
        except ValueError: pass
        else: raise AssertionError(indices)


def test_larger_offsets_preserve_reference_and_change_only_one_variable_at_a_time():
    a,w=sample()
    small=conditions(); larger=conditions(.02,20.,4)
    assert larger[0] == small[0]
    for c in larger[1:]:
        active = sum([any(c['translation_m']), any(c['rotation_deg']), c['close_offset_steps'] != 0])
        assert active == 1
        changed=perturb_actions(a,w,c)
        if any(c['translation_m']):
            np.testing.assert_allclose(changed[-1,:3]-a[-1,:3],c['translation_m'])
            assert np.linalg.norm(c['translation_m']) == .02
        if any(c['rotation_deg']):
            relative=Rotation.from_euler('xyz',a[-1,3:6]).inv()*Rotation.from_euler('xyz',changed[-1,3:6])
            np.testing.assert_allclose(np.rad2deg(relative.magnitude()),20.)
        if c['close_offset_steps']:
            assert np.flatnonzero(changed[:,6]<=.02)[0] == w['close']+c['close_offset_steps']
    for values in [(0,20.,4),(.02,float('nan'),4),(.02,20.,0),(.02,20.,1.5)]:
        try: conditions(*values)
        except ValueError: pass
        else: raise AssertionError(values)
