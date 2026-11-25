import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from diagnostics.evaluation_scenarios import load_scenarios
from fixed_scenarios import action_prefix, last_info, validate_manifest, batch_observations


def test_action_execution_matches_one_step_budget_and_does_not_mutate_policy_output():
    predicted = np.zeros((1,8,7)); predicted[0,:,0] = np.arange(8)
    predicted[0,:,6] = [-.2,.3,.01,.02,.03,.04,.01,0.]
    original = predicted.copy()
    one = action_prefix(predicted,1,120)
    np.testing.assert_array_equal(one, np.zeros((1,7)))
    assert action_prefix(predicted,8,2).shape == (2,7)
    assert action_prefix(predicted,8,2)[1,6] == .04
    np.testing.assert_array_equal(predicted,original)
    for bad in [predicted[0], np.full((1,8,7),np.nan), np.zeros((2,8,7))]:
        try: action_prefix(bad,1,120)
        except ValueError: pass
        else: raise AssertionError('Invalid actions accepted')


def test_final_native_success_is_not_wrapper_time_limit_or_earlier_history():
    info = last_info({'Success':np.array([True,False]), 'Completion Rate':[.2,.4],
                      'Steps':[119,120], 'ep_id':['123','123']})
    assert info['Success'] is False and info['Steps'] == 120
    assert info['ep_id'] == '123'


def test_fixed_manifest_is_exact_and_preflight_preserves_original_scenarios():
    metadata = Path(__file__).resolve().parents[2]/'datasets/PartInstruct/episodes_meta_test.json'
    cases = load_scenarios(metadata,'test1',['bottle','mug','pliers','scissors'])
    manifest = {'split':'test1','evaluation_seeds':[42,43,44],'scenarios':cases}
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'scenarios.json';p.write_text(json.dumps(manifest))
        _, plan, _, _ = validate_manifest(p,metadata)
        assert len(plan) == 447
        _, subset, _, _ = validate_manifest(p,metadata,[0,50],[43])
        assert subset == [(cases[0],0,43),(cases[50],0,43)]
        bad = copy.deepcopy(manifest);bad['scenarios'][0]['episode']['ep_id']='wrong'
        p.write_text(json.dumps(bad))
        try: validate_manifest(p,metadata)
        except ValueError: pass
        else: raise AssertionError('Mismatched scenario accepted')


def test_direct_observations_match_native_vector_buffer_shapes_and_dtypes():
    observations={
        'input_ids':np.arange(12).reshape(2,1,6),
        'attention_mask':np.ones((2,1,6),dtype=np.int64),
        'agentview_part_mask':np.arange(24).reshape(2,3,4),
        'joint_states':np.zeros((2,7),dtype=np.float64),
    }
    space=SimpleNamespace(spaces={
        'input_ids':SimpleNamespace(shape=(2,6),dtype=np.int64),
        'attention_mask':SimpleNamespace(shape=(2,6),dtype=np.int64),
        'agentview_part_mask':SimpleNamespace(shape=(2,1,3,4),dtype=np.uint8),
        'joint_states':SimpleNamespace(shape=(2,7),dtype=np.float32),
    })
    batch=batch_observations(observations,space)
    assert batch['input_ids'].shape==(1,2,6)
    assert batch['agentview_part_mask'].shape==(1,2,1,3,4)
    assert batch['joint_states'].dtype==np.float32
    np.testing.assert_array_equal(batch['input_ids'].reshape(-1),observations['input_ids'].reshape(-1))
    assert observations['input_ids'].shape==(2,1,6)
    broken=dict(observations,input_ids=np.zeros((2,5)))
    try:batch_observations(broken,space)
    except ValueError:pass
    else:raise AssertionError('Unexpected token count accepted')
