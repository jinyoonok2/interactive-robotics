"""Catch missing runtime globals and enforce trained-arm rollout preflights."""
import builtins
import dis
import json
from pathlib import Path
import tempfile
import types
from unittest.mock import patch

import evaluate_visual_oracle_pose_v3 as adapter


def missing_globals(function):
    missing=set()
    def visit(code):
        for operation in dis.get_instructions(code):
            if operation.opname=='LOAD_GLOBAL':
                name=operation.argval
                if name not in function.__globals__ and not hasattr(builtins,name):
                    missing.add(name)
        for constant in code.co_consts:
            if isinstance(constant,types.CodeType):
                visit(constant)
    visit(function.__code__)
    return missing


def test_all_evaluation_runtime_globals_are_defined():
    assert missing_globals(adapter.evaluate)==set()
    assert missing_globals(adapter.main)==set()
    assert missing_globals(adapter.require_smoke)==set()
    # Prove the check detects the exact class of bug that stopped V2 before
    # any physical rollout; importing a module alone cannot detect this.
    scope={}
    exec('def broken():\n    return omitted_reference_helper()',scope)
    assert missing_globals(scope['broken'])=={'omitted_reference_helper'}


def test_reference_path_helper_loads_existing_receipt():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        path=adapter.oracle_path(root,17)
        assert path==root/'oracle_references/017.json'
        adapter.dump(path,{'status':'completed','trial_index':17})
        assert adapter.read(path)['trial_index']==17


def test_full_evaluation_requires_both_current_checkpoint_smoke_receipts():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for arm in adapter.ARMS:
            checkpoint=root/'models'/arm/'last.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(arm.encode())
        try:
            adapter.require_smoke(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Missing smoke checks must block full evaluation')
        for arm in adapter.ARMS:
            checkpoint=root/'models'/arm/'last.pt'
            adapter.dump(root/'evaluation_recovery_v3/smoke'/arm/'completed.json',
                         {'status':'completed','checkpoint_sha256':adapter.sha(checkpoint)})
        adapter.require_smoke(root)
        (root/'models/oracle_pose/last.pt').write_bytes(b'changed weights')
        try:
            adapter.require_smoke(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError('A stale checkpoint must invalidate its smoke test')


def test_smoke_dispatch_checks_both_arms_without_changing_plan():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        plan={'config':{'evaluation':{'shards':4}}}
        adapter.dump(root/'plan.json',plan)
        adapter.dump(root/'evaluation_revision_v3.json',{'source_sha256':{}})
        for index,expected in [(0,0),(1,4)]:
            with patch('sys.argv',['evaluate','--mode','smoke','--out-dir',str(root),'--index',str(index)]), \
                 patch.object(adapter,'verify'),patch.object(adapter,'evaluate') as call:
                adapter.main()
                args,actual_plan=call.call_args.args
                assert args.index==expected
                assert actual_plan==plan
