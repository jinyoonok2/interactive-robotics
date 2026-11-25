"""Shared execution settings; budgets count actions, never policy predictions."""
import json
import copy
from pathlib import Path

import numpy as np

DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[1]/'configs/evaluation/partinstruct_native.json'


def align_native_environment(environment, native):
    """Keep policy modalities/runtime paths; use released camera/physics thresholds."""
    runtime = {'config_name','shape_meta','device','data_root','obj_class','obj_id',
               'obj_position','skill_chain','chain_params'}
    for key,value in native.items():
        if key not in runtime:
            environment[key] = copy.deepcopy(value)
    return environment


def execution_settings(split='test1', *, config_path=DEFAULT_PROTOCOL_PATH,
                       max_steps=None, execute_steps=None):
    record = json.loads(Path(config_path).read_text())
    if record['schema_version'] != 1 or record['budget_unit'] != 'executed_environment_actions':
        raise ValueError('Unsupported evaluation execution protocol')
    budget = record['action_budgets'][split] if max_steps is None else max_steps
    cadence = record['execute_steps'] if execute_steps is None else execute_steps
    for value in (budget,cadence):
        if type(value) is not int or value < 1:
            raise ValueError('Action budget and execution cadence must be positive integers')
    return {'protocol':record['protocol'],'split':split,'max_steps':budget,
        'execute_steps':cadence,'budget_unit':record['budget_unit'],
        'execution_override':max_steps is not None or execute_steps is not None}


def execution_prefix(chunk, execute_steps, remaining):
    """Validate the action chunk and trim its last block to the remaining budget."""
    chunk = np.asarray(chunk)
    if type(execute_steps) is not int or type(remaining) is not int or min(execute_steps,remaining) < 1:
        raise ValueError('Invalid remaining action budget or execution cadence')
    if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
        raise ValueError('Expected a finite (horizon, 7) action chunk')
    if len(chunk) < min(execute_steps,remaining):
        raise ValueError('Policy horizon is shorter than the requested execution block')
    return chunk[:min(execute_steps,remaining)].copy()
