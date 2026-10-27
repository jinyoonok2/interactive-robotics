"""Enumerate held-out scenarios and reset PartGym to an explicit scenario."""
import copy
import json
from types import MethodType
import math


DEFAULT_EVALUATION_SEEDS = (42, 43, 44)


def rollout_plan(scenarios, seeds=DEFAULT_EVALUATION_SEEDS):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Evaluation seeds must be nonempty and unique')
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Evaluation seeds must be integers in [0, 2**32)')
    return [(scenario, index, seed) for scenario in scenarios
            for index, seed in enumerate(seeds)]


def success_statistics(results):
    by_seed = {}
    seen = set()
    for row in results:
        episode_id = row.get('requested_episode_id') or row.get('info', {}).get('ep_id')
        scenario = (row['obj_class'], str(row['task_type']), str(episode_id))
        seed = int(row['seed'])
        if (scenario, seed) in seen:
            raise ValueError(f'Duplicate scenario/seed result: {scenario}, {seed}')
        seen.add((scenario, seed))
        by_seed.setdefault(seed, []).append(row)
    per_seed = {str(seed): {'trials': len(rows),
                           'successes': sum(bool(row['success']) for row in rows),
                           'success_rate': sum(bool(row['success']) for row in rows) / len(rows)}
                for seed, rows in sorted(by_seed.items())}
    rates = [row['success_rate'] for row in per_seed.values()]
    mean = sum(rates) / len(rates) if rates else 0.
    std = math.sqrt(sum((rate - mean)**2 for rate in rates) / (len(rates)-1)) if len(rates)>1 else None
    return {'per_seed': per_seed, 'mean_seed_success_rate': mean,
            'std_seed_success_rate': std, 'std_ddof': 1,
            'uncertainty_scope': 'evaluation randomness; one training seed per model'}


def load_scenarios(meta_path, split, obj_classes=None, task_types=None, limit=None):
    if limit is not None and limit < 1:
        raise ValueError('Scenario limit must be positive')
    meta = json.loads(meta_path.read_text())
    scenarios, seen = [], set()
    for obj in sorted(meta):
        if obj_classes and obj not in obj_classes:
            continue
        for task, episodes in sorted(meta[obj].get(split, {}).items()):
            if task_types and task not in task_types:
                continue
            for episode in sorted(episodes, key=lambda row: str(row['ep_id'])):
                key = (obj, task, str(episode['ep_id']))
                if key in seen:
                    raise ValueError(f'Duplicate evaluation scenario: {key}')
                seen.add(key)
                scenarios.append({'obj_class': obj, 'task_type': task,
                                  'episode': copy.deepcopy(episode)})
    if not scenarios:
        raise ValueError(f'No scenarios matched split {split}')
    return scenarios if limit is None else scenarios[:limit]


def pin_scenario(env, episode, seed):
    """Preserve PartGym's reset semantics while bypassing random task sampling.

    Like upstream sample_chain_params, initialize the object at z=0.2 m.
    Metadata is used only to initialize the scene and score the task.
    """
    episode = copy.deepcopy(episode)

    def select(self):
        self.obj_id = episode['obj_id']
        self.ep_id = episode['ep_id']
        self.chain_params = copy.deepcopy(episode['chain_params'])
        self.task_instruction = episode['task_instruction']
        self.skill_instructions = list(episode['skill_instructions'])
        self.obj_init_position = list(episode['obj_pose'][4:])
        self.obj_init_position[-1] = 0.2
        self.obj_init_orientation = list(episode['obj_pose'][:4])
        self.obj_scale = episode['obj_scale']

    env.sample_chain_params = MethodType(select, env)
    env.seed(seed)
