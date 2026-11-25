"""Deterministic, skill-chain-stratified demonstration selection for development."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import h5py

from .targets import segments


def rank(seed, *values):
    return hashlib.sha256((str(seed)+':'+':'.join(map(str,values))).encode()).hexdigest()


def stratified_selection(candidates, *, per_object, validation_fraction, seed):
    """Round-robin chains within objects, then hold out entire demos per chain."""
    if per_object < 2 or not 0 < validation_fraction < 1:
        raise ValueError('Pilot needs at least two demos per object and a held-out fraction')
    by_object = defaultdict(lambda:defaultdict(list))
    for item in candidates:
        by_object[item['object']][tuple(item['skills'])].append(item)
    selected, summary = [], {}
    for obj, groups in sorted(by_object.items()):
        for signature,items in groups.items():
            items.sort(key=lambda row:rank(seed,obj,signature,row['demo']))
        signatures = sorted(groups,key=lambda key:rank(seed,obj,key))
        chosen = []
        while len(chosen)<per_object and any(groups.values()):
            for signature in signatures:
                if groups[signature] and len(chosen)<per_object:
                    chosen.append(groups[signature].pop())
        if len(chosen)!=per_object:
            raise ValueError('Insufficient demonstrations for pilot object '+obj)
        selected_groups = defaultdict(list)
        for item in chosen:selected_groups[tuple(item['skills'])].append(item)
        counts = Counter()
        for signature,items in sorted(selected_groups.items()):
            items.sort(key=lambda row:rank(seed,'validation',obj,row['demo']))
            n = max(1,round(len(items)*validation_fraction)) if len(items)>1 else 0
            n = min(n,len(items)-1)
            for i,item in enumerate(items):
                split = 'validation' if i<n else 'train'
                selected.append({**item,'split':split})
                counts[split]+=1
        if not counts['validation']:
            raise ValueError('No held-out chain examples for '+obj)
        summary[obj] = {'demonstrations':len(chosen),'by_split':dict(counts),
            'skill_chains':len(selected_groups)}
    return selected,summary


def prepare_pilot(paths, *, per_object=64, validation_fraction=.1, seed=42,
                  action_chunk=8, max_skill_slots=4):
    candidates, skipped = [], Counter()
    for path in paths:
        path = Path(path).resolve(strict=True)
        with h5py.File(path,'r') as handle:
            for key in sorted(handle['data']):
                demo = handle['data'][key]
                if 'actions' not in demo or len(demo['actions']) <= action_chunk:
                    skipped['missing_or_short_actions']+=1;continue
                if any(name not in demo['obs'] for name in ('agentview_pcd','agentview_part_pcd','tcp_pose')):
                    skipped['missing_observations']+=1;continue
                phases = segments(demo['skill_instructions'][:])
                if len(phases)>max_skill_slots or any(p['kind']=='unknown' for p in phases):
                    skipped['unsupported_skill_chain']+=1;continue
                candidates.append({'file':str(path),'object':path.stem,'demo':key,
                    'skills':[p['kind'] for p in phases],
                    'phase_starts':[p['start'] for p in phases],
                    'phase_ends':[p['end'] for p in phases],
                    'n_steps':len(demo['actions'])})
    selected,summary = stratified_selection(candidates,per_object=per_object,
        validation_fraction=validation_fraction,seed=seed)
    return {'schema_version':1,'scope':'Development subset selected from training HDF5 only; no benchmark test scenarios.',
        'selection':'Round-robin skill chains within each object; whole-demo stratified holdout.',
        'seed':seed,'requested_validation_fraction':validation_fraction,
        'per_object':per_object,'objects':summary,'skipped':dict(skipped),
        'demonstrations':selected}
