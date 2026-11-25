"""Completion-only observations: no fabricated future actions or oracle inputs.

An annotated transition supplies evidence that the demonstrator advanced. Full
metadata-chain endpoints are explicitly weak annotation proxies, not physical
success labels. Incomplete/mismatched final chains are masked, never positive.
"""
from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .targets import segments
from .dataset import observation_from_batch


def completion_records(phases, n_steps, *, complete_chain, post_transition_steps=8,
                       allow_endpoint_proxy=True):
    """Previous-skill queries remain positive briefly after its observed transition.

    Pre-boundary frames stay negative: there is no anticipatory 'finish soon'
    label and no replacement of the deployed threshold.
    """
    records, masked = [], []
    for phase, item in enumerate(phases):
        final = phase == len(phases)-1
        valid = not final or (complete_chain and allow_endpoint_proxy)
        source = 'annotated_transition' if not final else 'complete_chain_endpoint_proxy'
        if not valid:
            masked.append({'phase': phase, 'skill': item['kind'], 'reason': 'unverified_or_incomplete_final_chain'})
            continue
        stop = min(n_steps-1, item['end']+(0 if final else post_transition_steps))
        for t in range(item['start'], stop+1):
            records.append({'t': t, 'query_phase': phase, 'skill': item['kind'],
                'target': int(t >= item['end']), 'source': source,
                'boundary_t': item['end'], 'physical_success_verified': False})
    return records, masked


class CompletionDataset(Dataset):
    def __init__(self, manifest, *, split, n_obs_steps=2, post_transition_steps=8,
                 allow_endpoint_proxy=True):
        self.manifest = Path(manifest)
        selection = json.loads(self.manifest.read_text())['demonstrations']
        self.n_obs_steps = n_obs_steps
        self._files = {}
        self._samples, self.masked = [], []
        roots = {Path(r['file']).parents[1] for r in selection}
        if len(roots) != 1:
            raise ValueError('Completion metadata must have one explicit data root')
        self.metadata_path = roots.pop()/'episodes_meta_train.json'
        metadata = json.loads(self.metadata_path.read_text())
        lookups = {obj: {str(e['ep_id']): e for rows in value['train'].values() for e in rows}
                   for obj, value in metadata.items()}
        for demo in selection:
            if demo['split'] != split:
                continue
            with h5py.File(demo['file'], 'r') as handle:
                g = handle['data'][demo['demo']]
                phases = segments(g['skill_instructions'][:])
                eid = self.decode(g['metadata/ep_id'][()])
                episode = lookups[demo['object']].get(eid)
                norm = lambda s: ' '.join(str(s).strip().lower().split())
                complete = bool(episode and [norm(p['instruction']) for p in phases]
                                == [norm(s) for s in episode['skill_instructions']])
                recorded_steps = len(g['actions'])
                if any(len(g['obs'][key]) != recorded_steps for key in
                       ('agentview_rgb', 'agentview_pcd', 'tcp_pose', 'gripper_state')):
                    raise ValueError('Completion observations are misaligned')
                if phases[-1]['end'] != recorded_steps-1:
                    raise ValueError('Completion annotations do not cover the recording')
                task = self.decode(g['task_instructions'][0])
            records, masked = completion_records(phases, recorded_steps, complete_chain=complete,
                post_transition_steps=post_transition_steps, allow_endpoint_proxy=allow_endpoint_proxy)
            base = {key: demo[key] for key in ('file', 'object', 'demo')}
            self._samples.extend({**base, **r, 'task_instruction': task} for r in records)
            self.masked.extend({**base, **r} for r in masked)
        if not self._samples:
            raise ValueError('No eligible completion observations')
        counts = Counter((r['skill'], r['target']) for r in self._samples)
        self.audit = {'split': split, 'observations': len(self),
            'by_skill': {k: {'negative': counts[k, 0], 'positive': counts[k, 1]}
                         for k in sorted({r['skill'] for r in self._samples})},
            'positive_sources': dict(Counter(r['source'] for r in self._samples if r['target'])),
            'masked_final_phases': self.masked,
            'limits': 'Labels denote annotated transitions or complete-chain endpoint proxies; no physical-success flags exist in these released HDF5s.'}

    @staticmethod
    def decode(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        row = self._samples[index]
        if row['file'] not in self._files:
            self._files[row['file']] = h5py.File(row['file'], 'r')
        obs = self._files[row['file']]['data'][row['demo']]['obs']
        t = row['t']
        frames = [max(0, t-i) for i in reversed(range(self.n_obs_steps))]
        rgb = np.stack([np.asarray(obs['agentview_rgb'][i], dtype=np.uint8) for i in frames])
        return {'rgb': torch.from_numpy(rgb).permute(0, 3, 1, 2).float()/255.,
            'agentview_pcd': torch.from_numpy(np.asarray(obs['agentview_pcd'][t], dtype=np.float32)),
            'tcp_pose': torch.from_numpy(np.asarray(obs['tcp_pose'][t], dtype=np.float32)),
            'gripper_state': torch.from_numpy(np.asarray(obs['gripper_state'][t], dtype=np.float32)).reshape(1),
            'task_instruction': row['task_instruction'],
            'completion_query_phase': torch.tensor(row['query_phase'], dtype=torch.long),
            'phase_termination': torch.tensor(row['target'], dtype=torch.float32),
            'meta': copy.deepcopy(row)}

    def close(self):
        for handle in self._files.values():
            handle.close()
        self._files.clear()


class BalancedCompletionSampler(Sampler):
    """Every four-observation batch has two positive and two negative queries.

    Uniform skill choice within each class prevents long grasp phases from
    eliminating release/rotation updates. Main action sampling is unchanged.
    """
    def __init__(self, dataset, batches, seed):
        self.batches, self.seed, self.epoch = batches, seed, 0
        self.groups = {0: defaultdict(list), 1: defaultdict(list)}
        for i, r in enumerate(dataset._samples):
            self.groups[r['target']][r['skill']].append(i)
        if not all(self.groups.values()):
            raise ValueError('Completion balancing needs both labels')

    def __len__(self):
        return self.batches*4

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        self.epoch += 1
        for _ in range(self.batches):
            for label in (0, 1, 0, 1):
                skills = sorted(self.groups[label])
                skill = skills[int(rng.integers(len(skills)))]
                pool = self.groups[label][skill]
                yield pool[int(rng.integers(len(pool)))]


def completion_loss(hierarchy, batch):
    logits = hierarchy['completion_logits_by_phase'].gather(
        1, batch['completion_query_phase'][:, None]).squeeze(1)
    return F.binary_cross_entropy_with_logits(logits.float(), batch['phase_termination'].float())


def validate_completion(model, loader, directory):
    from .metrics import classification
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    rows = []
    device = next(model.parameters()).device
    previous_mode = model.training
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            encoded = model.blocks['encoder'](observation_from_batch(batch))
            probability = encoded.hierarchy['completion_logits_by_phase'].gather(
                1, batch['completion_query_phase'][:, None]).squeeze(1).float().sigmoid().cpu().tolist()
            rows.extend({**meta, 'probability': p} for meta, p in zip(batch['meta'], probability))
    def score(subset):
        result = classification([r['probability'] >= .5 for r in subset], [r['target'] for r in subset])
        result['false_positive_rate'] = result['fp']/(result['fp']+result['tn']) if result['fp']+result['tn'] else None
        return result
    phases = defaultdict(list)
    for row in rows:
        phases[(row.get('object',''),row.get('demo',''),row.get('query_phase',0))].append(row)
    events = []
    for (obj,demo,phase), samples in sorted(phases.items()):
        positive = [r.get('t',0) for r in samples if r['target'] and r['probability']>=.5]
        early = [r.get('t',0) for r in samples if not r['target'] and r['probability']>=.5]
        events.append({'object':obj,'demo':demo,'query_phase':phase,'skill':samples[0]['skill'],
            'source':samples[0]['source'],'boundary_t':samples[0].get('boundary_t'),
            'positive_coverage':any(r['target'] for r in samples),
            'first_positive_trigger_t':min(positive) if positive else None,
            'first_premature_trigger_t':min(early) if early else None,
            'missed':bool(any(r['target'] for r in samples) and not positive), 'premature':bool(early)})
    result = {'scope': 'Held-out completion queries, including late observations and previous-skill queries after transitions. Endpoint proxies are not physical-success labels.',
        'threshold': .5, 'overall': score(rows),
        'by_skill': {k: score([r for r in rows if r['skill'] == k]) for k in sorted({r['skill'] for r in rows})},
        'by_source': {k: score([r for r in rows if r['source'] == k]) for k in sorted({r['source'] for r in rows})},
        'events': {'eligible_phases':len(events),'phases_with_positive_labels':sum(r['positive_coverage'] for r in events),
            'missed_phases':sum(r['missed'] for r in events),
            'phases_with_premature_trigger':sum(r['premature'] for r in events)}}
    (directory/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (directory/'samples.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    with (directory/'phase_events.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(events[0]))
        writer.writeheader();writer.writerows(events)
    text = ['# Completion validation', '', result['scope'], '',
        '| Skill | Positive labels | Recall | False positive rate |', '| --- | ---: | ---: | ---: |']
    for k, v in result['by_skill'].items():
        text.append(f"| {k} | {v['positive_targets']} | {v['recall']} | {v['false_positive_rate']} |")
    text += ['', '[Per-phase timing and premature triggers](phase_events.csv)', '', str(result['events'])]
    (directory/'README.md').write_text('\n'.join(text)+'\n')
    model.train(previous_mode)
    return result
