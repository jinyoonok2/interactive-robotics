"""Select and export an explicitly identified expert demonstration for a trial."""
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / 'datasets' / 'PartInstruct'


def decode(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def signature(chain):
    return json.dumps(chain, sort_keys=True)


def metadata_entries(value):
    if isinstance(value, dict):
        if 'ep_id' in value and 'chain_params' in value:
            yield value
        else:
            for child in value.values():
                yield from metadata_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from metadata_entries(child)


class GroundTruthReferences:
    def __init__(self, cache_dir, data_root=DEFAULT_DATA_ROOT):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root).resolve(strict=True)
        self.metadata = {}
        for filename in ['episodes_meta_train.json', 'episodes_meta_test.json']:
            path = self.data_root / filename
            for entry in metadata_entries(json.loads(path.read_text())):
                self.metadata[str(entry['ep_id'])] = entry
        self.indices = {}

    def index(self, obj):
        if obj in self.indices:
            return self.indices[obj]
        path = self.data_root / 'demos' / f'{obj}.hdf5'
        cache = self.cache_dir / f'{obj}_demo_index.json'
        fingerprints = [path, self.data_root/'episodes_meta_train.json', self.data_root/'episodes_meta_test.json']
        fingerprint = [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in fingerprints]
        if cache.is_file():
            saved = json.loads(cache.read_text())
            if saved.get('fingerprint') == fingerprint:
                self.indices[obj] = saved['demos']
                return saved['demos']
        demos, skipped = [], []
        with h5py.File(path, 'r') as handle:
            names=[name for name in handle['data'] if name.startswith('demo_') and name[5:].isdigit()]
            for name in sorted(names, key=lambda k: int(k[5:])):
                demo = handle['data'][name]
                if 'metadata' not in demo or 'ep_id' not in demo['metadata']:
                    skipped.append(name)
                    continue
                ep = decode(demo['metadata']['ep_id'][()])
                info = self.metadata.get(ep)
                if info is None or 'agentview_rgb' not in demo['obs']:
                    skipped.append(name)
                    continue
                demos.append({'demo_key': name, 'episode_id': ep, 'chain_params': info['chain_params'],
                              'object_id': decode(demo['metadata']['obj_id'][()]),
                              'object_pose': demo['metadata']['obj_pose'][()].tolist(),
                              'object_scale': float(demo['metadata']['obj_scale'][()])})
        cache.write_text(json.dumps({'fingerprint': fingerprint, 'demos': demos, 'skipped': skipped}, indent=2)+'\n')
        self.indices[obj] = demos
        return demos

    def select(self, trial):
        obj, ep = trial['obj_class'], str(trial['info']['ep_id'])
        chain = trial['info']['chain_params']
        candidates = [d for d in self.index(obj) if signature(d['chain_params']) == signature(chain)]
        if not candidates:
            raise ValueError(f'No expert demonstration matches the complete skill chain for {obj} episode {ep}; refusing to substitute another task')
        pose = np.r_[trial['info']['Object init orient'], trial['info']['Object init pose']]
        object_id = str(trial['info']['Object id'])
        scale = float(trial['info']['Object scale'])

        def rank(demo):
            q = np.asarray(demo['object_pose'][:4])
            # q and -q represent the same orientation.
            orientation = 1 - min(1., abs(float(np.dot(q, pose[:4]))))
            position = float(np.linalg.norm(np.asarray(demo['object_pose'][4:]) - pose[4:]))
            return (demo['episode_id'] != ep, demo['object_id'] != object_id,
                    abs(demo['object_scale']-scale), orientation, position, demo['demo_key'])

        selected = min(candidates, key=rank)
        return {**selected, 'evaluation_episode_id': ep,
                'match_type': 'exact_episode' if selected['episode_id'] == ep else 'task_matched_reference',
                'skill_chain_exact': True, 'same_object_instance': selected['object_id'] == object_id,
                'initial_position_difference_cm': 100*float(np.linalg.norm(np.asarray(selected['object_pose'][4:])-pose[4:])),
                'hdf5': str(self.data_root/'demos'/f'{obj}.hdf5')}

    def trial(self, evaluated_trial, fps=20.):
        reference = self.select(evaluated_trial)
        stem = f'{evaluated_trial["obj_class"]}_{reference["demo_key"]}_{fps:g}fps'
        video = self.cache_dir/f'{stem}.mp4'
        state_path = self.cache_dir/f'{stem}.json'
        source=Path(reference['hdf5'])
        fingerprint=[str(source),source.stat().st_size,source.stat().st_mtime_ns]
        state=json.loads(state_path.read_text()) if state_path.is_file() else {}
        if not video.is_file() or state.get('fingerprint')!=fingerprint:
            with h5py.File(reference['hdf5'], 'r') as handle:
                demo = handle['data'][reference['demo_key']]
                instructions = [decode(v) for v in demo['skill_instructions'][()]]
                frames = demo['obs']['agentview_rgb']
                masks = demo['obs'].get('agentview_part_mask')
                height, width = frames.shape[1:3]
                temporary = video.with_name(video.stem+'.tmp.mp4')
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot write expert reference: {temporary}')
                try:
                    for i in range(len(frames)):
                        frame = np.asarray(frames[i]).copy()
                        if masks is not None:
                            mask = np.asarray(masks[i]).squeeze() > 0
                            frame[mask] = (.65*frame[mask]+.35*np.array([70,225,90])).astype(np.uint8)
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
                temporary.replace(video)
                state = {'instructions': instructions, 'frames': len(frames), 'fps': fps, 'fingerprint':fingerprint}
                state_path.write_text(json.dumps(state, indent=2)+'\n')
        state = json.loads(state_path.read_text())
        capture=cv2.VideoCapture(str(video))
        valid=capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==state['frames']
        capture.release()
        if not valid:
            raise ValueError(f'Expert reference video has missing frames: {video}')
        return {**evaluated_trial, 'rollout_video': str(video), 'phase_events': [],
                'reference_kind': 'expert_demonstration', 'ground_truth_reference': reference,
                'reference_instructions': state['instructions'], 'success': None}
