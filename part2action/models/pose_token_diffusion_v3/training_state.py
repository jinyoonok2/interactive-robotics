"""Atomic epoch-boundary recovery, separate from inference checkpoints."""
import os
from pathlib import Path
import random
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

from . import ARCHITECTURE_ID
from .checkpoint import checkpoint_state


def archive_incomplete_epochs(directory, completed_epoch):
    """Keep files from interrupted later epochs, without overwriting results."""
    directory = Path(directory)
    artifacts = list(directory.glob('epoch_*.pt')) + list((directory/'validation').glob('epoch_*'))
    later = [path for path in artifacts if int(path.stem.split('_')[-1]) > completed_epoch]
    if not later:
        return
    target = directory/'interrupted_artifacts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for path in later:
        destination = target/path.relative_to(directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)


def capture_rng():
    numpy = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    numpy = state['numpy']
    random.setstate(state['python'])
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_torch_save(path, payload):
    """Replace latest only after a full fsynced save; retain its predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = backup_temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            backup_temporary = temporary.with_suffix('.previous.tmp')
            try:
                os.link(path, backup_temporary)
            except OSError:
                shutil.copy2(path, backup_temporary)
            os.replace(backup_temporary, path.with_name(path.stem+'_previous'+path.suffix))
        os.replace(temporary, path)
    finally:
        for name in (temporary, backup_temporary):
            if name is not None:
                name.unlink(missing_ok=True)


def save_training_state(model, optimizer, path, *, history, generators,
                        progress_sampler, identity, initialization):
    if not history or history[-1].get('epoch', 0) < 1:
        raise ValueError('Recovery snapshots require a completed training epoch')
    atomic_torch_save(path, {
        'format': 'part2action_v3_training_state', 'schema': 1,
        'architecture': ARCHITECTURE_ID, 'config': model.config.to_dict(),
        'model': checkpoint_state(model), 'optimizer': optimizer.state_dict(),
        'history': history, 'rng': capture_rng(),
        'generators': {name: generator.get_state() for name, generator in generators.items()},
        'progress_sampler_epoch': progress_sampler.epoch,
        'identity': identity, 'initialization': initialization,
        'resume_scope': 'Completed epoch boundary; current dataset observations/labels are deterministic. Incomplete epochs are replayed.'})


def load_training_state(model, optimizer, path, *, generators, progress_sampler, identity):
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if payload.get('format') != 'part2action_v3_training_state' or payload.get('schema') != 1 or payload.get('architecture') != ARCHITECTURE_ID:
        raise ValueError('Not a resumable v3 training snapshot; model-only checkpoints cannot resume')
    if payload.get('identity') != identity or payload.get('config') != model.config.to_dict():
        raise ValueError('Resume requires the same recipe, sources, manifest and model configuration')
    if payload.get('initialization', {}).get('checkpoint_loaded') is not False:
        raise ValueError('Resume must continue a policy originally initialized from scratch')
    expected, incoming = checkpoint_state(model), payload['model']
    if expected.keys() != incoming.keys() or any(
        not isinstance(incoming[name], torch.Tensor) or incoming[name].shape != value.shape or incoming[name].dtype != value.dtype
        for name, value in expected.items()):
        raise ValueError('Resume model tensors differ')
    if set(payload['generators']) != set(generators):
        raise ValueError('Resume data generators differ')
    if len(payload['rng']['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('Resume CUDA random-state device count differs')
    history = payload['history']
    if not history or history[-1].get('epoch', 0) < 1 or history[-1].get('optimizer_steps', 0) < 1:
        raise ValueError('Recovery snapshot has no completed epoch')
    model.load_state_dict(incoming, strict=False)
    optimizer.load_state_dict(payload['optimizer'])
    for name, generator in generators.items():
        generator.set_state(payload['generators'][name])
    progress_sampler.epoch = payload['progress_sampler_epoch']
    restore_rng(payload['rng'])
    return payload
