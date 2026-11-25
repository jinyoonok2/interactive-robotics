"""Keep generated Part2Action artifacts on the project allocation.

Repository results paths are browsing aliases. Resolve fresh legacy-style paths
onto project storage before creating them; reject arbitrary home destinations.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess

REPOSITORY = Path(__file__).resolve().parents[2]


def project_results_root(storage_root=None):
    allocation = Path(storage_root or os.environ.get('PART2ACTION_STORAGE_ROOT', '/p/part2action')).expanduser().resolve()
    if allocation.is_relative_to(Path.home().resolve()):
        raise ValueError('Part2Action storage must be outside the home allocation')
    results = (allocation/'results').resolve()
    if not results.is_relative_to(allocation):
        raise ValueError('The results directory must remain on the project allocation')
    return results


def require_project_result_dir(path, *, storage_root=None):
    root = project_results_root(storage_root).resolve()
    target = Path(path).expanduser().resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Generated outputs must resolve under {root}, not {target}')
    return target


def configure_cache_environment(storage_root=None):
    """Set project-local cache paths before importing Hugging Face backbones."""
    allocation = project_results_root(storage_root).parent
    cache = (allocation/'caches').resolve()
    if not cache.is_relative_to(allocation):
        raise ValueError('Cache directories must remain on the project allocation')
    values = {'HF_HOME': cache/'huggingface', 'HF_HUB_CACHE': cache/'huggingface/hub',
        'HUGGINGFACE_HUB_CACHE': cache/'huggingface/hub', 'TORCH_HOME': cache/'torch',
        'XDG_CACHE_HOME': cache/'xdg', 'MPLCONFIGDIR': cache/'matplotlib',
        'CUDA_CACHE_PATH': cache/'cuda', 'TRITON_CACHE_DIR': cache/'triton',
        'TORCHINDUCTOR_CACHE_DIR': cache/'torchinductor', 'NUMBA_CACHE_DIR': cache/'numba'}
    for path in set(values.values()):
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
    values.update(DINOV2_HUB_DIR=cache/'torch/hub/facebookresearch_dinov2_main',
                  DINOV2_CKPT=cache/'torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth')
    os.environ.update({key: str(value) for key, value in values.items()})
    os.environ.pop('TRANSFORMERS_CACHE', None)
    return values


def result_dir(path, *, storage_root=None, repository_root=None):
    """Canonical physical output path, including fresh repository browsing paths."""
    root = project_results_root(storage_root).resolve()
    repository = Path(repository_root or REPOSITORY).resolve()
    requested = Path(os.path.abspath(Path(path).expanduser()))
    resolved = requested.resolve()
    if resolved.is_relative_to(root):
        return resolved
    for alias_root in (repository/'results', repository/'part2action/results'):
        if requested.is_relative_to(alias_root):
            if requested.exists() or requested.is_symlink():
                raise ValueError('Existing output is outside project storage; migrate it before reuse: ' + str(requested))
            target = (root/requested.relative_to(alias_root)).resolve()
            return require_project_result_dir(target, storage_root=storage_root)
    return require_project_result_dir(resolved, storage_root=storage_root)


def register_result_dir(target, *, storage_root=None, repository_root=None):
    """Leave a small browsing link; never replace another run or real directory."""
    target = require_project_result_dir(target, storage_root=storage_root)
    if not target.is_dir():
        raise ValueError('Create the physical result directory before registering it')
    repository = Path(repository_root or REPOSITORY).resolve()
    alias = repository/'results'/target.relative_to(project_results_root(storage_root).resolve())
    if alias.resolve() == target:
        return alias
    if alias.exists() or alias.is_symlink():
        raise FileExistsError('Refusing to overwrite an existing browsing path: ' + str(alias))
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target, target_is_directory=True)
    return alias


def validate_slurm_logs(job_description, *, storage_root=None):
    fields = dict(re.findall(r'\b(StdOut|StdErr)=(\S+)', job_description))
    if set(fields) != {'StdOut', 'StdErr'}:
        raise ValueError('Cannot verify both Slurm log paths')
    for key, value in fields.items():
        if value != '/dev/null':
            target = Path(value).expanduser().resolve()
            logs = project_results_root(storage_root).parent/'logs'
            logs = logs.resolve()
            if not logs.is_relative_to(project_results_root(storage_root).parent):
                raise ValueError('Slurm logs must remain on the project allocation')
            if not target.is_relative_to(logs):
                require_project_result_dir(target, storage_root=storage_root)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, required=True)
    parser.add_argument('--check-slurm-logs', action='store_true')
    args = parser.parse_args()
    target = require_project_result_dir(args.check)
    if args.check_slurm_logs and os.environ.get('SLURM_JOB_ID'):
        description = subprocess.check_output(['scontrol', 'show', 'job', '--oneliner',
                                              os.environ['SLURM_JOB_ID']], text=True)
        validate_slurm_logs(description)
    print(target)


if __name__ == '__main__':
    main()
