"""Validate the current config inventory without loading models or simulator."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from validate_configs import validate

ROOT=Path(__file__).resolve().parents[1]


def audit(check_files=False):
    directory=ROOT/'configs'
    index=json.loads((directory/'index.json').read_text())
    active={entry['role']:directory/entry['path'] for entry in index['active']}
    if len(active)!=3:
        raise ValueError('Expected the baseline and two current visual-pose configurations')
    expected_regular={p.resolve() for p in active.values()} | {(directory/'index.json').resolve()}
    regular={p.resolve() for p in directory.rglob('*')
             if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.yaml','.yml'}}
    if regular!=expected_regular:
        raise ValueError(f'Unindexed or missing active configs: {regular ^ expected_regular}')
    for role,path in active.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing regular active config: {role}')
        if path.suffix=='.yaml':
            errors=validate(path,check_files=check_files)
            if errors:
                raise ValueError(f'{path}: {errors}')
    study=json.loads(active['matched_visual_gt_pose_study'].read_text())
    if (ROOT.parent/study['baseline_config']).resolve()!=active['original_visual_baseline'].resolve():
        raise ValueError('GT study must use the indexed original visual baseline')
    predicted=yaml.safe_load(active['shared_grounding_predicted_pose'].read_text())
    baseline=yaml.safe_load(active['original_visual_baseline'].read_text())
    for config in [baseline,predicted]:
        if config['model']['diffusion_denoiser_type']!='transformer' or not config['data']['use_pcd']:
            raise ValueError('Current recipes retain the visual/scene Transformer policy')
    if not predicted['predicted_pose']['pose_conditioning_enabled']:
        raise ValueError('Predicted-pose recipe must enable its pose feature path')
    if predicted['model']['contact_conditioning_enabled']:
        raise ValueError('Do not add an independent direct contact-to-action path')
    if check_files and not Path(study['init_checkpoint']).is_file():
        raise ValueError('Missing GT-study initialization checkpoint')
    for alias in index['compatibility_aliases']:
        path=directory/alias['path'];target=directory/alias['target']
        if not path.is_symlink() or path.resolve()!=target.resolve() or not target.is_file():
            raise ValueError('Broken frozen-tooling config alias: '+str(path))
    history=directory/index['historical_configs']
    manifest=json.loads((history/'manifest.json').read_text())
    for entry in manifest['archived_configs']:
        if hashlib.sha256((history/entry['name']).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Archived config changed: '+entry['name'])
    return {'status':'passed','current_regular_configs':len(active),
            'compatibility_aliases':len(index['compatibility_aliases']),
            'archived_configs_moved':sum(e['action']=='moved out of active configs' for e in manifest['archived_configs']),
            'active':{role:str(path.relative_to(ROOT)) for role,path in active.items()},
            'checked_installed_files':check_files}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-files',action='store_true')
    args=parser.parse_args()
    print(json.dumps(audit(args.check_files),indent=2))
