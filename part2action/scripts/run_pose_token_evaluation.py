"""Prepare, gate and report matched evaluations of completed pose-token policies.

Training weights stay unchanged. New studies pin their execution protocol;
historical plans/results retain their original settings. Smoke rollouts only
validate execution; the report uses full benchmark shards only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from diagnostics.evaluation_protocol import execution_settings, DEFAULT_PROTOCOL_PATH

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('disabled', 'provided')
LAUNCHER = ROOT/'part2action/slurm/evaluate_pose_token_diffusion.sbatch'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    training, oracle = args.training.resolve(strict=True), args.oracle.resolve(strict=True)
    reference_plan = read(oracle/'plan.json')
    if reference_plan['config']['evaluation']['seeds'] != [42, 43, 44]:
        raise ValueError('Expected the existing three-seed reference population')
    training_plan = read(training/'plan.json')
    # The execution-protocol migration intentionally changes the evaluator and
    # records the original helper's relocation in its preservation manifest.
    # Trained model/data/optimizer sources are never exempted from verification.
    revised_runtime = {str((ROOT/name).resolve()) for name in (
        'part2action/scripts/rollout_partgym.py',
        'part2action/models/pose_token_diffusion/evaluation.py',
        'part2action/models/part_grounded_diffusion/preserved_sources.json')}
    runtime_revisions = {}
    for name, expected in training_plan['source_sha256'].items():
        if sha(name) != expected:
            if name not in revised_runtime:
                raise ValueError('Training source changed: '+name)
            runtime_revisions[name] = {'training_snapshot_sha256':expected,'evaluation_sha256':sha(name)}
    matched, checkpoints, configs, initials, histories = {}, {}, {}, {}, {}
    import torch
    for arm in ARMS:
        folder = training/'models'/arm
        completion = read(folder/'completion.json')
        if completion['status'] != 'completed' or completion['smoke']:
            raise ValueError('Full training did not complete: '+arm)
        checkpoint = folder/f"epoch_{completion['epochs']:03d}.pt"
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('architecture') != 'pose_token_diffusion_v1' or payload['config']['pose_source'] != arm:
            raise ValueError('Unexpected checkpoint identity: '+arm)
        config = read(folder/'config_resolved.json')
        config['model'].pop('pose_source')
        configs[arm] = config
        initials[arm] = read(folder/'initialization.json')
        histories[arm] = read(folder/'history.json')
        matched[arm] = {'completion': completion, 'data_files': read(folder/'data_files.json'),
            'initial_state': initials[arm]['initial_trainable_state_sha256'],
            'source_checkpoint': initials[arm]['checkpoint_sha256'],
            'batch_orders': [row['batch_order_sha256'] for row in histories[arm]],
            'updates_per_epoch': [row['optimizer_steps'] for row in histories[arm]]}
        checkpoints[arm] = {'path': str(checkpoint), 'sha256': sha(checkpoint)}
        del payload
    if configs['disabled'] != configs['provided'] or matched['disabled'] != matched['provided']:
        raise ValueError('The two training arms are not matched')
    from diagnostics.evaluation_scenarios import load_scenarios
    data_root = Path(reference_plan['config']['data_root']).resolve(strict=True)
    scenarios = load_scenarios(data_root/'episodes_meta_test.json', 'test1',
                              ['bottle', 'mug', 'pliers', 'scissors'])
    if scenarios != reference_plan['scenarios']:
        raise ValueError('Reference scenarios differ from the held-out metadata')
    trials = reference_plan['trials']
    if len(scenarios) != 149 or len(trials) != 447:
        raise ValueError('Expected 149 scenarios and 447 scenario/seed trials')
    catalog, missing = {}, []
    for index, trial in enumerate(trials):
        path = oracle/'oracle_references'/f'{index:03d}.json'
        available = path.is_file() and read(path)['status'] == 'completed'
        if path.is_file():
            catalog[str(path)] = sha(path)
        if trial['needs_grasp_pose'] and not available:
            missing.append(index)
    sources = {name:sha(name) for name in training_plan['source_sha256']}
    paths = [Path(__file__), LAUNCHER,
        ROOT/'part2action/scripts/evaluate_pose_token_diffusion.py',
        ROOT/'part2action/scripts/generate_logged_references.py',
        ROOT/'part2action/scripts/run_interaction_tolerance_pilot.py',
        ROOT/'part2action/diagnostics/evaluation_scenarios.py',
        ROOT/'part2action/diagnostics/execution_trace.py',
        ROOT/'part2action/diagnostics/evaluation_protocol.py', DEFAULT_PROTOCOL_PATH,
        ROOT/'part2action/docs/archive/2026-11-16_evaluation_protocol/rollout_partgym_120_1.py',
        ROOT/'PartInstruct/PartInstruct/baselines/evaluation/config/env/env_config.yaml']
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym').rglob('*.py'))
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym/config').rglob('*.yaml'))
    for path in paths:
        sources[str(path.resolve())] = sha(path)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.out_dir/'source_snapshot'
    snapshot.mkdir()
    for index, name in enumerate(sorted(sources)):
        shutil.copy2(name, snapshot/f'{index:03d}_{Path(name).name}')
    write(args.out_dir/'plan.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Matched GT-input versus pose-off comparison after shared legacy initialization; not from-scratch or predicted-pose deployment.',
        'training': str(training), 'oracle': str(oracle), 'data_root': str(data_root),
        'seeds': [42, 43, 44], 'shards': args.shards, **execution_settings('test1'),
        'execution_settings':execution_settings('test1'),'runtime_revisions':runtime_revisions,
        'training_source_sha256':training_plan['source_sha256'],
        'scenarios': scenarios, 'trials': trials, 'reference_gaps': missing,
        'checkpoints': checkpoints, 'matched_training': matched['disabled'],
        'source_sha256': sources, 'reference_sha256': catalog,
        'metadata_sha256': {str(oracle/'plan.json'): sha(oracle/'plan.json'),
            str(data_root/'episodes_meta_test.json'): sha(data_root/'episodes_meta_test.json')}})
    print(json.dumps({'study': str(args.out_dir), 'trials_per_model': len(trials),
        'required_reference_gaps': missing, 'matched_training': True}), flush=True)


def verify(plan):
    for group in ('source_sha256', 'reference_sha256', 'metadata_sha256'):
        for path, expected in plan[group].items():
            if sha(path) != expected:
                raise ValueError('Pinned evaluation input changed: '+path)
    for checkpoint in plan['checkpoints'].values():
        if sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Trained checkpoint changed')


def require_smoke(root, plan):
    for arm in ARMS:
        receipt = read(root/'smoke'/arm/'runtime_passed.json')
        if receipt['status'] != 'passed' or receipt['checkpoint_sha256'] != plan['checkpoints'][arm]['sha256']:
            raise ValueError('Both exact trained checkpoints must pass native rollout smoke first')
        if 'execution_settings' in plan and any(receipt.get(k) != plan[k] for k in ('max_steps','execute_steps')):
            raise ValueError('Smoke receipt uses a different action budget or execution cadence')


def run(args, plan):
    smoke = args.mode == 'smoke'
    arm_index, shard = (args.index, 0) if smoke else divmod(args.index, plan['shards'])
    if not 0 <= arm_index < len(ARMS):
        raise ValueError('Invalid evaluation array index')
    if not smoke:
        require_smoke(args.out_dir, plan)
    arm = ARMS[arm_index]
    output = args.out_dir/'smoke'/arm if smoke else args.out_dir/'evaluations'/arm/f'shard_{shard:02d}'
    command = [sys.executable, str(ROOT/'part2action/scripts/evaluate_pose_token_diffusion.py'),
        '--ckpt', plan['checkpoints'][arm]['path'], '--out-dir', str(output),
        '--data-root', plan['data_root'], '--oracle-run-dir', plan['oracle'],
        '--seeds', *map(str, plan['seeds']), '--shards', str(1 if smoke else plan['shards']),
        '--shard-index', str(shard),'--max-steps',str(plan['max_steps']),
        '--execute-steps',str(plan.get('execute_steps',1))]
    if smoke:
        command += ['--limit', '1']
    subprocess.run(command, cwd=ROOT, check=True)
    if smoke:
        row = read(output/'000.json')
        if row['status'] != 'completed' or row['num_actions'] < 1 or not row['actions']:
            raise ValueError('Smoke must execute actual policy actions')
        if any(len(action) != 7 or not all(math.isfinite(x) for x in action) for action in row['actions']):
            raise ValueError('Smoke actions must be finite seven-dimensional commands')
        write(output/'runtime_passed.json', {'status': 'passed',
            'checkpoint_sha256': plan['checkpoints'][arm]['sha256'],
            'num_actions': row['num_actions'], 'max_steps': plan['max_steps'],
            'execute_steps':plan.get('execute_steps',1),
            'scope': 'Runtime validation only; excluded from benchmark statistics.'})
    else:
        write(output/'completed.json', {'status': 'completed', 'arm': arm, 'shard': shard,
            'checkpoint_sha256': plan['checkpoints'][arm]['sha256']})


def metric(rows, paired, field):
    def value(row):
        return bool(row['success']) if field == 'task' else bool(row['grasp'][field])
    counts = {arm: sum(value(rows[arm][i]) for i in paired) for arm in ARMS}
    return {'trials': len(paired), 'successes': counts,
        'rates': {arm: counts[arm]/len(paired) if paired else None for arm in ARMS},
        'difference_percentage_points': 100*(counts['provided']-counts['disabled'])/len(paired) if paired else None,
        'provided_only_success': sum(value(rows['provided'][i]) and not value(rows['disabled'][i]) for i in paired),
        'disabled_only_success': sum(value(rows['disabled'][i]) and not value(rows['provided'][i]) for i in paired)}


def aggregate(plan, rows):
    planned = set(range(len(plan['trials'])))
    for arm in ARMS:
        for index, row in rows[arm].items():
            if index not in planned or any(row.get(k) != v for k, v in plan['trials'][index].items()):
                raise ValueError('Result differs from the pinned scenario/seed trial')
            if row['status'] == 'completed' and 'execution_settings' in plan:
                if any(row.get(key) != plan[key] for key in ('max_steps','execute_steps')) or row['num_actions'] > plan['max_steps']:
                    raise ValueError('Result execution settings differ from the study protocol')
    paired = sorted(i for i in planned if all(rows[arm].get(i, {}).get('status') == 'completed' for arm in ARMS))
    grasp = [i for i in paired if plan['trials'][i]['needs_grasp_pose']]
    coverage = {arm: dict(Counter(rows[arm].get(i, {}).get('status', 'not_recorded') for i in planned)) for arm in ARMS}
    return {'scope': plan['scope'], 'execution_settings':plan.get('execution_settings',{'max_steps':120,'execute_steps':1,'historical':True}),
        'planned_per_model': len(planned), 'coverage': coverage,
        'expected_reference_gaps': plan['reference_gaps'], 'paired_trial_indices': paired,
        'task': metric(rows, paired, 'task'), 'grasp_acquired': metric(rows, grasp, 'grasp_acquired'),
        'initial_phase_end_grasp': metric(rows, grasp, 'initial_phase_last_grasp'),
        'per_seed_task': {str(seed): metric(rows, [i for i in paired if plan['trials'][i]['seed'] == seed], 'task')
                          for seed in plan['seeds']},
        'per_object_task': {obj: metric(rows, [i for i in paired if plan['scenarios'][plan['trials'][i]['scenario_index']]['obj_class'] == obj], 'task')
                            for obj in sorted({s['obj_class'] for s in plan['scenarios']})},
        'limitations': ['One training seed; evaluation seeds do not measure training-seed variability.',
            'Both models have auxiliary pose supervision; this isolates explicit GT pose input.',
            'Native grasp scoring permits either-finger contact and does not establish sustained retention.',
            'Missing references and runtime errors remain coverage gaps, not physical policy failures.',
            'No claim about from-scratch training or deployment with predicted poses.']}


def report(args, plan):
    rows, missing_shards = {arm: {} for arm in ARMS}, []
    for arm in ARMS:
        for shard in range(plan['shards']):
            folder = args.out_dir/'evaluations'/arm/f'shard_{shard:02d}'
            if not (folder/'completed.json').is_file():
                missing_shards.append([arm, shard])
            for path in folder.glob('[0-9][0-9][0-9].json'):
                row = read(path)
                index = row['trial_index']
                if index in rows[arm] or index % plan['shards'] != shard:
                    raise ValueError('Duplicate or incorrectly sharded trial')
                if row['status'] == 'completed' and row['checkpoint_sha256'] != plan['checkpoints'][arm]['sha256']:
                    raise ValueError('Result checkpoint differs from the pinned checkpoint')
                rows[arm][index] = row
    result = aggregate(plan, rows)
    result['missing_shards'] = missing_shards
    result['status'] = 'complete' if not missing_shards and all(
        row['status'] == 'completed' or (arm == 'provided' and index in plan['reference_gaps']
            and row['status'] == 'blocked_missing_oracle')
        for arm in ARMS for index, row in rows[arm].items()) and all(
            len(rows[arm]) == len(plan['trials']) for arm in ARMS) else 'incomplete'
    write(args.out_dir/'summary.json', result)
    lines = ['# Pose-token diffusion: matched GT-input evaluation', '',
        f"Status: **{result['status']}**. {plan['scope']}", '',
        '| Metric on identical completed trials | Pose off | GT pose | Difference |',
        '| --- | --- | --- | --- |']
    for label, key in [('Full task', 'task'), ('Grasp acquired', 'grasp_acquired'),
                       ('Initial phase-end grasp', 'initial_phase_end_grasp')]:
        item = result[key]
        cells = [f"{item['successes'][arm]}/{item['trials']} ({100*item['rates'][arm]:.1f}%)"
                 if item['trials'] else 'No completed pairs' for arm in ARMS]
        delta = f"{item['difference_percentage_points']:+.1f} pp" if item['trials'] else 'Pending'
        lines.append(f'| {label} | {cells[0]} | {cells[1]} | {delta} |')
    lines += ['', f"Planned: {len(plan['trials'])} trials per model. Required GT reference gaps: {plan['reference_gaps']}.",
              '', 'Coverage: `'+json.dumps(result['coverage'], sort_keys=True)+'`.', '', *result['limitations'], '']
    with (args.out_dir/'README.md').open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('status', 'coverage', 'task', 'initial_phase_end_grasp')}), flush=True)
    if result['status'] != 'complete':
        raise RuntimeError('Partial report written; benchmark coverage is incomplete')


def submit(args, plan):
    receipt = args.out_dir/'submissions.json'
    if receipt.exists():
        raise FileExistsError('Study already has submission receipts; do not submit duplicates')
    jobs = []
    def enqueue(role, options):
        command = ['sbatch', '--parsable', '--partition=gpu', '--no-requeue',
            '--job-name=p2a-token-'+role, '--output='+str(args.out_dir/(role+'_%A_%a.out')),
            '--error='+str(args.out_dir/(role+'_%A_%a.err')),
            '--export=ALL,P2A_TOKEN_STUDY='+str(args.out_dir)+',MODE='+role, *options, str(LAUNCHER)]
        job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0]
        jobs.append({'role': role, 'job_id': job, 'command': command})
        receipt.write_text(json.dumps(jobs, indent=2)+'\n')
        return job
    gpu = ['--gres=gpu:1', '--cpus-per-task=4', '--mem=16G',
           '--constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4000|a16|a4500|rtx_4000_ada']
    smoke = enqueue('smoke', [*gpu, '--time=00:30:00', '--array=0-1%2'])
    full = enqueue('evaluate', [*gpu, '--time=12:00:00',
        f"--array=0-{2*plan['shards']-1}%4", '--dependency=afterok:'+smoke])
    enqueue('report', ['--cpus-per-task=2', '--mem=4G', '--time=00:10:00', '--dependency=afterany:'+full])
    print(json.dumps(jobs, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'submit', 'smoke', 'evaluate', 'report'), required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--oracle', type=Path)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.mode == 'prepare':
        if args.training is None or args.oracle is None or args.shards < 1:
            parser.error('Preparation requires training, oracle and a positive shard count')
        prepare(args)
        return
    plan = read(args.out_dir/'plan.json')
    verify(plan)
    if args.mode in ('smoke', 'evaluate'):
        run(args, plan)
    else:
        {'submit': submit, 'report': report}[args.mode](args, plan)


if __name__ == '__main__':
    main()
