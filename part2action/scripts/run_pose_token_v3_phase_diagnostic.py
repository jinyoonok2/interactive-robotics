"""Same v3 checkpoint, supplied plan, and resets; intervene on progression only.

This is a privileged development diagnostic, never a deployable planner. It is
kept outside the model package so it cannot change the smoke-validated trainer.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from diagnostics.storage import result_dir, register_result_dir, configure_cache_environment

from models.pose_token_diffusion_v3.runtime import HierarchicalRolloutPolicy
from models.pose_token_diffusion_v3.evaluation import SuppliedPlanDiagnostic
from models.pose_token_diffusion_v3.planner import PlannerDecision

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('learned_progression', 'oracle_progression')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preserved_completion_query(env, oracle, phase):
    """Restore physics, clock, recording flag, parser caches, and random streams."""
    parser = getattr(env, 'parser', None)
    caches = {key: copy.deepcopy(getattr(parser, key)) for key in
              ('part_pcds', 'part_pcds_last_time', 'transform_last_time') if hasattr(parser, key)}
    clock = env.world.sim_time
    recording = env.world.recording
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        env.world.recording = False
        return oracle.is_skill_complete(env, phase, exhaustive=True, preserve_sim_state=True)
    finally:
        env.world.sim_time, env.world.recording = clock, recording
        for key, value in caches.items():
            setattr(parser, key, value)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class OracleDiagnosticPlanner(SuppliedPlanDiagnostic):
    source = 'supplied_plan_simulator_progression_diagnostic'

    def plan(self, request):
        if request.reason == 'oracle_skill_complete' and self.index < len(self.skills)-1:
            self.index += 1
        return PlannerDecision('execute', self.skills[self.index])


class OracleDiagnosticRuntime(HierarchicalRolloutPolicy):
    def __init__(self, policy, planner, env, oracle):
        super().__init__(policy, planner)
        self.env, self.oracle = env, oracle
        self.oracle_events = []

    def reset(self):
        super().reset()
        self.oracle_events = []

    def _review_reason(self):
        reason = super()._review_reason()
        if self.skill is None or self.planner.index >= len(self.planner.skills)-1 or self.skill_chunks < self.min_skill_chunks:
            return reason
        decision = preserved_completion_query(self.env, self.oracle, self.planner.index)
        self.oracle_events.append({'prediction': len(self.oracle_events), 'phase': self.planner.index,
                                   'monitor_probability': self.progress_probability, **asdict(decision)})
        if not decision.supported:
            raise ValueError('Unsupported oracle phase diagnostic: ' + decision.reason)
        return 'oracle_skill_complete' if decision.complete else reason


def checkpoint_for_study(training, epoch=None):
    """Default to the configured final epoch, never a hard-coded pilot epoch."""
    training = Path(training)
    configured = read(training/'train/config_resolved.json')['train']['epochs']
    if type(configured) is not int or configured < 1:
        raise ValueError('Training epoch budget must be a positive integer')
    epoch = configured if epoch is None else epoch
    if type(epoch) is not int or not 1 <= epoch <= configured:
        raise ValueError('Requested checkpoint epoch is outside the training budget')
    return training/'train'/f'epoch_{epoch:03d}.pt', epoch


def training_evidence(training, epoch, *, allow_interrupted=False):
    """An interrupted run needs an explicit, fully recorded completed epoch."""
    training = Path(training)
    completion_path = training/'train/completion.json'
    if completion_path.exists():
        completion = read(completion_path)
        if completion.get('status') != 'completed' or completion.get('smoke'):
            raise ValueError('Expected completed non-smoke training')
        status = 'completed'
    elif allow_interrupted:
        status_record = read(training/'run_status.json')
        if status_record.get('status') not in ('failed', 'cancelled'):
            raise ValueError('Interrupted-checkpoint evaluation requires a recorded stopped run')
        status = status_record['status']
    else:
        raise ValueError('Wait for training, or explicitly select a checkpoint from an interrupted run')
    history = read(training/'train/history.json')
    rows = [row for row in history if row.get('epoch') == epoch]
    if len(rows) != 1 or rows[0].get('optimizer_steps', 0) < 1:
        raise ValueError('Checkpoint epoch has no unique completed training-history entry')
    return {'training_status': status, 'interrupted_checkpoint_diagnostic': status != 'completed',
            'history_sha256': sha(training/'train/history.json'), 'completed_epoch': rows[0]}


def verify_training_sources(training, review_path=None):
    """Keep source locking; permit only explicitly reviewed operational changes."""
    saved = read(Path(training)/'train/source_hashes.json')
    review = read(review_path) if review_path else {'schema': 1, 'changes': []}
    if review.get('schema') != 1 or not isinstance(review.get('changes'), list):
        raise ValueError('Invalid source-change review')
    reviewed = {item['path']: item for item in review['changes']}
    if len(reviewed) != len(review['changes']):
        raise ValueError('Duplicate reviewed source')
    # These entrypoints may change output/cache routing. Policy blocks, weights,
    # data labels, runtime, and model configuration must still match exactly.
    operational = {str(ROOT/'part2action'/name) for name in (
        'models/pose_token_diffusion_v3/evaluation.py',
        'models/pose_token_diffusion_v3/training.py',
        'scripts/run_pose_token_diffusion_v3.py', 'scripts/train_pose_token_diffusion_v3.py',
        'slurm/pose_token_diffusion_v3.sbatch')}
    changes = []
    for name, wanted in saved.items():
        current = sha(name)
        if current == wanted:
            continue
        item = reviewed.pop(name, None)
        if name not in operational or not item or item.get('training_sha256') != wanted or item.get('evaluation_sha256') != current or not item.get('reason', '').strip():
            raise ValueError('Unreviewed training implementation change: ' + name)
        changes.append(item)
    if reviewed:
        raise ValueError('Source-change review contains stale or unrelated entries')
    return {'training_source_receipt_sha256': sha(Path(training)/'train/source_hashes.json'),
            'reviewed_operational_changes': changes}


def prepare(args):
    import torch
    from models.pose_token_diffusion_v3.training import source_paths
    training = args.training_study.resolve(strict=True)
    if args.allow_interrupted_training and args.checkpoint_epoch is None:
        raise ValueError('Interrupted-run evaluation requires --checkpoint-epoch')
    checkpoint, epoch = checkpoint_for_study(training, args.checkpoint_epoch)
    evidence = training_evidence(training, epoch, allow_interrupted=args.allow_interrupted_training)
    evidence.update(verify_training_sources(training, args.source_change_review))
    initial = read(training/'train/initialization.json')
    if initial.get('checkpoint_loaded') is not False:
        raise ValueError('Expected fresh-policy training')
    payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if payload['architecture'] != 'pose_token_diffusion_v3' or payload['training']['epoch'] != epoch:
        raise ValueError('V3 checkpoint identity or epoch mismatch')
    if payload['training'] != evidence['completed_epoch']:
        raise ValueError('Checkpoint training metadata differs from the recorded completed epoch')
    reference = read(args.reference_plan)
    if sha(training/'demonstrations.json') != sha(Path(reference['training_study'])/'demonstrations.json'):
        raise ValueError('Phase diagnostic needs the same held-out demonstration selection')
    paths = source_paths() + [Path(__file__), ROOT/'part2action/slurm/evaluate_pose_token_v3_phase.sbatch',
        ROOT/'part2action/scripts/rollout_partgym.py', ROOT/'part2action/diagnostics/evaluation_protocol.py',
        ROOT/'part2action/diagnostics/execution_trace.py', ROOT/'part2action/diagnostics/evaluation_scenarios.py',
        ROOT/'part2action/diagnostics/partgym_skill_oracle.py',
        ROOT/'part2action/configs/evaluation/partinstruct_native.json']
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym').rglob('*.py'))
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym/config').glob('*.yaml'))
    if args.source_change_review:
        paths.append(args.source_change_review.resolve(strict=True))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    register_result_dir(args.out_dir)
    if (args.out_dir/'plan.json').exists():
        raise FileExistsError('Phase diagnostic plan already exists')
    plan = {'scope': 'Privileged phase-progression intervention on held-out training-release scenarios; not Test1 or VLM planning.',
        'training_study': str(training), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'checkpoint_epoch': epoch, 'checkpoint_selection': 'configured final epoch' if args.checkpoint_epoch is None else 'explicit epoch',
        'training_evidence': evidence,
        'initialization': initial, 'arms': list(ARMS), 'data_root': reference['data_root'],
        'scenarios': reference['scenarios'], 'trials': reference['trials'],
        'execution_settings': reference['execution_settings'], 'reference_plan': str(args.reference_plan.resolve()),
        'reference_plan_sha256': sha(args.reference_plan),
        'source_sha256': {str(p.resolve()): sha(p) for p in sorted(set(paths))},
        'limits': f'Epoch-{epoch} checkpoint, one training seed, one reset seed, four objects. Oracle predicates are simulator definitions, not proof of secure retention or an optimal controller.'}
    write(args.out_dir/'plan.json', plan)
    snapshot = args.out_dir/'source_snapshot'
    snapshot.mkdir()
    for i, path in enumerate(sorted(set(paths))):
        shutil.copy2(path, snapshot/f'{i:04d}_{path.name}')


def verify(plan):
    if sha(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    for name, wanted in plan['source_sha256'].items():
        if sha(name) != wanted:
            raise ValueError('Diagnostic implementation changed: ' + name)


def evaluate(args):
    import gc
    import torch
    from omegaconf import OmegaConf
    import rollout_partgym as helpers
    from diagnostics.evaluation_protocol import align_native_environment
    from diagnostics.evaluation_scenarios import pin_scenario
    from diagnostics.partgym_skill_oracle import PartGymSkillOracle
    from models.pose_token_diffusion_v3.config import PoseTokenDiffusionV3Config
    from models.pose_token_diffusion_v3.policy import build_policy
    from models.pose_token_diffusion_v3.checkpoint import load_checkpoint
    from models.pose_token_diffusion_v3.evaluation import rollout_trial
    plan = read(args.out_dir/'plan.json')
    verify(plan)
    arm = ARMS[args.index]
    folder = args.out_dir/arm
    folder.mkdir(exist_ok=False)
    payload = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
    config = PoseTokenDiffusionV3Config.from_mapping(payload['config'])
    del payload
    helpers.set_seed(42)
    model = build_policy(config)
    load_checkpoint(model, plan['checkpoint'])
    model.cuda().eval()
    env_class = helpers._load_partgym_env(ROOT/'PartInstruct')
    sim = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    native = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/baselines/evaluation/config/env/env_config.yaml')
    align_native_environment(sim, native)
    sim.data_root, sim.device, sim.meta_path = plan['data_root'], 'cpu', 'episodes_meta_train.json'
    full = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config.yaml')
    helpers.configure_partgym_agentview_pcd(sim, full)
    for i, trial in enumerate(plan['trials']):
        scenario = plan['scenarios'][trial['scenario_index']]
        identity = {'trial_index': i, 'seed': trial['seed'], 'scenario_index': trial['scenario_index'],
            'family': scenario['family'], 'object': scenario['obj_class'], 'checkpoint_sha256': plan['checkpoint_sha256']}
        env = None
        directory = folder/f'{i:03d}'
        directory.mkdir()
        try:
            helpers.set_seed(trial['seed'])
            env = env_class(config=sim, gui=False, record=True, evaluation=True, skill_mode=False,
                obj_class=scenario['obj_class'], split='train', task_type=scenario['task_type'], track_samples=False)
            pin_scenario(env, scenario['episode'], trial['seed'])
            instructions = scenario['episode']['skill_instructions']
            if arm == 'oracle_progression':
                runtime = OracleDiagnosticRuntime(model, OracleDiagnosticPlanner(instructions), env, PartGymSkillOracle())
            else:
                runtime = HierarchicalRolloutPolicy(model, SuppliedPlanDiagnostic(instructions))
            result = rollout_trial(env, runtime, device=torch.device('cuda'), n_obs_steps=config.n_obs_steps,
                                   settings=plan['execution_settings'])
            if str(env.ep_id) != str(scenario['episode']['ep_id']):
                raise ValueError('Pinned reset scenario mismatch')
            result.update(identity, status='completed', arm=arm,
                oracle_events=getattr(runtime, 'oracle_events', []),
                phase_advances=sum(a['skill']['step_id'] != b['skill']['step_id'] for a, b in
                    zip(runtime.history, runtime.history[1:]) if a['skill'] and b['skill']))
            env.dump_buffers()
            video = directory/'rollout.mp4'
            env.save_renders(str(video), video_only=True)
            result['rollout_video'] = str(video.resolve())
        except Exception as error:
            result = {**identity, 'status': 'runtime_error', 'arm': arm, 'error': repr(error)}
        finally:
            if env is not None:
                env.close()
        write(directory/'result.json', result)
        print(json.dumps({k: result.get(k) for k in ('trial_index', 'arm', 'status', 'success', 'phase_advances', 'error')}), flush=True)
        gc.collect()


def paired_summary(plan, records):
    paired, gaps = [], []
    by_family = {}
    for i, trial in enumerate(plan['trials']):
        rows = [records.get(arm, {}).get(i) for arm in ARMS]
        if any(not row or row.get('status') != 'completed' for row in rows):
            gaps.append({'trial_index': i, 'reason': 'missing_or_error'})
            continue
        for row in rows:
            if row['trial_index'] != i or row['scenario_index'] != trial['scenario_index'] or row['seed'] != trial['seed'] or row['checkpoint_sha256'] != plan['checkpoint_sha256'] or row['execution_settings'] != plan['execution_settings']:
                raise ValueError('Mismatched checkpoint, seed, or execution settings')
            if not 0 <= row['num_actions'] <= plan['execution_settings']['max_steps']:
                raise ValueError('Trial exceeds the executed-action budget')
        state_a, state_b = rows[0]['initial_state'], rows[1]['initial_state']
        if state_a.keys() != state_b.keys() or any(not np.allclose(np.asarray(state_a[k]), np.asarray(state_b[k]), atol=1e-6, rtol=0) for k in state_a):
            gaps.append({'trial_index': i, 'reason': 'physical_reset_mismatch'})
            continue
        paired.append(i)
        family = plan['scenarios'][trial['scenario_index']]['family']
        entry = by_family.setdefault(family, {'paired_trials': 0, **{arm: {'task_successes': 0, 'trials_with_phase_advance': 0} for arm in ARMS}})
        entry['paired_trials'] += 1
        for arm, row in zip(ARMS, rows):
            entry[arm]['task_successes'] += bool(row['success'])
            entry[arm]['trials_with_phase_advance'] += row['phase_advances'] > 0
    return {'status': 'complete' if len(paired) == len(plan['trials']) else 'incomplete',
        'scope': plan['scope'], 'planned_per_arm': len(plan['trials']), 'paired_trial_indices': paired,
        'coverage_gaps': gaps, 'by_family': by_family,
        'interpretation': 'An oracle-progression gain suggests a progression limitation under these predicates. Failure in both arms leaves execution/grounding/goals or insufficient training unresolved. No bottleneck conclusion is drawn from unmatched or missing trials.'}


def report(args):
    plan = read(args.out_dir/'plan.json')
    records = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for path in (args.out_dir/arm).glob('*/result.json'):
            row = read(path)
            records[arm][row['trial_index']] = row
    summary = paired_summary(plan, records)
    write(args.out_dir/'summary.json', summary)
    lines = ['# V3 phase-progression diagnostic', '', plan['scope'], '',
        '| Group | Matched trials | Learned progression success | Oracle progression success |',
        '| --- | ---: | ---: | ---: |']
    for family, row in summary['by_family'].items():
        n = row['paired_trials']
        lines.append(f"| {family} | {n} | {row[ARMS[0]]['task_successes']}/{n} | {row[ARMS[1]]['task_successes']}/{n} |")
    lines += ['', summary['interpretation'], '', plan['limits'], '',
              'See per-arm trial folders for videos, action/goal traces, planner reviews and oracle decisions.']
    (args.out_dir/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'evaluate', 'report'], required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--training-study', type=Path)
    parser.add_argument('--reference-plan', type=Path)
    parser.add_argument('--checkpoint-epoch', type=int, help='Default: final epoch from the training recipe')
    parser.add_argument('--allow-interrupted-training', action='store_true',
                        help='Evaluate an explicitly selected completed epoch from a recorded failed/cancelled run')
    parser.add_argument('--source-change-review', type=Path,
                        help='Hash-pinned review of operational-only changes since training')
    parser.add_argument('--index', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    configure_cache_environment()
    args.out_dir = result_dir(args.out_dir)
    {'prepare': prepare, 'evaluate': evaluate, 'report': report}[args.mode](args)


if __name__ == '__main__':
    main()
