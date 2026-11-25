"""Independent PartGym runner for predicted-VLM or explicitly supplied-plan v3 trials."""
from collections import deque
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from diagnostics.storage import result_dir, register_result_dir, configure_cache_environment

from .config import PoseTokenDiffusionV3Config
from .checkpoint import load_checkpoint
from .policy import build_policy
from .planner import PlannerDecision, QwenVLMPlanner
from .skills import annotated_skill
from .runtime import HierarchicalRolloutPolicy

ROOT = Path(__file__).resolve().parents[3]


class SuppliedPlanDiagnostic:
    """Privileged semantic-plan diagnostic; transitions still use the learned monitor.

    The supplied list is intentionally labelled as a diagnostic. No simulator
    predicate, demonstration timestamp, or GT geometric goal advances this plan.
    """
    source = 'supplied_plan_diagnostic'

    def __init__(self, instructions):
        if not instructions:
            raise ValueError('Supplied-plan diagnostic needs skill instructions')
        self.skills = [annotated_skill(text, previous=instructions[:i], step_id=str(i))
                       for i, text in enumerate(instructions)]
        self.index = 0

    def plan(self, request):
        if request.reason == 'monitor_requested_review' and self.index < len(self.skills)-1:
            self.index += 1
        return PlannerDecision('execute', self.skills[self.index])


def rollout_trial(env, runtime, *, device, n_obs_steps, settings):
    import rollout_partgym as helpers
    from data.geometry import bound_gripper_command
    from diagnostics.evaluation_protocol import execution_prefix
    from diagnostics.execution_trace import physical_state
    observation = env.reset()
    initial = physical_state(env)
    history = deque([np.asarray(observation['agentview_rgb'], dtype=np.uint8)], maxlen=n_obs_steps)
    actions, goals, latency = [], [], []
    info, stop = {}, 'action_budget'
    runtime.reset()
    while len(actions) < settings['max_steps']:
        rgb = helpers._make_rgb_tensor(history, n_obs_steps, device)
        inputs = helpers._model_observations(observation, device)
        started = time.perf_counter()
        output = runtime.act(rgb=rgb, scene_points_camera=inputs['agentview_pcd'],
            tcp_pose=inputs['tcp_pose'], gripper_opening=inputs['gripper_state'], task=env.task_instruction)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        latency.append((time.perf_counter()-started)*1000)
        if output['status'] != 'execute':
            stop = output['status']
            break
        goals.append({'prediction': len(goals), 'skill': output['current_skill'],
            'position_world': output['predicted_goal_position_world'][0, 0].float().cpu().tolist(),
            'opening_m': float(output['predicted_goal_opening'][0, 0]),
            'progress_probability': output['progress_probability']})
        chunk = output['action_chunk'][0].detach().float().cpu().numpy()
        done = False
        for command in execution_prefix(chunk, settings['execute_steps'], settings['max_steps']-len(actions)):
            command = bound_gripper_command(command)
            observation, _, done, info = env.step(command)
            actions.append(command.astype(float).tolist())
            history.append(np.asarray(observation['agentview_rgb'], dtype=np.uint8))
            if done:
                stop = 'environment_done'
                break
        if done:
            break
    return {'success': bool(info.get('Success', False)), 'stop_reason': stop,
        'completion_rate': float(info.get('Completion Rate', 0.)), 'num_actions': len(actions),
        'num_policy_predictions': len(latency), 'inference_latency_ms': latency,
        'planner_source': runtime.planner.source, 'planner_history': runtime.history,
        'success_definition': 'Native PartGym Success only; planner done/readiness are not success',
        'initial_state': initial, 'final_state': physical_state(env),
        'actions': actions, 'goals': goals, 'execution_settings': settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--development-plan', type=Path, required=True)
    parser.add_argument('--trial-index', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--planner', choices=['qwen', 'supplied-plan'], default='qwen')
    parser.add_argument('--planner-revision')
    parser.add_argument('--planner-device', default='cuda')
    args = parser.parse_args()
    configure_cache_environment()
    args.out_dir = result_dir(args.out_dir)
    if args.out_dir.exists():
        raise FileExistsError('Choose a fresh v3 rollout directory')
    if not torch.cuda.is_available():
        raise RuntimeError('Use a GPU allocation with PartGym dependencies')
    import rollout_partgym as helpers
    from omegaconf import OmegaConf
    from diagnostics.evaluation_protocol import align_native_environment, execution_settings
    from diagnostics.evaluation_scenarios import pin_scenario
    plan = json.loads(args.development_plan.read_text())
    if not 0 <= args.trial_index < len(plan['trials']):
        raise ValueError('Invalid trial index')
    trial = plan['trials'][args.trial_index]
    scenario = plan['scenarios'][trial['scenario_index']]
    payload = torch.load(args.ckpt, weights_only=True, map_location='cpu')
    config = PoseTokenDiffusionV3Config.from_mapping(payload['config'])
    del payload
    helpers.set_seed(trial['seed'])
    model = build_policy(config)
    load_checkpoint(model, args.ckpt)
    model.cuda().eval()
    planner = (QwenVLMPlanner(revision=args.planner_revision, device=args.planner_device)
               if args.planner == 'qwen' else SuppliedPlanDiagnostic(scenario['episode']['skill_instructions']))
    runtime = HierarchicalRolloutPolicy(model, planner)
    settings = execution_settings('test1')
    if config.action_horizon < settings['execute_steps']:
        raise ValueError('Policy horizon is shorter than released execution cadence')
    sim = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    native = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/baselines/evaluation/config/env/env_config.yaml')
    align_native_environment(sim, native)
    sim.data_root, sim.device, sim.meta_path = plan['data_root'], 'cpu', 'episodes_meta_train.json'
    full = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config.yaml')
    helpers.configure_partgym_agentview_pcd(sim, full)
    env_class = helpers._load_partgym_env(ROOT/'PartInstruct')
    env = env_class(config=sim, gui=False, record=True, evaluation=True, skill_mode=False,
        obj_class=scenario['obj_class'], split='train', task_type=scenario['task_type'], track_samples=False)
    args.out_dir.mkdir(parents=True)
    register_result_dir(args.out_dir)
    try:
        pin_scenario(env, scenario['episode'], trial['seed'])
        result = rollout_trial(env, runtime, device=torch.device('cuda'), n_obs_steps=config.n_obs_steps, settings=settings)
        if str(env.ep_id) != str(scenario['episode']['ep_id']):
            raise ValueError('Reset did not select the pinned development scenario')
        env.dump_buffers()
        video = args.out_dir/'rollout.mp4'
        env.save_renders(str(video), video_only=True)
        result.update(status='completed', trial_index=args.trial_index, seed=trial['seed'],
            checkpoint_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
            development_plan_sha256=hashlib.sha256(args.development_plan.read_bytes()).hexdigest(),
            rollout_video=str(video.resolve()), scope='Held-out training-release development; not the reserved benchmark.',
            planner_model=QwenVLMPlanner.MODEL_ID if args.planner == 'qwen' else None,
            planner_revision=args.planner_revision)
        (args.out_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except Exception as error:
        (args.out_dir/'error.json').write_text(json.dumps({'status': 'runtime_error', 'error': repr(error)})+'\n')
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
