"""Isolated evaluation adapter around the unchanged Part2Action rollout runner."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from diagnostics.frozen_policy import FrozenPolicyDiagnostic, MODES


def oracle_mask(env, phase):
    from diagnostics.partgym_reach import _target_part_points
    from PartInstruct.PartGym.env.backend.utils.vision_utils import project_point_cloud_to_mask
    metadata, points = _target_part_points(env, phase)
    if points is None:
        return None, metadata
    c, p = env.config, env.world.p
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=c.cam_static_target,
        distance=c.cam_static_dist, yaw=c.cam_static_yaw, pitch=c.cam_static_pitch,
        roll=0, upAxisIndex=c.up_axis_index)
    projection = p.computeProjectionMatrixFOV(fov=60, aspect=float(c.render_width)/c.render_height,
        nearVal=c.cam_near, farVal=c.cam_far)
    mask = project_point_cloud_to_mask(points, c.render_width, c.render_height, view, projection)
    return mask, metadata


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-mode', required=True, choices=MODES)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    import rollout_partgym as base
    context = FrozenPolicyDiagnostic(args.diagnostic_mode, oracle_mask)
    load_model, load_env = base._load_model, base._load_partgym_env
    rollout, diagnostic = base.rollout_one, base._rollout_diagnostics

    def model_loader(*a, **kw):
        return context.attach(load_model(*a, **kw))

    def env_loader(*a, **kw):
        parent = load_env(*a, **kw)
        class TrackedEnvironment(parent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                context.env = self
        return TrackedEnvironment

    def trial(*a, **kw):
        context.begin_trial(kw['episode_seed'])
        row = rollout(*a, **kw)
        row['policy_diagnostic_mode'] = args.diagnostic_mode
        return row

    def step_diagnostic(*a, **kw):
        row = diagnostic(*a, **kw)
        row['policy_diagnostic'] = dict(context.last)
        return row

    base._load_model, base._load_partgym_env = model_loader, env_loader
    base.rollout_one, base._rollout_diagnostics = trial, step_diagnostic
    base.main()
    out = Path(remaining[remaining.index('--out-dir') + 1]) / 'hierarchical'
    metadata = {'mode':args.diagnostic_mode,
        'rng_protocol':'trial/step-specific forked DDPM noise; independent of global RNG' if args.diagnostic_mode.startswith('target_') else 'unchanged standard runner RNG',
        'oracle_scope':'GT grasp/touch part at learned phase index; initial call uses index 0; unsupported phases use identity gate' if args.diagnostic_mode == 'gate_oracle' else None,
        'oracle_mask_source':'PartGym GT part geometry projected with native camera/mask routine; not occlusion-corrected visible segmentation' if args.diagnostic_mode == 'gate_oracle' else None,
        'privileged_policy_input': args.diagnostic_mode == 'gate_oracle',
        'limits':['Frozen inference intervention can introduce distribution shift.',
                  'Diagnostic forward timings include intervention hooks and oracle projection where used; not a pure model-speed comparison.',
                  'Gate interventions also affect phase/contact features, not only action features.',
                  'Target prediction traces remain original; conditioned target coordinates are separately logged.',
                  'Closed-loop observations diverge; matching noise does not make later observations identical.'],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).parents[1]/'diagnostics/frozen_policy.py']}}
    for path in out.glob('rollout_results_*.json'):
        payload=json.loads(path.read_text());payload['frozen_policy_diagnostic']=metadata
        path.write_text(json.dumps(payload,indent=2)+'\n')
    (out/'diagnostic_protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    main()
