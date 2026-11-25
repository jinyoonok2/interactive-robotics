"""Prepare an isolated v3 pilot; preparation never submits a job or downloads weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml
from diagnostics.storage import result_dir, register_result_dir, configure_cache_environment
from models.pose_token_diffusion_v2.pilot import prepare_pilot

PACKAGE = Path(__file__).resolve().parents[1]/'models/pose_token_diffusion_v3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path)
    parser.add_argument('--per-object', type=int, default=64)
    args = parser.parse_args()
    configure_cache_environment()
    args.out_dir = result_dir(args.out_dir)
    if args.out_dir.exists():
        raise FileExistsError('Choose a new v3 study directory')
    recipe = yaml.safe_load((PACKAGE/'configs/pilot.yaml').read_text())
    if args.reference_manifest:
        manifest = json.loads(args.reference_manifest.read_text())
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported reference demonstration manifest')
    else:
        manifest = prepare_pilot(recipe['data']['train_hdf5'], per_object=args.per_object,
            validation_fraction=recipe['data']['validation_fraction'], seed=recipe['train']['seed'])
    args.out_dir.mkdir(parents=True)
    register_result_dir(args.out_dir)
    target = args.out_dir/'demonstrations.json'
    if args.reference_manifest:
        shutil.copy2(args.reference_manifest, target)
    else:
        target.write_text(json.dumps(manifest, indent=2)+'\n')
    recipe['data']['demo_manifest'] = str(target.resolve())
    recipe['model_config'] = str((PACKAGE/'configs/model.yaml').resolve())
    (args.out_dir/'pilot_resolved.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    (args.out_dir/'preparation.json').write_text(json.dumps({
        'architecture': 'pose_token_diffusion_v3', 'status': 'prepared_not_submitted',
        'reference_manifest': str(args.reference_manifest.resolve()) if args.reference_manifest else None,
        'manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'demonstrations': len(manifest['demonstrations']),
        'training_conditioning': 'Annotated current skills; predicted geometric goals. Not VLM planning.',
        'comparison_limits': 'V3 includes short/tail observations and masks cross-skill actions. Same demos do not imply matched updates, labels, or architecture.'}, indent=2)+'\n')
    (args.out_dir/'README.md').write_text(
        '# V3 development pilot preparation\n\nPrepared only; no training submitted. '
        'Fresh trainable policy with frozen cached DINOv2/Flan-T5. '
        'Current skill annotations are training conditioning, not predicted VLM outputs.\n\n'
        'Run GPU smoke using pilot_resolved.yaml; training requires its matching smoke.json receipt.\n')
    print(str(args.out_dir/'pilot_resolved.yaml'))


if __name__ == '__main__':
    main()
