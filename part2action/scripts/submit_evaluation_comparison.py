"""Queue matched GPU evaluations, then automatically export one flat GIF review."""
import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

from register_result import register
from diagnostics.evaluation_scenarios import load_scenarios, DEFAULT_EVALUATION_SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=TRAINING_DIRECTORY; repeat for every panel')
    parser.add_argument('--out-dir', type=Path, required=True, help='New comparison directory under project storage')
    parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[2]/'datasets'/'PartInstruct',help='Dataset used for evaluations and the leftmost expert reference')
    parser.add_argument('--dry-run', action='store_true', help='Print all commands without creating files or submitting jobs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.out_dir.resolve()
    data_root=args.data_root.resolve(strict=True)
    scenarios = load_scenarios(data_root / 'episodes_meta_test.json', 'test1',
                              ['bottle', 'mug', 'pliers', 'scissors'])
    if len(scenarios) != 149:
        parser.error(f'Expected 149 test1 scenarios, found {len(scenarios)}')
    if ',' in str(data_root):
        parser.error('Slurm export paths cannot contain commas')
    runs = {}
    for spec in args.run:
        label, separator, directory = spec.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', label) or label in runs:
            parser.error('Use unique labels containing letters, numbers, underscores or hyphens')
        directory = Path(directory).resolve(strict=True)
        for required in ['last.pt', 'config_source.yaml']:
            if not (directory / required).is_file():
                parser.error(f'Missing training artifact: {directory / required}')
        if ',' in str(directory) or ',' in str(output):
            parser.error('Slurm export paths cannot contain commas')
        runs[label] = directory
    if len(runs) < 2:
        parser.error('Supply at least two --run arguments for a comparison')
    if output.exists():
        parser.error('Choose a new output directory; previous evaluations are never overwritten')
    request = {'out_dir': str(output), 'sources': [f'{label}={output / "_raw" / label}' for label in runs],
               'panel_order': ['ground_truth_reference'] + list(runs), 'data_root':str(data_root),'speed': .5, 'fps': 5,
               'evaluation_protocol': 'all_test1_scenarios_fixed_seeds', 'expected_episodes': 149,
               'evaluation_seeds': list(DEFAULT_EVALUATION_SEEDS), 'trials_per_scenario': 3, 'expected_rollouts_per_model': 447}
    request['metadata_sha256'] = hashlib.sha256((data_root / 'episodes_meta_test.json').read_bytes()).hexdigest()
    request['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [root / 'part2action/scripts/rollout_partgym.py',
                     root / 'part2action/diagnostics/evaluation_scenarios.py']}
    if not args.dry_run:
        output.mkdir(parents=True)
        register('part2action/evaluations', output)
        (root / 'part2action/logs/slurm').mkdir(parents=True, exist_ok=True)
        (output / 'comparison_request.json').write_text(json.dumps(request, indent=2) + '\n')
        (output / 'evaluation_scenarios.json').write_text(json.dumps(
            {'split': 'test1', 'evaluation_seeds': list(DEFAULT_EVALUATION_SEEDS), 'trials_per_scenario': 3, 'expected_rollouts_per_model': 447, 'scenarios': scenarios}, indent=2) + '\n')
    submissions, ids = [], []
    for label, directory in runs.items():
        raw = output / '_raw' / label
        exports = f'ALL,RUN_DIR={directory},OUT_DIR={raw},DATA_ROOT={data_root},REGISTER_RESULTS=0,FLAT_REVIEW=0,RECORD=1,CONTACT_DIAGNOSTICS=1,CONTACT_REPORT=1,EVAL_SEEDS=42 43 44,EXPECTED_EPISODES=149,NUM_EPISODES='
        command = ['sbatch', '--parsable', f'--export={exports}', str(root / 'part2action/slurm/rollout_bigtemp_full_eval_a40.sbatch')]
        print(shlex.join(command), flush=True)
        job = f'DRY_RUN_{label}' if args.dry_run else subprocess.check_output(command, cwd=root, text=True).strip().split(';')[0]
        ids.append(job)
        submissions.append({'model': label, 'job_id': job, 'run_dir': str(directory), 'out_dir': str(raw)})
        if not args.dry_run:
            (output / 'submissions.json').write_text(json.dumps(submissions, indent=2) + '\n')
    report = ['sbatch', '--parsable', f'--dependency=afterok:{":".join(ids)}',
              f'--export=ALL,COMPARISON_REQUEST={output / "comparison_request.json"}',
              str(root / 'part2action/slurm/export_evaluation_gifs.sbatch')]
    print(shlex.join(report), flush=True)
    if not args.dry_run:
        report_id = subprocess.check_output(report, cwd=root, text=True).strip().split(';')[0]
        request['report_job_id'] = report_id
        (output / 'comparison_request.json').write_text(json.dumps(request, indent=2) + '\n')
        print(f'GIF review will be generated automatically after all evaluations succeed: {output}')


if __name__ == '__main__':
    main()
