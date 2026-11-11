"""Export per-seed and paired outcomes after the frozen visual-pose reporter."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args(); root=args.out_dir.resolve()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=args.source_sha256:
        raise RuntimeError('Statistics source changed after submission')
    summary=json.loads((root/'summary.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    rows=[]; means={}
    for arm in ('control','oracle'):
        arm_rows=[]
        for seed in plan['config']['evaluation']['seeds']:
            paired=[r for r in summary['paired'] if r['seed']==seed]
            grasps=[r for r in paired if r['grasp_eligible']]
            row={'arm':arm,'seed':seed,'paired_trials':len(paired),'planned_trials':len(plan['scenarios']),
                 'task_successes':sum(r[arm+'_task'] for r in paired),
                 'task_success_rate':sum(r[arm+'_task'] for r in paired)/len(paired) if paired else None,
                 'grasp_trials':len(grasps),'grasp_successes':sum(r[arm+'_grasp'] for r in grasps)}
            rows.append(row); arm_rows.append(row)
        rates=[r['task_success_rate'] for r in arm_rows]
        complete=all(r['paired_trials']==r['planned_trials'] for r in arm_rows)
        means[arm]={'full_coverage':complete,
                    'mean_seed_task_success_rate':statistics.mean(rates) if complete else None,
                    'sample_std_seed_task_success_rate':statistics.stdev(rates) if complete and len(rates)>1 else None,
                    'scope':'evaluation randomness; one training seed; means withheld with coverage gaps'}
    for name,data in [('per_seed_outcomes.csv',rows),('paired_outcomes.csv',summary['paired'])]:
        with (root/name).open('w') as handle:
            if data:
                writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (root/'per_seed_statistics.json').write_text(json.dumps({'per_seed':rows,'statistics':means},indent=2)+'\n')
    lines=['','## Per-seed matched outcomes','',
           '| Arm | Seed | Paired / planned | Task success | Initial-phase-end grasp |',
           '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f'| {r["arm"]} | {r["seed"]} | {r["paired_trials"]}/{r["planned_trials"]} | '
                     f'{r["task_successes"]}/{r["paired_trials"]} | {r["grasp_successes"]}/{r["grasp_trials"]} |')
    lines += ['', 'Seed means and sample standard deviations are reported only with full planned coverage.',
              '[Per-seed statistics](per_seed_statistics.json) · [Per-seed CSV](per_seed_outcomes.csv) · [Paired CSV](paired_outcomes.csv)']
    with (root/'README.md').open('a') as handle:
        handle.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
