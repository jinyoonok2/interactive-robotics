"""Reusable matched-rollout report using existing contact-motion summaries.

Example: --source baseline=.../_raw/baseline --source transformer=.../_raw/transformer
Only recorded measurements are used. Geometry is not physical contact evidence.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from summarize_contact_impact import load_trials

GEOMETRY = ['start_contact_to_part_cm', 'mean_contact_to_part_cm',
            'end_contact_to_part_cm', 'mean_tcp_to_part_cm', 'end_tcp_to_part_cm',
            'mean_action_to_part_cm', 'phase_changes']
LIMITS = [
    'One training seed per model. Evaluation seeds do not measure training variability.',
    'Success-difference interval resamples whole scenarios, keeping evaluation-seed repeats together; it describes scenario sampling uncertainty, not training uncertainty.',
    'Distances refer to the first ground-truth task part tracked with object motion. They are sampled-surface proximity, not physical contact, stable grasp, or unique contact-target accuracy.',
    'First-skill groups describe tasks starting with that skill; their outcome is full-task success, not individual-skill success.',
    'Trajectory averages cover different visited states and lengths. They do not establish a causal mechanism or fair speed-to-success on their own.',
    'Gripper clipping measures command saturation, not actual finger closure or grasp success.',
    'Policy latency excludes observation preparation, simulation, diagnostics and media export. Hardware is reported separately.',
    'No new training, inference, contact intervention or physics measurement is performed by this report.',
]


def mean(values):
    values=[v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(values) if values else None


def aggregate(rows):
    result={'trials':len(rows),'successes':sum(r['success'] for r in rows),
            'success_rate':mean([int(r['success']) for r in rows])}
    for field in GEOMETRY+['completion_rate','gripper_clip_fraction','steps']:
        vals=[r[field] for r in rows if r[field] is not None]
        result[field]={'mean':mean(vals),'n':len(vals)}
    latency=[v for r in rows for v in r['steady_latency']]
    result['policy_forward_steady_ms']={'mean':mean(latency),'cycles':len(latency)}
    return result


def percentile(values, p):
    values=sorted(values);at=(len(values)-1)*p;lo=math.floor(at);hi=math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def compare(a,b,repetitions):
    assert set(a)==set(b), 'Scenario/seed keys differ'
    groups={}
    outcomes={'both_success':0,'a_only_success':0,'b_only_success':0,'both_fail':0}
    for k in a:
        x,y=bool(a[k]['success']),bool(b[k]['success'])
        category='both_success' if x and y else 'a_only_success' if x else 'b_only_success' if y else 'both_fail'
        outcomes[category]+=1
        groups.setdefault(k[:3],[]).append(int(y)-int(x))
    # Equal seed counts required so equal-scenario and equal-trial estimands agree.
    assert len({len(v) for v in groups.values()})==1
    values=[statistics.mean(v) for v in groups.values()]
    rng=random.Random(42)
    samples=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(repetitions)]
    return {'paired_outcomes':outcomes,'success_difference_b_minus_a':statistics.mean(values),
            'scenario_cluster_bootstrap_95_interval':[percentile(samples,.025),percentile(samples,.975)],
            'bootstrap_repetitions':repetitions,'bootstrap_seed':42,'unique_scenarios':len(values)}


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',action='append',required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--bootstrap-repetitions',type=int,default=5000)
    args=parser.parse_args()
    if args.bootstrap_repetitions<100: parser.error('Use at least 100 bootstrap repetitions')
    root=Path(__file__).resolve().parents[2]
    meta_path=args.data_root/'episodes_meta_test.json'
    meta=json.loads(meta_path.read_text())
    scenarios={(o,str(t),str(ep['ep_id'])):ep for o,s in meta.items()
               for t,eps in s.get('test1',{}).items() for ep in eps}
    sources={};provenance={};protocols=[];gpu={};interventions={}
    for spec in args.source:
        label,sep,directory=spec.partition('=')
        if not sep or not label or label in sources: parser.error('Use unique LABEL=DIRECTORY sources')
        directory=Path(directory).resolve(strict=True)
        csv_path=directory/'contact_motion_report/per_trial.csv'
        rows=load_trials(csv_path)
        paths=list((directory/'hierarchical').glob('rollout_results_*.json'))
        if len(paths)!=1: raise ValueError('Expected one rollout JSON: '+str(directory))
        payload=json.loads(paths[0].read_text())
        if payload.get('frozen_policy_diagnostic'):
            interventions[label]=payload['frozen_policy_diagnostic']
        protocols.append({f:payload.get(f) for f in ['split','termination_mode','evaluation_protocol','evaluation_seeds','gripper_command_bounds_m']})
        raw={(r['obj_class'],str(r['task_type']),str(r['requested_episode_id']),int(r['seed'])):r for r in payload['results']}
        assert len(raw)==len(payload['results']) and set(raw)==set(rows)
        for k,row in rows.items():
            trial=raw[k]
            assert row['success']==bool(trial['success'])
            row['completion_rate']=float(trial['completion_rate'])
            row['steps']=int(trial['steps'])
            row['steady_latency']=[float(v) for v in trial['policy_forward_latency_ms'][1:]]
            row['gripper_clip_fraction']=row['gripper_clipped_count']/trial['num_actions'] if trial['num_actions'] else None
            row['first_skill']=scenarios[k[:3]]['chain_params'][0]['skill_name']
        sources[label]=rows;gpu[label]=payload.get('gpu_name')
        provenance[label]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [csv_path,paths[0]]}
    assert len(sources)>=2
    assert all(p==protocols[0] for p in protocols), 'Evaluation protocol fields differ'
    labels=list(sources);reference=labels[0];keys=set(sources[reference])
    assert all(set(rows)==keys for rows in sources.values())
    for k in keys:
        assert len({sources[label][k]['first_skill'] for label in labels})==1
    report={'reference':reference,'protocol':protocols[0],'gpu':gpu,'models':{},'comparisons':{},
            'per_object':{},'per_task':{},'per_first_skill':{},'per_seed':{},'limits':LIMITS,
            'source_sha256':provenance,'metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest()}
    if interventions:
        report['interventions']=interventions
        report['limits']=LIMITS+sorted({limit for item in interventions.values() for limit in item.get('limits',[])})
        report['limits'].append('Oracle gates use privileged GT part geometry and are diagnostics, not deployable-policy or retrained-ablation results.')
    for label,rows in sources.items():
        report['models'][label]=aggregate(list(rows.values()))
        for field,target in [('object','per_object'),('task_type','per_task'),('first_skill','per_first_skill'),('seed','per_seed')]:
            for value in sorted({str(r[field]) for r in rows.values()}):
                report[target].setdefault(value,{})[label]=aggregate([r for r in rows.values() if str(r[field])==value])
        if label!=reference:
            report['comparisons'][label]=compare(sources[reference],rows,args.bootstrap_repetitions)
    out=args.out_dir;out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    table_rows=[];pairs=[]
    for k in sorted(keys):
        pair={'object':k[0],'task_type':k[1],'episode_id':k[2],'seed':k[3]}
        for label,rows in sources.items():
            row=rows[k];pair[label+'_success']=row['success']
            table_rows.append({'model':label,**{f:row[f] for f in ['object','task_type','episode_id','seed','first_skill','success','completion_rate','steps',*GEOMETRY,'gripper_clip_fraction']}})
        pairs.append(pair)
    write_csv(out/'per_trial.csv',table_rows);write_csv(out/'paired_outcomes.csv',pairs)
    # Reuse the existing slide-table renderer rather than introducing one.
    scripts=root/'part2action/docs/slide_history/scripts';sys.path.insert(0,str(scripts))
    from export_slide_tables import render
    date=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    media=out/'media'/date;media.mkdir(parents=True)
    metrics=[('Task success','success_rate'),('Completion (0-1)','completion_rate'),
             ('Contact / part (cm)','mean_contact_to_part_cm'),('TCP / part (cm)','mean_tcp_to_part_cm'),
             ('Final TCP / part (cm)','end_tcp_to_part_cm'),('Gripper clip fraction','gripper_clip_fraction'),
             ('Policy forward (ms)','policy_forward_steady_ms')]
    overall=[['Metric',*labels]]
    for title,metric in metrics:
        values=[]
        for label in labels:
            item=report['models'][label][metric];value=item['mean'] if isinstance(item,dict) else item
            values.append('n/a' if value is None else f'{100*value:.1f}%' if metric=='success_rate' else f'{value:.3f}')
        overall.append([title,*values])
    fractions=[.46]+[.54/len(labels)]*len(labels)
    render(overall,fractions,media/'overall_metrics.png')
    for group,title,filename in [('per_object','Object','per_object_success'),
                                 ('per_first_skill','First task skill','per_first_skill_success')]:
        table=[[title,*labels]]
        for value,models in report[group].items():
            table.append([value,*[f"{models[l]['successes']}/{models[l]['trials']} ({100*models[l]['success_rate']:.1f}%)" for l in labels]])
        render(table,fractions,media/(filename+'.png'))
    sections=[]
    for label,c in report['comparisons'].items():
        low,high=c['scenario_cluster_bootstrap_95_interval']
        sections.append(f'<p>{html.escape(label)} minus {html.escape(reference)}: {100*c["success_difference_b_minus_a"]:.2f} percentage points; scenario-cluster bootstrap 95% interval [{100*low:.2f}, {100*high:.2f}]. Paired outcomes: {html.escape(str(c["paired_outcomes"]))}.</p>')
    page='<!doctype html><meta charset="utf-8"><title>Matched quantitative evaluation</title><style>body{font:17px system-ui;max-width:1200px;margin:auto;padding:24px}img{max-width:100%}li{margin:10px}</style><h1>Matched quantitative evaluation</h1>'
    page+=''.join(sections)+f'<p>Hardware: {html.escape(str(gpu))}</p>'
    for name in ['overall_metrics','per_object_success','per_first_skill_success']:
        path=f'media/{date}/{name}.png';page+=f'<p><a href="{path}" download>Download {name} PNG</a></p><img src="{path}">'
    page+='<p><a href="per_trial.csv">Per-trial CSV</a> · <a href="paired_outcomes.csv">Paired outcomes CSV</a> · <a href="summary.json">Full metrics JSON</a></p><h2>Interpretation limits</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in report['limits'])+'</ul>'
    (out/'index.html').write_text(page)
    print(json.dumps({'models':report['models'],'comparisons':report['comparisons'],'report':str(out)},indent=2))


if __name__=='__main__':
    main()
