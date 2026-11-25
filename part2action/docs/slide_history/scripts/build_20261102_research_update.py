"""Build the November 2 meeting deck from a timestamped evidence snapshot."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_slide_tables import render


def snapshot():
    sources={}
    def read(relative):
        p=ROOT/relative;blob=p.read_bytes()
        sources[relative]={'sha256':hashlib.sha256(blob).hexdigest()}
        return json.loads(blob)
    evaluation='results/part2action/evaluations/'
    policy={
        'decoder':read(evaluation+'diffusion_screening_10epoch_next_action_20261018/quantitative_report/summary.json'),
        'stride':read(evaluation+'diffusion_transformer_stride_comparison_20261021/summary.json'),
        'target_input':read(evaluation+'diffusion_transformer_contact_comparison_20261027_recovery1/quantitative_report/summary.json'),
        'gate':read(evaluation+'frozen_gate_diagnostics_20261022/quantitative_report/summary.json'),
        'target_interventions':read(evaluation+'frozen_target_diagnostics_20261027_recovery1/quantitative_report/summary.json')}
    released={}
    for model in ['DP-S','DP3-S']:
        released[model]={
            'native':read(f'results/partinstruct/evaluations/official_release_20261022/{model}/full/summary.json')}
    baseline_root='results/diagnostics/full_trajectory_baselines_20261102/'
    plan=read(baseline_root+'full/plan.json')
    preflight=read(baseline_root+'preflight_v2/summary.json')
    counts={obj:{'planned':sum(c['object']==obj for c in plan['cases']),
                 'finished':0,'eligible':0,'runtime_errors':0}
            for obj in ['bottle','mug','pliers','scissors']}
    signatures=[];indices=[]
    # The running collector only writes its final aggregate after completion;
    # derive this presentation snapshot from atomically written case records.
    for p in sorted((ROOT/baseline_root/'full/cases').glob('*/result.json')):
        blob=p.read_bytes();r=json.loads(blob);obj=r['case']['object']
        counts[obj]['finished']+=1;counts[obj]['eligible']+=int(r['eligible'])
        counts[obj]['runtime_errors']+=int(r['status']=='runtime_error')
        indices.append(r['case_index'])
        signatures.append([r['case_index'],hashlib.sha256(blob).hexdigest()])
    assert len(indices)==len(set(indices))
    original=read('results/diagnostics/full_dataset_trajectory_replay_20261102/mug_demo_1/summary.json')
    return {'captured_at':datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds'),
            'policy':policy,'released':released,
            'baselines':{'by_object':counts,'finished':len(indices),
                         'planned':len(plan['cases']),'eligible':sum(v['eligible'] for v in counts.values()),
                         'runtime_errors':sum(v['runtime_errors'] for v in counts.values()),
                         'status':'completed' if len(indices)==len(plan['cases']) else 'in_progress',
                         'case_result_sha256':signatures,'preflight':preflight,'first_full_replay':original,
                         'limits':'Incomplete coverage follows inventory order; not an unbiased success estimate. '
                                  'Two same-seed repeats screen reproducibility, not multi-seed robustness.'},
            'gpc':{'title':'Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling',
                   'paper':'https://arxiv.org/abs/2502.00622','version':'2502.00622v4',
                   'venue':'IEEE Robotics and Automation Letters; accepted February 28, 2026',
                   'project':'https://computationalrobotics.seas.harvard.edu/GPC/'},
            'source_files':sources,
            'new_method_status':'Full-trajectory baseline collection implemented. New perturbation collection, '
                                'full-task critic training, structured tolerance representation and policy-guided selection remain proposed.',
            'historical_pilot':'archive_grasp_lift_pilot/slides.md; its 107 labels use different grasp/lift criteria and are not mixed into the new task protocol.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'part2action/docs/slide_history/media/2026-11-02/robustness_pilot')
    parser.add_argument('--slides',type=Path,default=ROOT/'part2action/docs/slide_history/slides/2026-11-02_robustness_training_update.md')
    parser.add_argument('--use-snapshot',action='store_true',help='Regenerate the exact saved report without refreshing running-job counts')
    # Preserve the old launcher interface; legacy grasp labels are not used here.
    for argument in ['--study','--training','--native-context','--proposals']:
        parser.add_argument(argument,help=argparse.SUPPRESS)
    parser.add_argument('--probes',nargs='+',help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    evidence_path=out/'evidence.json'
    evidence=json.loads(evidence_path.read_text()) if args.use_snapshot else snapshot()
    evidence_path.write_text(json.dumps(evidence,indent=2)+'\n')
    def link_to(path):
        return Path(os.path.relpath(path, args.slides.parent)).as_posix()
    tables=[]
    def table(slide,title,rows,fractions):
        file=f'slide{slide:02d}_{title}.png'
        size=render(rows,fractions,out/file)
        tables.append({'slide':slide,'file':file,'rows':rows,'column_fractions':fractions,
                       'dimensions':size,'sha256':hashlib.sha256((out/file).read_bytes()).hexdigest()})
        target=link_to(out/file)
        return f'![Slide {slide} table]({target})\n\nUse this image: [{file}]({target}).\n'
    policy=evidence['policy']
    def fraction(row):return f'{row["successes"]}/{row["trials"]} ({100*row["successes"]/row["trials"]:.1f}%)'
    decoder=policy['decoder']['models'];stride=policy['stride'];target=policy['target_input']['models']
    gate=policy['gate']['models'];intervention=policy['target_interventions']['models']
    t1=table(1,'recovered_policy_results',[
        ['Separately trained model / checkpoint','Decoder / sample stride','Explicit target input to action','Success'],
        ['A: Earlier diffusion (6724877)','MLP denoiser; stride 1','OFF: zero relative geometry',fraction(decoder['baseline'])],
        ['B: Transformer control (6724878)','Transformer; stride 1','OFF: zero relative geometry',fraction(target['contact_off'])],
        ['C: Transformer stride test (6731746)','Transformer; stride 2','OFF: zero relative geometry',fraction(stride['Transformer stride 2'])],
        ['D: Transformer target-input model (6744368)','Transformer; stride 1','ON: predicted target relative to TCP',fraction(target['contact_on'])]], [.32,.25,.25,.18])
    t1_gate=table(1,'gate_diagnostics_detailed',[
        ['Gate diagnostic variant','Weights / training','Visual input at evaluation','Target geometry input','Success'],
        ['Predicted gate (control)','Reuse B: 6724878; no retraining','Model-predicted part emphasis','OFF',fraction(gate['predicted'])],
        ['Oracle gate','Reuse B: 6724878; no retraining','GT-part projection at the learned phase; phase remains predicted','OFF',fraction(gate['oracle'])],
        ['No visual gate','Reuse B: 6724878; no retraining','Identity visual weighting; no part emphasis','OFF',fraction(gate['no_gate'])]], [.19,.24,.25,.16,.16])
    t1_target=table(1,'target_diagnostics_detailed',[
        ['Target diagnostic variant','Weights / training','Geometry fed into action features','Visual gate','Success'],
        ['Identity (control)','Reuse D: 6744368; no retraining','Unmodified predicted target relative to measured TCP','Predicted',fraction(intervention['identity'])],
        ['Zero vector','Reuse D: 6744368; no retraining','Zero the relative geometry; keep ON-trained projection, bias and weights','Predicted',fraction(intervention['zero_input'])],
        ['+5 cm world-X shift','Reuse D: 6744368; no retraining','Shift the target used for conditioning +0.05 m along world X','Predicted',fraction(intervention['shift_x_5cm'])]], [.18,.22,.30,.14,.16])
    released=evidence['released']
    def subset(row):
        groups=[g for g in row['groups'] if any('/'+obj+'/' in g['path'] for obj in ['bottle','mug','pliers','scissors'])]
        return {'successes':sum(g['successes'] for g in groups),'trials':sum(g['trials'] for g in groups)}
    t2=table(2,'released_dp_dp3_results',[
        ['Earlier native evaluation','11 categories','Four-category subset','Status'],
        ['DP-S',fraction(released['DP-S']['native']),fraction(subset(released['DP-S']['native'])),'COMPLETED'],
        ['DP3-S',fraction(released['DP3-S']['native']),fraction(subset(released['DP3-S']['native'])),'COMPLETED']], [.27,.25,.26,.22])
    t3=table(3,'gpc_method',[
        ['GPC component','Role'],
        ['Frozen diffusion policy','Propose multiple plausible action chunks from current observations'],
        ['Action-conditioned world model','Predict future outcomes; learn from demonstrations and random exploration'],
        ['GPC-RANK','Score predicted outcomes with a reward function; select the best candidate'],
        ['GPC-OPT','Refine a proposal through world-model gradients; can combine with ranking']], [.30,.70])
    t4=table(4,'proposed_tolerance_architecture',[
        ['Component','Inputs','Output / purpose'],
        ['Existing perception','RGB / scene PCD, language, robot state and predicted part features','Observation and task features'],
        ['Existing action policy','Current observation history and task','Candidate action chunks'],
        ['New outcome scorer','Observation features + candidate action sequence','Task-success score under a declared continuation'],
        ['Later tolerance representation','Joint position, wrist rotation and closing-time changes','State- and part-conditioned acceptable interaction set'],
        ['Later candidate selection','Same candidate budget, scored alternatives','Execute chosen action, observe again']], [.25,.39,.36])
    baseline=evidence['baselines'];preflight=baseline['preflight']
    rows=[['Source category','Finished / planned','Accepted baselines','Runtime errors']]
    for obj,c in baseline['by_object'].items():rows.append([obj,f'{c["finished"]}/{c["planned"]}',str(c['eligible']),str(c['runtime_errors'])])
    rows.append(['TOTAL',f'{baseline["finished"]}/{baseline["planned"]}',str(baseline['eligible']),str(baseline['runtime_errors'])])
    t5=table(5,'implemented_baseline_collection',rows,[.25,.27,.28,.20])
    t6=table(6,'remaining_implementation',[
        ['Step','Implementation to add','Validation'],
        ['Collect variations','Preserve baseline prefix; change one factor, then joint factors; run the original task','Store actual pre-action RGB / PCD, candidate, continuation and outcome'],
        ['Train generic scorer','Freeze encoders; train candidate-sequence scores with BCE on success/failure labels','Split by object instance; report failure recall, calibration and candidate ranking'],
        ['Learn structured tolerance','Compare spatial-only and joint pose / closing-time representations','Compare against the generic scorer using identical records'],
        ['Connect to the actor','Rank a fixed number of policy proposals; execute and reobserve','Validate with policy continuation; compare rollout success and latency'],
        ['Matched controls','Policy only; successful-data augmentation; generic scorer; structured tolerance','Same source data, candidate count, evaluation cases and control budget']], [.23,.39,.38])
    t7=table(7,'gpc_adopted_and_different',[
        ['Aspect','GPC','Our proposed design'],
        ['Adopt: modular policy','Keep a trained diffusion actor fixed','Initially preserve our current actor and encoders'],
        ['Adopt: candidate evaluation','Evaluate proposals before execution','Score alternatives before selecting an action'],
        ['Adapt: additional data','Demonstrations plus random exploration','Successful full replays plus controlled local changes'],
        ['Change: prediction target','Future states / images, followed by reward evaluation','Direct outcome scores; later an explicit joint pose-time tolerance representation'],
        ['Evidence so far','Simulation and real-robot policy improvement','Reference collection only; no new closed-loop robustness gain established']], [.23,.385,.385])
    captured=evidence['captured_at']
    def result(relative,label):return f'[{label}]({link_to(ROOT/relative)})'
    def image_sources(links):return '\nSources: '+' · '.join(links)+'.\n'
    paper=link_to(out/'Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling.pdf')
    slides=(
        '# Part2Action research update — November 2, 2026\n\n'
        'Date: 2026-11-02\n\n'
        'Seven slides; short presentation text followed by PNG tables; no speaker notes.\n'
        f'Evidence snapshot: {captured} (America/New_York). Running-job counts below are provisional.\n\n'
        '## Slide 1 — Part2Action: Recovered policy results\n\n'
        'All conditions use 447 fixed rollouts (149 scenarios x 3 seeds); ten-epoch models use one training seed.\n'
        'A/B compare decoders; B/C compare sampling stride; B/D compare explicit target input using separate training runs.\n'
        'Target input OFF/ON controls predicted target-to-TCP geometry entering the action features.\n'
        'Target-head auxiliary loss and the predicted visual part gate stay enabled in all four training runs.\n'
        'Frozen means reuse the same checkpoint without training; predictions and rollout states can still change.\n'
        'Gate diagnostics reuse B in three conditions; target diagnostics reuse D in three conditions.\n'
        'D scores 63 in standard evaluation and 66 in the separate identity run with different DDPM-noise handling.\n\n'+t1+'\n'+t1_gate+'\n'+t1_target+
        image_sources([result('results/part2action/evaluations/diffusion_transformer_contact_comparison_20261027_recovery1/quantitative_report/summary.json','Recovered OFF/ON'),
                       result('results/part2action/evaluations/frozen_gate_diagnostics_20261022/quantitative_report/summary.json','Gate diagnostics'),
                       result('results/part2action/evaluations/frozen_target_diagnostics_20261027_recovery1/quantitative_report/summary.json','Target diagnostics')])+'\n'
        '## Slide 2 — Part2Action: Released DP and DP3 references\n\n'
        'Earlier native evaluations are complete for both DP-S and DP3-S: 176 trials per model.\n'
        'Their 64-trial four-category figures are subsets of those completed runs.\n'+
        'Superseded fixed 120/1 reference outputs were removed on November 16 at the user request.\n'
        'Retained native protocol: 250 actions, eight-action execution blocks.\n'
        'These models receive GT parts, oracle skill progression and sequential-skill scoring.\n'
        'The native sampling differs from our full fixed-scenario population.\n\n'+t2+
        image_sources([result('results/partinstruct/evaluations/official_release_20261022/DP-S/full/summary.json','Native DP'),
                       result('results/partinstruct/evaluations/official_release_20261022/DP3-S/full/summary.json','Native DP3')])+'\n'
        '## Slide 3 — Part2Action: GPC evaluates actions before execution\n\n'
        'GPC: Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling.\n'
        'IEEE Robotics and Automation Letters; accepted February 2026.\n'
        'Keep the diffusion actor frozen and train an action-conditioned predictive world model separately.\n'
        'Use predicted futures to rank or refine candidate actions at inference time.\n'
        'GPC shows policy improvement in simulation and on real robots.\n'
        'It motivates modular action evaluation; it does not validate our tolerance representation.\n\n'+t3+
        f'\nSources: [Uploaded paper](<{paper}>) · [Official project](https://computationalrobotics.seas.harvard.edu/GPC/) · [Paper v4](https://arxiv.org/html/2502.00622v4).\n\n'
        '## Slide 4 — Part2Action: Learn acceptable interaction alternatives\n\n'
        'Proposed: learn which position, wrist-orientation and closing-time changes preserve task success.\n'
        'Condition on the observed scene, target part, task and current robot state.\n'
        'Keep the existing Transformer diffusion policy as the initial proposal generator.\n'
        'Start with a generic candidate-success scorer; structured tolerance is the hypothesis to test.\n'
        'Score candidates from actual pre-action observations; exclude post-action outcomes from inputs.\n'
        'Do not require future expert poses or expert closing times at deployment.\n'
        'The full-task scorer and tolerance-guided policy have not been trained yet.\n\n'+t4+'\n'
        '## Slide 5 — Part2Action: Implemented reproducible baseline collection\n\n'
        'Replay every stored action once with native controller defaults and full task context.\n'
        'Accept task-successful executions that repeat reliably at the same seed; HDF5 path matching is not a gate.\n'
        f'Technical preflight: five source pairs reproduced; {preflight["eligible"]}/5 became successful baselines, with no runtime errors.\n'
        'Screen all 4,053 valid demonstrations in our four training categories; two entries lack actions.\n'
        'Native first-success events define task completion; add no lift or final-frame contact-hold test.\n'
        'The table is a collection snapshot in inventory order, not a policy benchmark or multi-seed robustness result.\n\n'+t5+
        image_sources([result('results/diagnostics/full_trajectory_baselines_20261102/current_study.json','Study ledger'),
                       result('results/diagnostics/full_trajectory_baselines_20261102/preflight_v2/summary.json','Preflight'),
                       result('results/diagnostics/full_dataset_trajectory_replay_20261102/mug_demo_1/README.md','First full replay and video')])+'\n'
        '## Slide 6 — Part2Action: Next implementation and validation\n\n'
        'Collect supported perturbations around accepted full-task baselines before new model training.\n'
        'Start with position/orientation; apply closing-time changes only to meaningful grasp windows.\n'
        'Keep controller and continuation fixed within each comparison, then test joint changes.\n'
        'Train the scorer on both outcomes; use successful variations only for the behavior-cloning control.\n'
        'Hold out whole object instances and keep every source trajectory with all of its variations.\n'
        'Validate under policy continuation before claiming an improvement in policy-controlled rollouts.\n'
        'The old 107-record grasp/lift pilot uses different labels and stays separate.\n\n'+t6+'\n'
        '## Slide 7 — Part2Action: What we adopt and what must differ\n\n'
        'Adopt the separation between action generation, additional outcome data and candidate evaluation.\n'
        'Propose controlled local variation data and an explicit joint pose-time tolerance representation.\n'
        'GPC already conditions on current observations; state conditioning alone is not a novelty claim.\n'
        'Test whether part-aware structure improves held-out control beyond generic scoring and data augmentation.\n'
        'Match training data, candidate budgets and rollout conditions; report success and latency.\n'
        'Lower cost, improved generalization and closed-loop robustness remain unverified hypotheses.\n\n'+t7+
        f'\nEvidence: [Snapshot]({link_to(evidence_path)}) · '
        f'[Table values]({link_to(out/"tables.json")}).\n')
    args.slides.write_text(slides)
    (out/'tables.json').write_text(json.dumps(tables,indent=2)+'\n')
    mapping={}
    for t in tables:mapping.setdefault(str(t['slide']),[]).append(t['file'])
    (out/'manifest.json').write_text(json.dumps({'meeting_date':'2026-11-02','captured_at':captured,
        'slide_mapping':mapping,
        'generator':str(Path(__file__).relative_to(ROOT)),
        'format':'Markdown and numbered PNG tables only; no generated PDF/HTML/ZIP report.'},indent=2)+'\n')
    print(json.dumps({'slides':str(args.slides),'tables':len(tables),'snapshot':captured,
                      'baseline_finished':baseline['finished'],'baseline_eligible':baseline['eligible'],
                      'fixed_dp':released['DP-S']['fixed']['trials'],'fixed_dp3':released['DP3-S']['fixed']['trials']}))


if __name__=='__main__':main()
