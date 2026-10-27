"""Compare completed diffusion control/contact rollouts without new inference."""
import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['start_contact_to_part_cm', 'mean_contact_to_part_cm',
           'end_contact_to_part_cm', 'mean_tcp_to_part_cm',
           'end_tcp_to_part_cm', 'mean_action_to_part_cm',
           'phase_changes', 'gripper_clipped_count']


def key(row):
    return row['object'], str(row['task_type']), str(row['episode_id']), int(row['seed'])


def load_trials(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError(f'Duplicate scenario/seed: {identity}')
        if row['success'] not in ['True', 'False']:
            raise ValueError('Unexpected success value')
        row['success'] = row['success'] == 'True'
        for metric in METRICS:
            value = float(row[metric]) if row.get(metric) else None
            row[metric] = value if value is not None and math.isfinite(value) else None
        result[identity] = row
    return result


def metrics(rows):
    return {name: statistics.mean(values) if values else None
            for name in METRICS
            for values in [[row[name] for row in rows if row[name] is not None]]}


def write_control_only_review(root, pairs):
    rows = [row for row in pairs if row['outcome'] == 'control_only_success']
    grouped = {}
    for row in rows:
        grouped.setdefault((row['object'], row['task_type']), []).append(row)
    counts, sections = [], []
    for (obj, task), trials in sorted(grouped.items()):
        scenes = sorted({row['episode_id'] for row in trials})
        anchor = f'{obj}-task{task}'
        counts.append(f'<tr><td><a href="#{anchor}">{obj}</a></td><td>{task}</td>'
                      f'<td>{html.escape(trials[0]["first_skill"])}</td>'
                      f'<td>{len(trials)}</td><td>{len(scenes)}</td></tr>')
        sections.append(f'<h2 id="{anchor}">{obj} / task {task} — '
                        f'{len(trials)} rollouts, {len(scenes)} scenarios</h2>')
        for episode in scenes:
            repeats = [row for row in trials if row['episode_id'] == episode]
            first = repeats[0]
            buttons = ''.join(
                f'<button data-src="{html.escape(row["gif"], quote=True)}" '
                f'onclick="this.closest(\'article\').querySelector(\'img\').src=this.dataset.src;'
                f'this.closest(\'article\').querySelector(\'.selected\').textContent=this.textContent">'
                f'Seed {row["seed"]}</button> '
                f'<a href="{html.escape(row["gif"], quote=True)}" download>Download</a> '
                for row in repeats)
            sections.append(f'<article><h3>Episode {episode}</h3>'
                            f'<p>{html.escape(first["instruction"])}</p><p>{buttons}</p>'
                            f'<p class="selected">Seed {first["seed"]}</p>'
                            f'<img loading="lazy" src="{html.escape(first["gif"], quote=True)}" '
                            f'alt="Expert, MLP control, diffusion control, diffusion contact"></article>')
    with (root/'control_only_success.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(rows)
    page = ('<!doctype html><meta charset="utf-8"><title>Control succeeds, contact fails</title>'
            '<style>body{font:16px system-ui;max-width:1920px;margin:auto;padding:24px;background:#17191d;color:#eee}'
            'article{padding:16px;margin:18px 0;background:#24272d}img{max-width:100%}a{color:#9bd0ff}'
            'td,th{padding:8px 16px;text-align:left}button{padding:8px;margin:4px;cursor:pointer}</style>'
            '<h1>Diffusion control succeeds; diffusion contact fails</h1>'
            f'<p>{len(rows)} matched rollouts across {len({key(row)[:3] for row in rows})} unique scenarios. '
            'Grouped by object and task; seed buttons show every qualifying repeat.</p>'
            '<p>Panels, left to right: expert reference → MLP control → <strong>diffusion control</strong> '
            '→ <strong>diffusion contact</strong>. Selection compares the last two panels.</p>'
            '<p>Classification uses recorded task success, not a visual diagnosis of the failure cause. '
            'First skill describes the start of a task; mug task 8 also includes later movement and rotation.</p>'
            '<p><a href="control_only_success.csv">Filtered CSV</a> · '
            '<a href="contact_impact_review.html">All outcome groups</a></p>'
            '<table><thead><tr><th>Object</th><th>Task</th><th>First skill</th>'
            '<th>Rollouts</th><th>Unique scenarios</th></tr></thead><tbody>'
            + ''.join(counts) + '</tbody></table>' + ''.join(sections))
    (root/'control_only_success.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--control', default='diffusion_control')
    parser.add_argument('--contact', default='diffusion_contact')
    parser.add_argument('--data-root', type=Path, default=ROOT/'datasets/PartInstruct')
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    paths = {label: root/'_raw'/label/'contact_motion_report/per_trial.csv'
             for label in [args.control, args.contact]}
    control, contact = [load_trials(paths[label]) for label in [args.control, args.contact]]
    if set(control) != set(contact):
        raise ValueError('Control and contact scenario/seed keys differ')
    if len(control) != 447 or {k[3] for k in control} != {42, 43, 44}:
        raise ValueError('Expected149 scenarios at each of three fixed seeds')
    if len({k[:3] for k in control}) != 149:
        raise ValueError('Expected149 unique scenarios')
    meta = json.loads((args.data_root/'episodes_meta_test.json').read_text())
    scenarios = {(obj, str(task), str(row['ep_id'])): row
                 for obj, splits in meta.items()
                 for task, rows in splits.get('test1', {}).items() for row in rows}
    manifest = json.loads((root/'manifest.json').read_text())
    media = {(row['object'], str(row['task']), str(row['episode_id']), int(row['seed'])): row['gif']
             for row in manifest}
    groups = {'both_success': [], 'control_only_success': [],
              'contact_only_success': [], 'both_fail': []}
    pairs = []
    for identity in sorted(control):
        a, b = control[identity], contact[identity]
        category = ('both_success' if a['success'] and b['success'] else
                    'control_only_success' if a['success'] else
                    'contact_only_success' if b['success'] else 'both_fail')
        scenario = scenarios[identity[:3]]
        row = {'object': identity[0], 'task_type': identity[1],
               'episode_id': identity[2], 'seed': identity[3], 'outcome': category,
               'first_skill': scenario['chain_params'][0]['skill_name'],
               'instruction': scenario['task_instruction'],
               'control_success': a['success'], 'contact_success': b['success'],
               'gif': media[identity]}
        for metric in METRICS:
            row['control_'+metric], row['contact_'+metric] = a[metric], b[metric]
        pairs.append(row)
        groups[category].append(identity)
    summary = {
        'control': args.control, 'contact': args.contact, 'trials': len(pairs),
        'unique_scenarios': 149, 'evaluation_seeds': [42, 43, 44],
        'control_successes': sum(row['success'] for row in control.values()),
        'contact_successes': sum(row['success'] for row in contact.values()),
        'paired_outcomes': {name: len(keys) for name, keys in groups.items()},
        'per_object': {}, 'per_first_skill': {},
        'geometry': {args.control: metrics(list(control.values())),
                     args.contact: metrics(list(contact.values()))},
        'outcome_geometry': {},
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths.values()},
        'limits': [
            'Separately trained policies: this comparison is not a causal intervention on one checkpoint.',
            'Scenario/seed pairs match, but observations diverge during each rollout.',
            'Geometry is measured against the first task part tracked with object motion; it is not physical contact.',
            'Averages weight trials equally;447 rollouts are149 scenarios repeated at three evaluation seeds.',
            'No frozen-model contact-input ablation or additional policy training was run for this report.']}
    for field, target in [('object', 'per_object'), ('first_skill', 'per_first_skill')]:
        for value in sorted({row[field] for row in pairs}):
            rows = [row for row in pairs if row[field] == value]
            summary[target][value] = {'trials': len(rows),
                'control_successes': sum(row['control_success'] for row in rows),
                'contact_successes': sum(row['contact_success'] for row in rows),
                'paired_outcomes': {name: sum(row['outcome'] == name for row in rows) for name in groups}}
    for name, keys in groups.items():
        summary['outcome_geometry'][name] = {
            'unique_scenarios': len({key[:3] for key in keys}),
            args.control: metrics([control[key] for key in keys]),
            args.contact: metrics([contact[key] for key in keys])}
    reference = json.loads((root/'summary.json').read_text())
    assert summary['control_successes'] == reference[args.control]['successes']
    assert summary['contact_successes'] == reference[args.contact]['successes']
    (root/'contact_impact_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (root/'contact_impact_per_trial.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    write_control_only_review(root, pairs)
    cards = []
    for name in ['control_only_success', 'contact_only_success', 'both_success']:
        rows = [row for row in pairs if row['outcome'] == name]
        cards.append(f'<h2>{html.escape(name.replace("_", " "))}: {len(rows)} rollouts</h2>')
        # Show one seed per scenario; every replicate remains in the main gallery/CSV.
        seen = set()
        for row in rows:
            scenario = row['object'], row['task_type'], row['episode_id']
            if scenario in seen:
                continue
            seen.add(scenario)
            cards.append(f'<article><h3>{html.escape(row["object"])} / task {row["task_type"]} / '
                         f'episode {row["episode_id"]} / seed {row["seed"]}</h3>'
                         f'<p>{html.escape(row["instruction"])}</p>'
                         f'<img loading="lazy" src="{row["gif"]}" alt="Matched policy comparison">'
                         f'<p><a href="{row["gif"]}" download>Download existing GIF</a></p></article>')
    page = ('<!doctype html><meta charset="utf-8"><title>Diffusion contact impact</title>'
            '<style>body{font:16px system-ui;max-width:1920px;margin:auto;padding:24px;background:#17191d;color:#eee}'
            'article{padding:16px;margin:18px 0;background:#24272d}img{max-width:100%}a{color:#9bd0ff}</style>'
            '<h1>Diffusion contact impact</h1>'
            '<p>Panels: expert reference → MLP control → diffusion control → diffusion contact.</p>'
            '<p>One example per scenario within each outcome group; all scenario/seed rows are in the CSV. '
            'These are separately trained models, not a frozen-policy intervention.</p>'
            '<p><a href="contact_impact_summary.json">Summary</a> · '
            '<a href="control_only_success.html">Control succeeds / contact fails: grouped review</a> · '
            '<a href="contact_impact_per_trial.csv">All paired rows</a> · '
            '<a href="index.html">Full447-GIF gallery</a></p>'+''.join(cards))
    (root/'contact_impact_review.html').write_text(page)
    print(json.dumps({key: summary[key] for key in ['paired_outcomes', 'per_object', 'per_first_skill', 'geometry']}, indent=2))


if __name__ == '__main__':
    main()
