"""Assemble the six selected October 12 slide GIFs from existing recordings.

No training, evaluation rollouts, or Slurm jobs are launched. Existing table
metadata is retained when rebuilding the preview and ZIP.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import argparse
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'part2action/scripts'))
sys.path.insert(0, str(ROOT/'part2action'))
from media_bundle import write_bundle

OUT = ROOT/'part2action/docs/slide_history/media/2026-10-12'
WORK = None
references = None
records = []
previous = {}
rerender = False


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def properties(path):
    with Image.open(path) as gif:
        duration = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
            duration += gif.info.get('duration', 0)
        if gif.n_frames < 2 or gif.info.get('loop') != 0:
            raise ValueError(f'Not a looping animation: {path}')
        return {'dimensions': list(gif.size), 'gif_frames': gif.n_frames,
                'duration_seconds': duration/1000, 'bytes': path.stat().st_size,
                'sha256': digest(path)}


def add_copy(slides, source, name, title, description, panels, provenance=None):
    source = ROOT/source
    target = OUT/name
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f'Refusing to overwrite a different existing asset: {target}')
    else:
        shutil.copyfile(source, target)
    if digest(source) != digest(target):
        raise ValueError(f'Copy differs from source: {source}')
    record = {'slides': slides, 'file': name, 'title': title, 'description': description,
              'panels': panels, 'source': str(source.relative_to(ROOT)), 'kind': 'unchanged_copy',
              **properties(target)}
    if provenance:
        record['provenance'] = provenance
    records.append(record)
    print(f'Copied {name}', flush=True)


def add_comparison(slides, sources, name, title, description):
    global references
    cached = previous.get(name)
    if (OUT/name).is_file() and not rerender:
        if not cached or cached.get('source_result_json') != sources:
            raise ValueError(f'No matching provenance for existing GIF: {name}; use --rerender-comparisons')
        if digest(OUT/name) != cached['sha256']:
            raise ValueError(f'Existing GIF differs from provenance: {name}')
        records.append(cached)
        print(f'Reused {name}', flush=True)
        return
    from export_evaluation_gifs import load_sources, encode_gif
    from build_policy_comparison_videos import render_panels, key
    from ground_truth_reference import GroundTruthReferences
    if references is None:
        references = GroundTruthReferences(WORK/'ground_truth', ROOT/'datasets/PartInstruct')
    data, payloads = load_sources(sources)
    labels = list(data)
    selected = [trial for trial in data[labels[0]].values()
                if trial['obj_class'] == 'bottle' and str(trial['task_type']) == '1'
                and trial['trial_index'] == 0]
    if len(selected) != 1:
        raise ValueError('Expected exactly one bottle grasp trial 0')
    trial_key = key(selected[0])
    trials = [data[label][trial_key] for label in labels]
    truth = references.trial(trials[0])
    panel_labels = ['Ground-truth reference']+labels
    panels = [truth]+trials
    video = WORK/Path(name).with_suffix('.mp4')
    print(f'Rendering {name}', flush=True)
    rendered = render_panels(panels, panel_labels, video, speed=.5, show_action_state=True)
    encoded = encode_gif(video, OUT/name, fps=5, panels=len(panels))
    records.append({'slides': slides, 'file': name, 'title': title, 'description': description,
                    'kind': 'comparison_from_existing_recordings', 'panels': panel_labels,
                    'trial_key': list(trial_key), 'instruction': trials[0]['instruction'],
                    'policy_successes': {label: bool(t['success']) for label,t in zip(labels,trials)},
                    'source_result_json': sources, 'source_videos': [t['rollout_video'] for t in panels],
                    'checkpoints': {label: payloads[label].get('ckpt') for label in labels},
                    'ground_truth_reference': truth['ground_truth_reference'],
                    'playback_speed': .5, 'fps': 5, **rendered, **encoded, **properties(OUT/name)})
    print(f'Finished {name}', flush=True)
def main():
    global OUT, WORK, references, records, previous, rerender
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    parser.add_argument('--rerender-comparisons', action='store_true',
                        help='Render the two selected comparisons again from their source recordings.')
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rerender = args.rerender_comparisons
    old_manifest = json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').is_file() else {}
    previous = {row['file']: row for row in old_manifest.get('assets', [])}
    records = []
    references = None
    with tempfile.TemporaryDirectory(prefix='p2a_slide_media_20261012_') as temporary:
        WORK = Path(temporary)
        add_copy([1], Path('assets/part2action/part2action_bottle_neck.gif'),
                 'slide01_bottle_neck_contact_demo.gif', 'Slide 1 — Project illustration',
                 'Expert demonstration, not a learned policy. Cyan: derived contact target; yellow: recorded TCP/path; green: dataset target part.',
                 ['Expert demonstration'], json.loads((ROOT/'assets/part2action/part2action_bottle_neck.json').read_text()))

        norm = ROOT/'results/part2action/evaluations/normalized_global_20261010'
        norm_manifest = json.loads((norm/'manifest.json').read_text())
        for slide, filename, name, title, desc in [
            (4, 'mug_task1_trial00_episode300083.gif', 'slide05_normalized_mug_grasp.gif',
             'Slide 4 — Normalized control versus contact',
             'Mug grasp: control succeeds and contact fails in this trial. This individual example illustrates the 4/39 versus 0/39 comparison.'),
            (4, 'mug_task2_trial02_episode300091.gif', 'slide06_stalled_mug_touch.gif',
             'Slide 4 — Stalled touch motion',
             'Mug touch, not grasp. Both policies remain near their starting pose. This does not establish gripper closure as the cause.'),
        ]:
            source_entry = next(entry for entry in norm_manifest if entry['gif'] == filename)
            add_copy([slide], Path('results/part2action/evaluations/normalized_global_20261010')/filename,
                     name, title, desc+' Expert is a task-matched demonstration from a different episode.',
                     source_entry.get('labels', ['Ground-truth reference', 'Unified control', 'Unified contact']), source_entry)

        add_copy([5], Path('results/diagnostics/observation_action_timing_20261010/mug_demo_1544_timing.gif'),
                 'slide07_observation_action_timing.gif', 'Slide 5 — Training-label timing',
                 'All three panels show the same expert RGB. The labels compare current-action and next-action targets; these are not predicted-policy rollouts.',
                 ['Recorded expert frame', 'Same-index label', 'Next-action candidate'])

        timing_root = ROOT/'results/part2action/evaluations/action_timing_20261011'
        timing_filename = 'mug_task3_trial00_seed642_episode300069.gif'
        timing_entry = next(row for row in json.loads((timing_root/'manifest.json').read_text())
                            if row['gif'] == timing_filename)
        add_copy([6], Path('results/part2action/evaluations/action_timing_20261011')/timing_filename,
                 'slide06_next_action_mug.gif', 'Slide 6 — Same-index versus next-action pilot',
                 'Same-index fails and next-action succeeds in this example. Both are contact-disabled MLP continuations; expert is a task-matched reference.',
                 timing_entry['labels'], timing_entry)

        filename = 'rollout_results_test1_bottle-mug-pliers-scissors_1-2-3-4_legacy.json'
        original = ROOT/'results/diagnostics/contact_motion_20261008'
        add_comparison([2], [f'Original control={original/"unified_control"/"hierarchical"/filename}',
                            f'Original contact={original/"unified_contact"/"hierarchical"/filename}'],
                       'slide02_original_contact_localization.gif', 'Slide 2 — Original contact predictions',
                       'Original 30-epoch policies with blue predicted contacts and green nearest true-part points. One illustrative trial; the slide table summarizes 39 initial observations.')

        intervention = ROOT/'results/diagnostics/contact_intervention_20261008'
        add_comparison([3], [f'Contact 100%={intervention/"normal"/"hierarchical"/filename}',
                            f'Contact 25%={intervention/"reduced"/"hierarchical"/filename}',
                            f'Contact 0%={intervention/"zero"/"hierarchical"/filename}'],
                       'slide03_contact_strength_100_25_0.gif', 'Slide 3 — Contact input strength',
                       'Same trained checkpoint with three contact-input strengths. All three policies fail this bottle trial; approach behavior differs. The slide table reports all 39 trials.')

        records.sort(key=lambda row: (min(row['slides']), row['file']))
        manifest = {
            'meeting_date': '2026-10-12', 'assets': records,
            'slide_mapping': {str(i): [row['file'] for row in records if i in row['slides']] for i in range(1,8)},
            'reference_note': 'Expert reference is leftmost in model comparisons. Task-matched references may have different source episodes and initial scenes. Playback follows nominal elapsed sample index, not task phase. Earlier finishes hold their final frame.',
            'unavailable': {'7': 'Expanded evaluation and diffusion outcomes are not yet exported in this presentation bundle.'},
        }
        manifest['table_images'] = old_manifest.get('table_images', [])
        manifest['diagram_images'] = old_manifest.get('diagram_images', [])
        for table in manifest['table_images']:
            for slide in table['slides']:
                manifest['slide_mapping'][str(slide)].append(table['file'])
        for diagram in manifest['diagram_images']:
            for slide in diagram['slides']:
                manifest['slide_mapping'][str(slide)].extend([diagram['file'], diagram['svg']])
        if manifest['diagram_images']:
            manifest['unavailable'].pop('4', None)
        manifest['generators'] = {
            'gifs': str(Path(__file__).resolve().relative_to(ROOT)),
            'tables': 'part2action/docs/slide_history/scripts/export_tables_20261012.py',
            'project_demo': 'part2action/docs/slide_history/scripts/export_project_gif.py',
            'bundle': 'part2action/docs/slide_history/scripts/media_bundle.py',
        }
        if manifest['diagram_images']:
            manifest['generators']['architecture'] = manifest['diagram_images'][0]['generator']
        print(json.dumps(write_bundle(OUT, manifest), indent=2))


if __name__ == '__main__':
    main()
