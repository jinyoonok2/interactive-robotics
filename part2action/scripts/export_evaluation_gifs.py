"""Export matched evaluations as flat, labeled, side-by-side GIFs."""
import argparse
import html
import json
import subprocess
import zipfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from build_policy_comparison_videos import key, render_panels
from ground_truth_reference import DEFAULT_DATA_ROOT, GroundTruthReferences
from diagnostics.evaluation_scenarios import success_statistics


def load_sources(sources):
    data, payloads = {}, {}
    for source in sources:
        label, separator, directory = source.partition('=')
        if not separator or not label or label in data:
            raise ValueError('Each source must be a unique LABEL=DIRECTORY')
        path = Path(directory).resolve(strict=True)
        files = [path] if path.is_file() else sorted(path.rglob('rollout_results_*.json'))
        if len(files) != 1:
            raise ValueError(f'{label}: expected exactly one rollout result JSON, found {len(files)}')
        payload = json.loads(files[0].read_text())
        trials = payload['results']
        keyed = {key(t): t for t in trials}
        if not keyed or len(keyed) != len(trials):
            raise ValueError(f'{label}: empty results or duplicate trial keys')
        # Relative media references are resolved against the result JSON.
        for trial in trials:
            video = Path(trial['rollout_video'])
            if not video.is_absolute():
                trial['rollout_video'] = str((files[0].parent / video).resolve())
            if not Path(trial['rollout_video']).is_file():
                raise FileNotFoundError(trial['rollout_video'])
        data[label], payloads[label] = keyed, payload
    reference = next(iter(data.values()))
    reference_payload = next(iter(payloads.values()))
    for label, trials in data.items():
        if set(trials) != set(reference):
            raise ValueError(f'{label}: episode, task, trial and seed keys do not match')
        for field in ['split', 'termination_mode', 'gripper_command_bounds_m', 'point_cloud_sampling_device']:
            if payloads[label].get(field) != reference_payload.get(field):
                raise ValueError(f'{label}: mismatched evaluation protocol field {field}')
        for k, trial in trials.items():
            if trial['instruction'] != reference[k]['instruction']:
                raise ValueError(f'{label}: mismatched instruction for {k}')
            if trial.get('diagnostics') and reference[k].get('diagnostics'):
                if not np.allclose(trial['diagnostics'][0]['tcp_xyz'], reference[k]['diagnostics'][0]['tcp_xyz']):
                    raise ValueError(f'{label}: mismatched initial TCP position for {k}')
    return data, payloads


def encode_gif(video, output, fps, panels):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(video),
                    '-filter_complex_threads', '1', '-filter_complex',
                    f'fps={fps:g},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer',
                    '-threads', '2', '-loop', '0', str(output)], check=True)
    with Image.open(output) as gif:
        if gif.size != (480 * panels, 720) or gif.n_frames < 2:
            raise ValueError(f'Invalid animated comparison: {output}')
        return {'gif_frames': gif.n_frames, 'size': list(gif.size)}


def write_bundle(root):
    """The portable review contains GIFs and reports, excluding raw recordings."""
    temporary = root / '.comparison_bundle.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(root.iterdir()):
            if file.is_file() and file.suffix in {'.gif', '.html', '.json', '.csv'}:
                bundle.write(file, file.name)
    temporary.replace(root / 'comparison_bundle.zip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, help='LABEL=DIRECTORY or LABEL=result.json; repeat in panel order')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT, help='PartInstruct data containing expert HDF5 demonstrations')
    parser.add_argument('--speed', type=float, default=.5)
    parser.add_argument('--fps', type=float, default=5)
    parser.add_argument('--limit', type=int, help='Export only the first N matched trials for a rendering smoke test')
    parser.add_argument('--reuse-media', action='store_true')
    args = parser.parse_args()
    if args.speed <= 0 or args.fps <= 0 or (args.limit is not None and args.limit < 1):
        parser.error('speed, fps and limit must be positive')
    data, payloads = load_sources(args.source)
    labels = list(data)
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous={entry['gif']:entry for entry in json.loads((root/'manifest.json').read_text())} if args.reuse_media and (root/'manifest.json').is_file() else {}
    raw = root / '_raw' / 'composites'
    raw.mkdir(parents=True, exist_ok=True)
    references = GroundTruthReferences(root/'_raw'/'ground_truth', args.data_root)
    episodes = sorted(next(iter(data.values())))[:args.limit]
    manifest, cards = [], []
    for k in episodes:
        trials = [data[label][k] for label in labels]
        truth = references.trial(trials[0])
        panels = [truth] + trials
        panel_labels = ['Ground-truth reference'] + labels
        obj, task, trial, seed, ep = k
        stem = f'{obj}_task{task}_trial{trial:02d}_seed{seed}_episode{ep}'
        video, gif = raw / f'{stem}.mp4', root / f'{stem}.gif'
        if args.reuse_media:
            if not video.is_file() or not gif.is_file():
                raise FileNotFoundError(f'Cannot reuse missing media: {stem}')
            entry=previous.get(gif.name,{})
            if entry.get('labels')!=panel_labels or entry.get('ground_truth_reference')!=truth['ground_truth_reference']:
                raise ValueError('Existing media has a different panel order or expert reference; re-export without --reuse-media')
            with Image.open(gif) as reused:
                if reused.size != (480*len(panels),720):
                    raise ValueError('Existing GIF lacks the leftmost ground-truth panel; re-export without --reuse-media')
            info={field:entry[field] for field in ['frames','input_fps','output_fps','input_frame_counts','gif_frames','size'] if field in entry}
        else:
            info = render_panels(panels, panel_labels, video, args.speed, show_action_state=True)
            info.update(encode_gif(video, gif, args.fps, len(panels)))
        outcomes = {label: bool(t['success']) for label, t in zip(labels, trials)}
        manifest.append({'object': obj, 'task': task, 'trial': trial, 'seed': seed, 'episode_id': ep,
                         'labels': panel_labels, 'success': outcomes, 'gif': gif.name,
                         'ground_truth_reference': truth['ground_truth_reference'],
                         'video': str(video.relative_to(root)), 'source_videos': [t['rollout_video'] for t in panels], **info})
        caption = ' | '.join(f'{label}: {"success" if ok else "failure"}' for label, ok in outcomes.items())
        reference=truth['ground_truth_reference']
        reference_caption=f'Ground-truth reference: {reference["demo_key"]}, source episode {reference["episode_id"]}; {reference["match_type"]}.'
        cards.append(f'<article><h2>{html.escape(stem)}</h2><p>{html.escape(trials[0]["instruction"])}</p><p>{html.escape(reference_caption)}</p>'
                     f'<p>{html.escape(caption)}</p><img loading="lazy" src="{gif.name}" alt="Matched policy comparison">'
                     f'<p><a href="{gif.name}" download>Download GIF</a></p></article>')
        print(f'[{len(manifest)}/{len(episodes)}] {gif.name}', flush=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = {label: {'trials': len(episodes), 'successes': sum(data[label][k]['success'] for k in episodes),
                       'checkpoint': payloads[label].get('ckpt'),
                       **success_statistics([data[label][k] for k in episodes])} for label in labels}
    for label in labels:
        timings = [value for k in episodes for value in data[label][k].get('policy_forward_latency_ms', [])[1:]]
        summary[label]['gpu_name'] = payloads[label].get('gpu_name')
        if timings:
            summary[label]['policy_forward_steady_ms'] = {
                'cycles': len(timings), 'mean': float(np.mean(timings)),
                'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95))}
            summary[label]['latency_scope'] = payloads[label].get('latency_scope')
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    intro = f'{len(episodes)} matched policy trials. Panels left to right: ground-truth reference, {", ".join(labels)}. Playback {args.speed:g}x; aligned from video start at nominal sample time, not by task phase. Earlier finishes hold their final frame. When an exact test-episode recording is unavailable, the reference is a training expert demonstration with the same complete skill chain and parameters, selected to prefer the same object instance. Its episode and scene can differ.'
    totals = ' | '.join(f'{label}: {row["successes"]}/{row["trials"]} successes' for label, row in summary.items())
    totals += ' | ' + ' | '.join(
        f'{label}: seed mean {row["mean_seed_success_rate"]:.1%}, sample SD {row["std_seed_success_rate"]:.1%}'
        for label, row in summary.items() if row['std_seed_success_rate'] is not None)
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Evaluation GIF comparisons</title><style>body{font:16px system-ui;background:#17191d;color:#eee;max-width:1600px;margin:auto;padding:24px}article{background:#24272d;padding:16px;margin:20px 0}img{width:100%}a{color:#9bd0ff}h2{font-size:18px}</style>'
            '<h1>Evaluation GIF comparisons</h1><p>' + html.escape(intro) + '</p><p>' + html.escape(totals) + '</p>'
            '<p>Blue: predicted contact. Green: nearest sampled point on the true target part. These overlays describe geometry, not physical contact or an expert trajectory. Separate policy rollouts share initial conditions but develop different observations.</p>'
            '<p><a href="comparison_bundle.zip">Download all comparison GIFs</a> · <a href="summary.json">Summary</a></p>' + ''.join(cards))
    (root / 'index.html').write_text(page)
    write_bundle(root)


if __name__ == '__main__':
    main()
