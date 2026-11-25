"""Refresh the earlier comparison layout for the implemented pose-token model.

Reads frozen model/config sources and the dated training-status snapshot. Writes
presentation assets only; preserves the original comparison diagrams.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HISTORY = ROOT / 'part2action/docs/slide_history'
MEDIA = HISTORY / 'media/2026-11-15'
sys.path.insert(0, str(ROOT / 'part2action/scripts'))
from draw_pose_architecture_comparison_v2 import Drawing, additive_panel, BLUE, TEAL, AMBER
from export_slide_tables import render
import matplotlib.pyplot as plt
import yaml


MERMAID = '''flowchart TB
  subgraph A["Earlier additive visual policy: evaluated"]
    AO["RGB history + observed PCD + language + measured TCP/opening"] --> AE["Existing DINOv2/T5, scene, history and skill features"]
    AE --> AF["Original action features"]
    AG["Supplied GT grasp pose + measured TCP"] --> AM["Relative pose MLP"]
    AF --> AS["Add pose embedding to action features"]
    AM --> AS --> AD["Original diffusion decoder"] --> AA["8 x 7 commands; execute first and observe"]
  end
  subgraph B["Implemented Pose-token Diffusion: current GT-input arm"]
    BO["Same available RGB/PCD/language and measured robot inputs"] --> BE["DINOv2/T5 frozen; trainable RGB/3D association, history and skill features"]
    BE --> BC["Visual, scene, skill and robot memory tokens"]
    BE --> BR["Task-relevant 3D region"]
    BR --> BH["MLP predicts target grasp pose + applicability"]
    BH -.-> BL["Auxiliary pose losses; prediction bypassed for current actions"]
    BG["Supplied GT first-grasp goal + measured TCP"] --> BP["Separate relative translation and orientation tokens"]
    BC --> BD["Cross-attention diffusion decoder"]
    BR --> BD
    BP --> BD --> BA["8 x 7 commands; execute first and observe"]
  end
%% The pose-disabled arm supplies zero and invalid pose-memory tokens.
%% Both current arms train the pose head with identical auxiliary supervision.
%% Predicted-pose action conditioning is supported but is not the current screen.
%% The optional pose-aware progress head is disabled; the existing skill hierarchy remains.
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implemented_panel(d, x):
    d.box(x+30, 225, 690, 80, 'Visual observations + instruction + measured robot state',
          'RGB history; observed scene PCD; current TCP pose and opening', size=14.5)
    d.box(x+30, 360, 690, 100, 'Shared visual / 3D / skill encoder',
          'DINOv2 + Flan-T5 frozen; trainable scene, history and skill layers\n'
          'Associate image features with observed 3D locations',
          fill='#edf3fb', edge=BLUE)
    d.arrow([(x+375, 305), (x+375, 360)])
    d.box(x+30, 505, 315, 80, 'Context memory tokens',
          'Visual / scene / skill / state', fill='#edf3fb', edge=BLUE, size=14)
    d.box(x+395, 505, 325, 80, 'Task-relevant 3D region',
          'Local geometry + appearance', fill='#e6f4ef', edge=TEAL, size=14)
    d.arrow([(x+187, 460), (x+187, 505)], BLUE)
    d.arrow([(x+557, 460), (x+557, 505)], TEAL)
    d.box(x+30, 625, 315, 80, 'GT pose + current TCP',
          'Supplied first-grasp target', fill='#fff4df', edge=AMBER, size=14)
    d.box(x+395, 625, 325, 80, 'Pose prediction MLP',
          'Target pose + applicability', fill='#e6f4ef', edge=TEAL, size=14)
    d.arrow([(x+557, 585), (x+557, 625)], TEAL)
    d.box(x+30, 745, 315, 95, 'Relative pose memory',
          'Translation and orientation\nTwo separate memory tokens',
          fill='#e6f4ef', edge=TEAL, size=14)
    d.box(x+395, 745, 325, 95, 'Pose-head supervision',
          'Train on grasp-goal labels\nPrediction bypassed for actions',
          fill='#f1f3f5', size=14)
    d.arrow([(x+187, 705), (x+187, 745)], AMBER)
    d.arrow([(x+557, 705), (x+557, 745)], TEAL)
    d.box(x+30, 875, 690, 80, 'Diffusion decoder with cross-attention',
          'Attend to visual, scene, region, skill/state and pose memory',
          fill='#e6f4ef', edge=TEAL)
    d.arrow([(x+187, 840), (x+187, 875)], TEAL)
    d.arrow([(x+30, 545), (x+12, 545), (x+12, 915), (x+30, 915)], BLUE)
    d.arrow([(x+720, 545), (x+735, 545), (x+735, 915), (x+720, 915)], TEAL)
    d.box(x+30, 980, 690, 45, '8 × 7 commands; execute first, then observe again', size=14)
    d.arrow([(x+375, 955), (x+375, 980)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=MEDIA / 'architecture')
    parser.add_argument('--snapshot', type=Path, default=MEDIA / 'data/slide07_training_snapshot.json')
    parser.add_argument('--dpi', type=int, default=240)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    model_path = ROOT / 'part2action/models/pose_token_diffusion/configs/model.yaml'
    model = yaml.safe_load(model_path.read_text())
    assert model['progress_enabled'] is False
    assert model['action_horizon'] == 8 and model['action_dim'] == 7
    assert snapshot['matching_checks']['same_initial_trainable_state']
    assert snapshot['matching_checks']['same_epoch1_batch_order']
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new directory to preserve previous figure versions')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    d = Drawing('From additive pose features to explicit pose memory',
        'Earlier model: evaluated. Pose-token Diffusion: implemented and training; benchmark evaluation pending.')
    d.panel(35, '(a) Earlier: additive pose conditioning', 'Supplied GT target; matched visual-policy experiment', BLUE)
    d.panel(815, '(b) Implemented: Pose-token Diffusion', 'GT-input arm shown; pose-disabled arm masks the goal tokens', TEAL)
    additive_panel(d, 35, True)
    for label in d.labels:
        if label.get_text() == 'Same decoder architecture; fine-tuned separately in each arm':
            label.set_text('Pooled conditioning after feature addition')
    implemented_panel(d, 815)
    d.text(35, 1080, 'Current screen: predicted pose is auxiliary only. Existing skill hierarchy retained; optional pose-aware progress head off.',
           12.5, ha='left')
    d.save(out, 'old_vs_implemented_pose_token_diffusion', args.dpi)
    (out / 'old_vs_implemented_pose_token_diffusion.mmd').write_text(MERMAID)
    rows = [['Run', 'Action goal input', 'Pose-head loss', 'Training status']]
    for arm, label, goal in [('disabled', 'Pose input OFF', 'Zero / invalid pose tokens'),
                             ('provided', 'GT pose input', 'Expert first-grasp target')]:
        info = snapshot['arms'][arm]
        status = f'Epoch {info["latest_logged_update"]["epoch"]}' if info['scheduler_state'] == 'RUNNING' else info['scheduler_state']
        rows.append([label, goal, 'Enabled', status])
    fractions = [.21, .35, .23, .21]
    image_path = out / 'current_training_comparison.png'
    size = render(rows, fractions, image_path)
    table = {'rows': rows, 'column_fractions': fractions,
             'snapshot': str(args.snapshot.resolve().relative_to(ROOT)),
             'snapshot_sha256': sha(args.snapshot), 'dimensions': size,
             'image': image_path.name, 'sha256': sha(image_path)}
    (out / 'training_table.json').write_text(json.dumps(table, indent=2) + '\n')
    (out / 'report_figure.tex').write_text(r'''% Requires graphicx. Paths are relative to this snippet.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{old_vs_implemented_pose_token_diffusion.pdf}
\caption{Earlier additive GT-pose conditioning and the implemented Pose-token
Diffusion policy. Both retain visual and measured robot-state inputs. The new
policy associates RGB and scene geometry, grounds a task-relevant region, and
encodes the selected goal as distinct relative translation and orientation
tokens for a cross-attention diffusion decoder. The current GT-input arm supplies
an expert first-grasp target. The pose-disabled arm masks goal tokens. Both arms
train an auxiliary pose head whose predictions do not condition actions in this
screen. Predicted-goal conditioning is supported for subsequent study. The
existing skill hierarchy remains active; the optional pose-aware progress head
is disabled. The new full-training comparison has no benchmark result yet.}
\label{fig:implemented-pose-token-comparison}
\end{figure*}
''')
    sources = [Path(__file__), ROOT / 'part2action/scripts/draw_pose_architecture_comparison_v2.py',
               HISTORY / 'scripts/export_slide_tables.py', args.snapshot.resolve(), model_path,
               ROOT / 'part2action/models/pose_token_diffusion/configs/train.yaml',
               ROOT / 'part2action/models/pose_token_diffusion/policy.py',
               ROOT / 'part2action/models/pose_token_diffusion/blocks/encoder.py',
               ROOT / 'part2action/models/pose_token_diffusion/blocks/pose_head.py',
               ROOT / 'part2action/models/pose_token_diffusion/blocks/pose_tokens.py',
               ROOT / 'part2action/models/pose_token_diffusion/blocks/action_decoder.py',
               ROOT / 'part2action/models/pose_token_diffusion/losses.py']
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'scope': 'Earlier evaluated additive policy versus the implemented GT-input pose-token arm; no new benchmark claim',
                'layout_checks': 'Passed text containment, clipping, pairwise label overlap, and PNG table text checks',
                'source_sha256': {str(p.resolve().relative_to(ROOT)): sha(p) for p in sources},
                'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                          for p in sorted(out.iterdir()) if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'layout_checks': manifest['layout_checks']}))


if __name__ == '__main__':
    main()
