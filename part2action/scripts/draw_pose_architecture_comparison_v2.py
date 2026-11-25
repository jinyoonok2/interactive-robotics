"""Report figures separating the evaluated GT-pose change from a proposed redesign.

Only figure artifacts are written. Training, evaluation, and old figures are
preserved. Internal auxiliary losses/heads are omitted from these flow diagrams.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'part2action/docs/slide_history/media/2026-11-15'
RUN = ROOT / 'results/diagnostics/visual_oracle_pose_20261110_01'
INK, MUTED = '#17324d', '#52677b'
BLUE, TEAL, AMBER, GRAY = '#3167aa', '#16786b', '#ad6917', '#8895a2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Drawing:
    def __init__(self, title, subtitle):
        self.fig, self.ax = plt.subplots(figsize=(16, 11.2), dpi=100)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set(xlim=(0, 1600), ylim=(1120, 0))
        self.ax.axis('off')
        self.labels, self.contained = [], []
        self.text(35, 43, title, 23, bold=True, ha='left')
        self.text(35, 88, subtitle, 13.5, color=MUTED, ha='left')

    def text(self, x, y, value, size=14, color=INK, bold=False, ha='center', rotation=0):
        label = self.ax.text(x, y, value, fontsize=size, color=color,
            fontweight='bold' if bold else 'normal', ha=ha, va='center',
            linespacing=1.4, rotation=rotation, zorder=5)
        self.labels.append(label)
        return label

    def rect(self, x, y, w, h, fill='#ffffff', edge='#c2d0dd', zorder=2):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0,rounding_size=12', facecolor=fill,
            edgecolor=edge, linewidth=1.35, zorder=zorder))

    def box(self, x, y, w, h, title, detail='', fill='white', edge='#c2d0dd', size=15):
        self.rect(x, y, w, h, fill, edge)
        label = self.text(x+w/2, y+(h/2 if not detail else 25), title, size, bold=True)
        self.contained.append((label, (x, y, w, h)))
        if detail:
            label = self.text(x+w/2, y+(h+34)/2, detail, 12.5, color=MUTED)
            self.contained.append((label, (x, y, w, h)))

    def arrow(self, points, color=MUTED):
        if len(points) > 2:
            self.ax.plot(*zip(*points[:-1]), color=color, lw=1.55, zorder=3)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            color=color, lw=1.55, mutation_scale=15, shrinkA=0, shrinkB=0, zorder=3))

    def panel(self, x, title, subtitle, color):
        self.rect(x, 130, 750, 915, fill='#fbfcfd', edge='#d7e1e9', zorder=0)
        self.text(x+375, 168, title, 19, color=color, bold=True)
        self.text(x+375, 199, subtitle, 12.5, color=MUTED)

    def save(self, out, stem, dpi):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        errors = []
        for label in self.labels:
            if not self.fig.bbox.contains(*label.get_window_extent(renderer).get_points()[0]) or not self.fig.bbox.contains(*label.get_window_extent(renderer).get_points()[1]):
                errors.append('Clipped text: '+label.get_text())
        for label, (x, y, w, h) in self.contained:
            bounds = label.get_window_extent(renderer)
            a, b = self.ax.transData.transform((x+6, y+4)), self.ax.transData.transform((x+w-6, y+h-4))
            if not (a[0] <= bounds.x0 and bounds.x1 <= b[0] and b[1] <= bounds.y0 and bounds.y1 <= a[1]):
                errors.append('Text exceeds box: '+label.get_text())
        for i, first in enumerate(self.labels):
            for second in self.labels[i+1:]:
                if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                    errors.append('Overlapping labels: '+first.get_text()+' / '+second.get_text())
        if errors:
            raise ValueError('\n'.join(errors))
        for extension in ('png', 'svg', 'pdf'):
            self.fig.savefig(out / f'{stem}.{extension}', dpi=dpi, facecolor='white')
        plt.close(self.fig)


def inputs(d, x, proposed=False):
    d.box(x+30, 225, 690, 80, 'Visual observations + instruction + measured robot state',
          'RGB history; observed scene point cloud; current TCP pose; gripper opening', size=14.5)
    d.box(x+30, 360, 690, 100,
          'Shared visual / 3D encoder' if proposed else 'Existing visual policy feature extraction',
          'Associate image features with observed 3D locations' if proposed else
          'DINOv2 + T5; scene, temporal, skill and fusion modules',
          fill='#edf3fb', edge=BLUE)
    d.arrow([(x+375, 305), (x+375, 360)])


def additive_panel(d, x, enabled):
    inputs(d, x)
    d.box(x+30, 520, 315, 76, 'Original action features', 'Visual / scene / skill / state', fill='#edf3fb', edge=BLUE)
    d.arrow([(x+187, 460), (x+187, 520)])
    d.box(x+395, 520, 325, 76,
          'Supplied GT grasp target' if enabled else 'No target grasp pose',
          'Expert target position + orientation' if enabled else 'No target-pose estimation',
          fill='#fff4df' if enabled else '#f1f3f5', edge=AMBER if enabled else GRAY, size=14)
    d.box(x+395, 645, 325, 100, 'Pose-conditioning MLP' if enabled else 'Pose module disabled',
          'Target relative to current TCP\n10 inputs → 256 → 256' if enabled else
          'Zero feature contribution\nNo supplied or estimated target',
          fill='#e6f4ef' if enabled else '#f1f3f5', edge=TEAL if enabled else GRAY, size=14)
    d.arrow([(x+557, 596), (x+557, 645)], AMBER if enabled else GRAY)
    if enabled:
        d.arrow([(x+720, 265), (x+735, 265), (x+735, 695), (x+720, 695)])
        d.text(x+752, 440, 'Current TCP', 10.5, rotation=90, color=MUTED)
    d.ax.add_patch(Circle((x+187, 695), 23, facecolor='white', edgecolor=TEAL if enabled else GRAY, lw=1.6, zorder=4))
    d.text(x+187, 695, '+', 22, bold=True)
    d.arrow([(x+187, 596), (x+187, 672)])
    d.arrow([(x+395, 695), (x+210, 695)], TEAL if enabled else GRAY)
    d.text(x+187, 754, 'h + pose embedding' if enabled else 'h + 0', 13, color=TEAL if enabled else MUTED)
    d.box(x+30, 800, 690, 90, 'Transformer diffusion action decoder',
          'Same decoder architecture; fine-tuned separately in each arm', fill='#edf3fb', edge=BLUE)
    d.arrow([(x+187, 718), (x+187, 737), (x+42, 737), (x+42, 781), (x+375, 781), (x+375, 800)])
    d.box(x+30, 950, 690, 75, 'Motion chunk + gripper command',
          'Execute first action, then refresh observations')
    d.arrow([(x+375, 890), (x+375, 950)])


def proposed_panel(d, x):
    inputs(d, x, proposed=True)
    d.box(x+30, 520, 315, 76, 'Scene / visual tokens', 'Retain global and local context', fill='#edf3fb', edge=BLUE, size=14)
    d.box(x+395, 520, 325, 76, 'Task-relevant 3D region', 'Relevant object / part geometry', fill='#e6f4ef', edge=TEAL, size=14)
    d.arrow([(x+187, 460), (x+187, 520)])
    d.arrow([(x+557, 460), (x+557, 520)])
    d.box(x+30, 645, 315, 100, 'Robot / skill tokens',
          'Current TCP + gripper opening\nRecent history + active skill', fill='#edf3fb', edge=BLUE, size=14)
    d.arrow([(x+30, 265), (x+15, 265), (x+15, 695), (x+30, 695)])
    d.box(x+395, 645, 325, 100, 'Learned target-pose head',
          'Target TCP position + orientation\nPredicted applicability', fill='#e6f4ef', edge=TEAL, size=14)
    d.arrow([(x+557, 596), (x+557, 645)], TEAL)
    d.arrow([(x+345, 695), (x+395, 695)])
    d.box(x+30, 800, 690, 90, 'Diffusion decoder with cross-attention',
          'Scene + pose + robot tokens; explicit TCP-relative pose error', fill='#e6f4ef', edge=TEAL)
    d.arrow([(x+187, 745), (x+187, 800)])
    d.arrow([(x+557, 745), (x+557, 800)], TEAL)
    d.arrow([(x+30, 558), (x+23, 558), (x+23, 845), (x+30, 845)], BLUE)
    d.box(x+30, 950, 465, 75, 'Motion chunk + gripper command',
          'Execute first action; observe again', size=14)
    d.box(x+535, 950, 185, 75, 'Skill progress', 'Update the goal', fill='#e6f4ef', edge=TEAL, size=13)
    d.arrow([(x+262, 890), (x+262, 950)])
    d.arrow([(x+627, 890), (x+627, 950)], TEAL)
    d.arrow([(x+720, 987), (x+735, 987), (x+735, 558), (x+720, 558)], TEAL)


ACTUAL_MERMAID = '''flowchart TB
  subgraph A["Evaluated: pose guidance OFF"]
    A0["RGB + observed PCD + language + measured TCP/opening"] --> A1["Original visual policy features h"]
    A1 --> A2["Action features h + 0"]
    AZ["Same pose module; disabled"] --> A2
    A2 --> A3["Transformer diffusion decoder"] --> A4["Motion and gripper commands"]
  end
  subgraph B["Evaluated: GT pose guidance ON"]
    B0["Same observations and measured robot state"] --> B1["Original visual policy features h"]
    BG["Supplied expert target grasp pose"] --> BP["Current-TCP-relative pose → MLP 10 → 256 → 256"]
    B0 --> BP
    BP --> B2["Action features h + pose embedding"]
    B1 --> B2 --> B3["Same diffusion decoder architecture"] --> B4["Motion and gripper commands"]
  end
%% Neither evaluated arm uses a learned target-pose estimator.
%% Both initialize from the same pretrained policy and fine-tune for two epochs.
'''

PROPOSED_MERMAID = '''flowchart TB
  subgraph A["Current evaluated GT-pose architecture"]
    A0["Visual observations + language + measured robot state"] --> A1["Original policy features"]
    AG["Supplied GT grasp pose + current TCP"] --> AP["Pose MLP"]
    AP --> AS["Add embedding to features"]
    A1 --> AS --> AD["Pooled conditioning → diffusion decoder"] --> AA["Actions"]
  end
  subgraph B["Proposed pose-first architecture: not trained/evaluated"]
    B0["Same available visual observations, language and robot state"] --> BE["Spatially associated RGB / 3D features"]
    BE --> BG["Task-conditioned region"] --> BP["Learned target TCP pose + applicability"]
    BE --> BD["Cross-attention diffusion decoder"]
    BP --> BD
    B0 --> BS["Robot / skill / history tokens + relative pose error"] --> BD
    BD --> BA["Actions"]
    BP --> BH["Skill progress from goal, scene and state"]
    BE --> BH
    BS --> BH --> BG
  end
%% GT can replace predicted target poses for a controlled interface experiment.
%% Later-skill goals require new supervision; a single pose is not a full task trajectory.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    configs = {arm: yaml.safe_load((RUN / f'{arm}.yaml').read_text()) for arm in ('control', 'oracle_pose')}
    for key in ('data', 'model', 'train', 'losses', 'heads', 'hierarchy', 'rollout'):
        if configs['control'][key] != configs['oracle_pose'][key]:
            raise ValueError('Unmatched experiment field: '+key)
    if configs['control']['pose_guidance']['enabled'] or not configs['oracle_pose']['pose_guidance']['enabled']:
        raise ValueError('Unexpected experiment arms')
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if list(out.iterdir()):
        raise FileExistsError('Use a new directory to preserve earlier artifacts')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    d = Drawing('What changed in the completed GT-pose experiment',
        'A supplied grasp target conditions the visual policy; neither arm contains a learned target-pose estimator.')
    d.panel(35, '(a) Visual policy: pose guidance OFF', 'Matched fine-tuned control', BLUE)
    d.panel(815, '(b) Visual policy: GT pose guidance ON', 'Same architecture + an active target-pose conditioning branch', TEAL)
    additive_panel(d, 35, False)
    additive_panel(d, 815, True)
    d.text(35, 1080, 'Both arms: same pretrained initialization, two further fine-tuning epochs; DINOv2 and T5 frozen.', 13, ha='left')
    d.save(out, 'evaluated_pose_off_vs_gt_pose', args.dpi)
    d = Drawing('Current architecture versus the proposed pose-first redesign',
        'The left architecture was evaluated. The right architecture is a design proposal with no reported performance result.')
    d.panel(35, '(a) Current: additive GT-pose conditioning', 'Evaluated architecture', BLUE)
    d.panel(815, '(b) Proposed: pose-to-action architecture', 'Design proposal; not trained or evaluated', TEAL)
    additive_panel(d, 35, True)
    proposed_panel(d, 815)
    d.text(35, 1080, 'Proposed testing: supply the same GT targets first; then replace them with predicted targets at the same interface.', 12.5, ha='left')
    d.save(out, 'current_vs_proposed_pose_first', args.dpi)
    for stem, mermaid in [('evaluated_pose_off_vs_gt_pose', ACTUAL_MERMAID),
                          ('current_vs_proposed_pose_first', PROPOSED_MERMAID)]:
        (out / f'{stem}.mmd').write_text(mermaid)
    (out / 'report_figures.tex').write_text(r'''% Requires graphicx. Copy the PDF files alongside this snippet or adjust paths.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{evaluated_pose_off_vs_gt_pose.pdf}
\caption{Evaluated comparison of target-pose guidance in the original visual policy.
Both arms retain RGB history, observed scene point clouds, language, measured current TCP pose,
and gripper opening. The control masks the pose module's contribution to zero. The GT arm
embeds a supplied expert-reference grasp TCP target relative to the measured current TCP and
adds the embedding to the original action features. The diffusion decoder architecture is
the same; learned weights are fine-tuned separately from the same pretrained initialization.
No learned target-pose estimator or separate action-overriding controller is used in either arm.
The first-grasp reference target is static and supplied through the episode where available.
Auxiliary heads and losses are omitted for readability.}
\label{fig:evaluated-pose-guidance}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{current_vs_proposed_pose_first.pdf}
\caption{Current additive GT-pose conditioning versus a proposed pose-first redesign.
The proposal associates RGB features with observed 3D locations, grounds a task-relevant
region, predicts a target TCP pose and applicability, and preserves scene, pose, and robot
context as tokens for a cross-attention diffusion actor. A progress head uses the goal,
scene, and robot state to support skill transitions. Its arrows are a conceptual dependency
diagram rather than a finalized software graph. A controlled experiment can substitute the
same supplied GT targets at the proposed pose interface before evaluating learned targets.
Later-skill goals require additional supervision. The proposed architecture has not been
trained or evaluated, and no performance improvement is claimed.}
\label{fig:proposed-pose-first}
\end{figure*}
''')
    sources = [Path(__file__), ROOT / 'part2action/models/visual_pose_policy_v1.py',
        ROOT / 'part2action/models/pose_conditioning_v1.py', ROOT / 'part2action/models/heads.py',
        ROOT / 'part2action/models/diffusion_transformer.py', ROOT / 'part2action/models/hierarchical.py',
        RUN / 'control.yaml', RUN / 'oracle_pose.yaml']
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'layout_checks': 'passed: text containment, clipping, pairwise label overlap',
        'figure_scope': {'evaluated_pose_off_vs_gt_pose': 'implemented and evaluated; not a learned-estimator comparison',
                         'current_vs_proposed_pose_first': 'evaluated current design versus unimplemented research proposal'},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
        'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir())}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output_dir': str(out), 'layout_checks': manifest['layout_checks']}))


if __name__ == '__main__':
    main()
