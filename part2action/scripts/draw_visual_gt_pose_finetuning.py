"""Draw the implemented GT-pose stage and its matched fine-tuning control.

This is a new report-figure version. Historical architecture figures and frozen
training/evaluation sources are preserved. The run configs are read as evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "part2action/docs/figures/2026-11-10_visual_gt_pose_finetuning"
DEFAULT_RUN = ROOT / "results/diagnostics/visual_oracle_pose_20261110_01"
STEM = "visual_gt_pose_finetuning"

MERMAID = """flowchart LR
  INIT["SAME previously trained visual-policy checkpoint<br/>Two further fine-tuning epochs per arm"]
  subgraph A["(a) Matched control: pose guidance OFF"]
    direction TB
    A0["RGB history + task text"]
    AS["Scene PCD + current TCP pose + opening"]
    A1["FROZEN: DINOv2 + Flan-T5"]
    A2["FINE-TUNE: scene encoder, visual part gate,<br/>temporal encoder, skill hierarchy and fusion"]
    AH["Original policy context h_t"]
    AZ["Same pose-module structure: disabled<br/>Zero feature contribution"]
    AF["Action features = h_t + 0"]
    A3["FINE-TUNE: original Transformer diffusion action head"]
    A4["8 x 7 command chunk<br/>Execute first command; observe again"]
    A0 --> A1 --> A2
    AS --> A2 --> AH --> AF --> A3 --> A4
    AZ --> AF
    A4 --> A0
    A4 --> AS
  end
  subgraph B["(b) GT pose guidance ON"]
    direction TB
    B0["SAME RGB history + task text"]
    BS["SAME scene PCD + current TCP pose + opening"]
    B1["FROZEN: DINOv2 + Flan-T5"]
    B2["FINE-TUNE: same scene encoder, visual part gate,<br/>temporal encoder, skill hierarchy and fusion"]
    BH["Original policy context h_t"]
    GT["Supplied GT target grasp pose<br/>Fixed target position + orientation"]
    P["TRAIN NEW pose-conditioning MLP<br/>TCP-relative pose + validity<br/>10 to 256 to 256"]
    BF["Action features = h_t + pose embedding"]
    B3["FINE-TUNE: same Transformer diffusion action head"]
    B4["8 x 7 command chunk<br/>Execute first command; observe again"]
    B0 --> B1 --> B2
    BS --> B2 --> BH --> BF --> B3 --> B4
    BS -->|"Current TCP pose"| P
    GT --> P --> BF
    B4 --> B0
    B4 --> BS
  end
  INIT -.-> A2
  INIT -.-> B2
  classDef frozen fill:#f0f2f5,stroke:#7b8794,color:#19334e;
  classDef finetune fill:#eaf2fc,stroke:#285da8,color:#19334e;
  classDef added fill:#e5f5f0,stroke:#18766d,color:#19334e;
  classDef oracle fill:#fff4df,stroke:#a76018,color:#73400e;
  class A1,B1,AZ frozen;
  class A2,A3,B2,B3 finetune;
  class P added;
  class GT oracle;
%% This first stage supplies GT pose; no learned pose estimator is present.
%% The new module's final projection is zero initialized: both arms start at the original policy.
%% New pose weights and existing trainable policy weights learn from the action loss.
%% GT target stays fixed; robot-relative pose features refresh as the robot moves.
%% No GT phase clock, recorded prefix/tail, or separate small action controller.
"""

CAPTION = r"""% Implemented first-stage training setup; success-rate results are separate.
% Requires graphicx. Copy the PDF beside this snippet or adjust its path.
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{visual_gt_pose_finetuning.pdf}
  \caption{Matched fine-tuning setup for supplied GT grasp-pose guidance in the
  original visual policy. Both arms initialize shared policy weights from the
  same previously trained checkpoint and retain RGB, scene point clouds,
  language, measured TCP pose, and gripper opening. DINOv2 and Flan-T5 remain
  frozen. Existing scene, grounding, temporal, hierarchy, fusion, and diffusion
  action-head parameters are fine-tuned for two additional epochs in both arms.
  In the GT arm, a newly initialized MLP embeds the target pose relative to the
  current TCP; its contribution is added to the original action features before
  the unchanged diffusion action-head architecture predicts commands. The
  control disables this contribution. A zero-initialized final projection makes
  both arms initially reproduce the old policy. The pose module and existing
  trainable policy parameters learn together from action supervision. GT pose is
  supplied throughout this first-stage oracle experiment; a learned pose
  estimator is reserved for a later stage. All task actions are policy generated.}
  \label{fig:visual-gt-pose-finetuning}
\end{figure*}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_configs(run):
    configs = {arm: yaml.safe_load((run / (arm + ".yaml")).read_text())
               for arm in ("control", "oracle_pose")}
    a, b = configs.values()
    for field in ("data", "model", "train", "losses", "heads", "hierarchy", "rollout"):
        if a[field] != b[field]:
            raise ValueError("Unmatched configuration field: " + field)
    if a["train"]["freeze_prefixes"] != ["visual", "text"]:
        raise ValueError("Frozen encoder description needs updating")
    if a["train"]["epochs"] != 2 or a["model"]["hidden_dim"] != 256:
        raise ValueError("Training/pose-module dimensions differ from this figure")
    if not a["train"].get("init_checkpoint"):
        raise ValueError("This diagram requires pretrained-policy fine-tuning")
    if a["pose_guidance"]["enabled"] or not b["pose_guidance"]["enabled"]:
        raise ValueError("Unexpected control/GT roles")
    return configs


def render(out, dpi):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(20, 16.7), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 2000), ylim=(1670, 0)); ax.axis("off")
    navy, muted = "#19334e", "#53677a"
    blue, teal, amber, gray = "#285da8", "#18766d", "#a76018", "#7b8794"
    old_bg, new_bg, oracle_bg, frozen_bg = "#eaf2fc", "#e5f5f0", "#fff4df", "#f0f2f5"
    labels, contained = [], []

    def text(x, y, label, size=17, color=navy, bold=False, ha="center", rotation=0):
        item = ax.text(x, y, label, fontsize=size, color=color,
            fontweight="bold" if bold else "normal", ha=ha, va="center",
            rotation=rotation, linespacing=1.35, zorder=6)
        labels.append(item); return item

    def rect(x, y, w, h, fill="white", edge="#b9c8d5", zorder=3, radius=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
            edgecolor=edge, linewidth=1.4, zorder=zorder))

    def box(x, y, w, h, title, detail="", fill="white", edge="#b9c8d5",
            title_size=19, detail_size=16, title_color=navy):
        rect(x, y, w, h, fill, edge)
        item = text(x+w/2, y+(29 if detail else h/2), title,
                    title_size, color=title_color, bold=True)
        contained.append((item, (x, y, w, h)))
        if detail:
            item = text(x+w/2, y+(h+35)/2, detail, detail_size, color=muted)
            contained.append((item, (x, y, w, h)))

    def arrow(points, color=muted):
        if len(points)>2:
            ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                    color=color, lw=1.7, zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
            color=color, lw=1.7, mutation_scale=18, shrinkA=0, shrinkB=0, zorder=2))

    text(40, 43, "Stage 1: GT grasp-pose guidance in the original visual policy", 27, bold=True, ha="left")
    text(40, 87, "Shared initialization: the SAME trained visual checkpoint → two further fine-tuning epochs per arm",
         18, color=muted, ha="left")
    rect(40, 130, 920, 1305, fill="#fbfcfe", edge="#d7e1e9", zorder=0)
    rect(1000, 130, 960, 1305, fill="#fbfdfc", edge="#d7e1e9", zorder=0)
    text(500, 171, "(a) Matched control", 25, blue, bold=True)
    text(1480, 171, "(b) GT pose guidance", 25, teal, bold=True)
    text(500, 208, "Fine-tune the existing policy · pose input OFF", 17, color=muted)
    text(1480, 208, "Fine-tune the policy + train the new pose module", 17, color=muted)

    for dx in (0, 960):
        x = 140+dx
        box(x, 265, 415, 85, "RGB history + task text", "Same observations in both arms", title_size=19)
        box(605+dx, 265, 265, 85, "Scene + state", "PCD + TCP + opening", title_size=16.5, detail_size=15)
        box(x, 410, 415, 100, "FROZEN: DINOv2 + Flan-T5", "Visual and language encoders",
            fill=frozen_bg, edge=gray, title_size=17)
        box(x, 570, 730, 130, "FINE-TUNE: existing policy modules",
            "Scene encoder + visual part gate\nTemporal encoder + skill hierarchy + fusion",
            fill=old_bg, edge=blue)
        arrow([(347.5+dx, 350), (347.5+dx, 410)])
        arrow([(347.5+dx, 510), (347.5+dx, 570)])
        arrow([(737.5+dx, 350), (737.5+dx, 570)])
        box(x, 765, 340, 95, "Policy context h_t",
            "Original visual / scene /\nskill / state features", title_size=18, detail_size=15)
        arrow([(505+dx, 700), (505+dx, 730), (310+dx, 730), (310+dx, 765)])

        enabled = bool(dx)
        box(545+dx, 765, 325, 95,
            "GT target grasp pose" if enabled else "No GT pose signal",
            "Position + orientation\nFixed target for the episode" if enabled else "Control input = zero\nSame training data / budget",
            fill=oracle_bg if enabled else frozen_bg,
            edge=amber if enabled else gray, title_size=17, detail_size=15,
            title_color=amber if enabled else navy)
        box(545+dx, 925, 325, 125,
            "TRAIN: pose module" if enabled else "Pose module disabled",
            "TCP-relative pose + validity\n10 → 256 → 256" if enabled else "Same module structure\nZero feature contribution",
            fill=new_bg if enabled else frozen_bg,
            edge=teal if enabled else gray, title_size=17, detail_size=15)
        box(x, 925, 340, 125, "Action features",
            "h_t + pose embedding" if enabled else "h_t + 0", title_size=19, detail_size=17)
        arrow([(707.5+dx, 860), (707.5+dx, 925)], amber if enabled else gray)
        arrow([(545+dx, 987.5), (480+dx, 987.5)], teal if enabled else gray)
        arrow([(310+dx, 860), (310+dx, 925)])
        if enabled:
            arrow([(870+dx, 307.5), (925+dx, 307.5), (925+dx, 987.5), (870+dx, 987.5)])
            text(946+dx, 665, "Current TCP pose", 14.5, color=muted, rotation=90)

        box(x, 1125, 730, 105, "FINE-TUNE: original diffusion action head",
            "Transformer diffusion → 8 × 7 command chunk", fill=old_bg, edge=blue, title_size=20)
        arrow([(310+dx, 1050), (310+dx, 1085), (505+dx, 1085), (505+dx, 1125)])
        box(x, 1290, 730, 85, "Execute first command",
            "TCP position + orientation + gripper", fill="#eef6f3", edge="#46796a")
        arrow([(505+dx, 1230), (505+dx, 1290)])
        arrow([(x, 1332.5), (90+dx, 1332.5), (90+dx, 245), (347.5+dx, 245), (347.5+dx, 265)])
        arrow([(90+dx, 245), (737.5+dx, 245), (737.5+dx, 265)])
        text(65+dx, 855, "Refresh visual observations + current robot state", 14.5, color=muted, rotation=90)

    for x, fill, edge, label in [(45, frozen_bg, gray, "Frozen / inactive blocks"),
            (545, old_bg, blue, "Fine-tuned policy weights"),
            (1060, new_bg, teal, "Newly trained pose weights"),
            (1610, oracle_bg, amber, "Supplied GT pose")]:
        rect(x, 1471, 25, 22, fill, edge, radius=4)
        text(x+39, 1482, label, 16, color=muted, ha="left")
    text(40, 1531, "The existing policy starts from trained weights. Only the added pose module has new weights; its final projection starts at zero.",
         16.5, bold=True, ha="left")
    text(40, 1577, "Action supervision trains the blue and teal blocks together. This stage supplies GT pose; it does not train a pose estimator.",
         16.5, color=muted, ha="left")
    text(40, 1620, "Both arms use matched data, initialization, training budget, and evaluation. All full-task actions come from the visual policy.",
         16.5, color=muted, ha="left")

    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); errors=[]
    for item in labels:
        b=item.get_window_extent(renderer)
        if not (fig.bbox.x0<=b.x0 and b.x1<=fig.bbox.x1 and fig.bbox.y0<=b.y0 and b.y1<=fig.bbox.y1):
            errors.append("Clipped text: "+item.get_text())
    for item,(x,y,w,h) in contained:
        b=item.get_window_extent(renderer)
        a,z=ax.transData.transform((x+7,y+4)),ax.transData.transform((x+w-7,y+h-4))
        if not (a[0]<=b.x0 and b.x1<=z[0] and z[1]<=b.y0 and b.y1<=a[1]):
            errors.append("Text exceeds box: "+item.get_text())
    for i,first in enumerate(labels):
        for second in labels[i+1:]:
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                errors.append("Overlapping text: "+first.get_text()+" / "+second.get_text())
    if errors:
        raise ValueError("\n".join(errors))
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"{STEM}.{ext}",dpi=dpi,facecolor="white")
    plt.close(fig)
    (out/f"{STEM}.mmd").write_text(MERMAID)
    (out/"report_figure.tex").write_text(CAPTION)
    with Image.open(out/f"{STEM}.png") as raster:
        dimensions=list(raster.size)
    return {"layout_checks":"passed","png_dimensions":dimensions}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--dpi',type=int,default=240)
    args=parser.parse_args();out=args.out_dir.resolve();run=args.run_dir.resolve()
    configs=validate_configs(run)
    out.mkdir(parents=True,exist_ok=True)
    if list(out.iterdir()):
        raise FileExistsError('Use a fresh output directory to preserve previous figures')
    checks=render(out,args.dpi)
    sources=[ROOT/'part2action/models/pose_conditioning_v1.py',ROOT/'part2action/models/visual_pose_policy_v1.py',
             ROOT/'part2action/scripts/train_visual_pose_v1.py',run/'control.yaml',run/'oracle_pose.yaml']
    manifest={'status':'implemented_stage1_training_setup_no_performance_claim',
        'created_utc':datetime.now(timezone.utc).isoformat(),'generator':str(Path(__file__).resolve().relative_to(ROOT)),
        'generator_sha256':sha(__file__),'source_sha256':{str(p):sha(p) for p in sources},
        'training':{'shared_pretrained_checkpoint':configs['control']['train']['init_checkpoint'],
                    'additional_epochs':2,'frozen_prefixes':['visual','text'],
                    'new_pose_module':'new weights, zero-initialized final projection',
                    'pose_estimator_present':False},
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir())},**checks}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(out),**checks},indent=2))


if __name__=='__main__':
    main()
