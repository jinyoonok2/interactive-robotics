"""Export the two supported visual pose paths as vector/report assets."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'part2action/docs/figures/2026-11-11_grounded_visual_pose'
STEM='visual_pose_architectures'
MMD='''flowchart TB
  subgraph A["A. Current GT-pose experiment — queued"]
    AO["RGB history + scene PCD + task + measured TCP/opening"] --> AE["Shared encoders, temporal features, hierarchy and fusion"]
    AE --> AG["Original global visual/scene/skill context"]
    AT["Supplied GT first-grasp TCP pose"] --> AC["Relative pose + validity → pose-conditioning MLP"]
    AO -->|"Measured TCP"| AC
    AG --> AF["Feature addition"]
    AC --> AF
    AF --> AA["Original Transformer diffusion head → 8 × 7 actions"]
  end
  subgraph B["B. Shared visual grounding + predicted pose — implemented, untrained"]
    BO["Same RGB history + scene PCD + task + measured TCP/opening"] --> BE["Shared frozen DINO/T5 + trainable scene/history/hierarchy"]
    BE --> BG["Original global visual/scene/skill context"]
    BE --> BR["Skill-conditioned soft part grounding"]
    BR --> BP["Weighted local geometry + visual attention + robot state → target TCP pose + applicability"]
    BP --> BC["Differentiable relative-pose conditioning MLP"]
    BG --> BF["Feature addition"]
    BC --> BF
    BF --> BA["Same Transformer diffusion head → 8 × 7 actions"]
    BL["TRAINING ONLY: part/contact labels + pre-closure grasp pose"] -.-> BR
    BL -.-> BP
  end
'''


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Refusing to overwrite report figure version')
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(16,11.8))
    ax.set(xlim=(0,16),ylim=(0,12));ax.axis('off');fig.patch.set_facecolor('white')
    colors={'input':'#f3f5f8','shared':'#eaf2fc','new':'#e5f5f0','gt':'#fff4df'}
    def box(x,y,w,h,label,kind='shared'):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.08',
                     facecolor=colors[kind],edgecolor='#547087',linewidth=1.2))
        ax.text(x,y,label,ha='center',va='center',fontsize=11,linespacing=1.35,color='#16324a')
    def arrow(start,end,dashed=False):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,
            linewidth=1.3,color='#49657c',linestyle='--' if dashed else '-',shrinkA=2,shrinkB=2))
    ax.text(8,11.6,'Visual policy with supplied or predicted grasp-pose guidance',ha='center',fontsize=20,weight='bold',color='#16324a')
    for x,title,subtitle in [(4,'A. Supplied GT pose','Current matched experiment · queued'),
                              (12,'B. Shared grounding and predicted pose','Implemented and CPU-tested · not trained')]:
        ax.text(x,10.85,title,ha='center',fontsize=16,weight='bold',color='#16324a')
        ax.text(x,10.45,subtitle,ha='center',fontsize=11,color='#566b7b')
        box(x,9.65,7.1,.85,'RGB history  •  observed scene PCD  •  task\nMeasured TCP pose and gripper opening','input')
        box(x,8.25,7.1,1.,'Shared RGB / scene / language features\nFrozen DINOv2 + T5; trainable history, hierarchy and fusion')
        arrow((x,9.18),(x,8.8))
        box(x-1.85,6.65,3.25,1.25,'Original global context\nVisual + scene + skill\nHistory + robot state')
        arrow((x-1.85,7.68),(x-1.85,7.33))
        box(x,2.5,7.1,.75,'Original action features + pose embedding')
        arrow((x-1.85,5.95),(x-1.85,2.93))
        box(x,1.2,7.1,1.,'Original Transformer diffusion action head\n8 × 7 chunk → execute first command → observe again')
        arrow((x,2.06),(x,1.76))
        if x==4:
            box(x+1.85,6.65,3.25,1.25,'Supplied GT grasp pose\nTarget TCP position\nand orientation','gt')
            box(x+1.85,3.95,3.25,1.25,'TCP-relative pose\n+ validity → MLP\n10 → 256 → 256','new')
            arrow((x+1.85,5.95),(x+1.85,4.63))
        else:
            box(x+1.85,6.65,3.25,1.25,'Soft target-part region\nSkill-conditioned\nshared scene grounding','new')
            arrow((x+1.85,7.68),(x+1.85,7.33))
            box(x+1.85,5.15,3.25,1.2,'Local geometry + visual\nattention + TCP/opening\n→ pose + applicability','new')
            arrow((x+1.85,5.95),(x+1.85,5.8))
            box(x+1.85,3.75,3.25,.95,'Differentiable pose MLP\n10 → 256 → 256','new')
            arrow((x+1.85,4.49),(x+1.85,4.29))
        arrow((x+1.85,3.25),(x+1.85,2.93))
    ax.axvline(8,ymin=.05,ymax=.9,color='#d7dfe7',linewidth=1)
    ax.text(8,.25,'B uses pose/part/contact labels only for training losses. Both retain vision and the original action decoder.',
            ha='center',fontsize=11,color='#49657c')
    fig.subplots_adjust(left=.025,right=.975,top=.985,bottom=.025)
    for ext in ('svg','pdf','png'):
        fig.savefig(OUT/f'{STEM}.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
    (OUT/f'{STEM}.mmd').write_text(MMD)
    (OUT/'report_figure.tex').write_text(r'''\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{visual_pose_architectures.pdf}
  \caption{Two visual-policy pose-conditioning paths. (A) The current matched
  experiment supplies an oracle first-grasp TCP pose. (B) The implemented next
  architecture predicts a target TCP pose from shared skill-conditioned scene
  grounding, local geometry, visual attention and measured robot state. Its
  differentiable pose embedding conditions the original diffusion action head
  alongside the global visual context. Part/contact and pose labels supervise
  training only in B. DINO/T5 remain frozen; shared policy layers fine-tune and
  new pose layers learn jointly. B has not yet been trained or evaluated.}
  \label{fig:visual-pose-architectures}
\end{figure*}
''')
    sources=[Path(__file__),ROOT/'part2action/models/grounded_visual_pose_policy_v1.py',
             ROOT/'part2action/models/grounded_pose_head_v1.py',ROOT/'part2action/models/visual_pose_policy_v1.py']
    (OUT/'manifest.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},
        'status':'GT experiment queued; new predicted-pose architecture implemented but untrained'},indent=2)+'\n')
    print(OUT)


if __name__=='__main__':
    main()
