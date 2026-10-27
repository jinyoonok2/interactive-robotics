"""Draw the simplified unified MLP architecture for presentation slides."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from PIL import Image
import yaml

from media_bundle import write_bundle

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT/'part2action/configs/architecture_update/hierarchical_world_geometry_10epoch_normalized_global_contact.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path,
                        default=ROOT/'part2action/docs/slide_history/media/2026-10-12')
    args = parser.parse_args()
    output = args.out_dir.resolve()
    manifest_path = output/'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Build the slide media first so its manifest exists.')
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg['data']['n_obs_steps'] == 2 and cfg['data']['action_chunk'] == 8
    assert cfg['model']['action_head_type'] == 'mlp'
    assert cfg['model']['contact_conditioning_mode'] == 'normalized_global'
    assert cfg['model']['use_part_gate'] and not cfg['model']['use_skill_gate']
    assert not cfg['model']['use_near_contact_actions']

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(16, 9.4), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1600)
    ax.set_ylim(940, 0)
    ax.axis('off')
    navy, muted, border = '#17324d', '#536779', '#b9c9d7'
    blue_bg, teal_bg, teal = '#f0f5fa', '#e7f7f5', '#087f8c'
    text_boxes = []
    cell_text = []

    def text(x, y, label, size=16, weight='normal', color=navy, ha='center'):
        artist = ax.text(x, y, label, fontsize=size, fontweight=weight,
                         color=color, ha=ha, va='center', linespacing=1.35)
        text_boxes.append(artist)
        return artist

    def box(x, y, width, height, title, detail='', fill=blue_bg, color=navy):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle='round,pad=0,rounding_size=14',
                     edgecolor=border if color==navy else color,
                     linewidth=1.5, facecolor=fill))
        title_y = y+27 if detail.count('\n')>=2 else y+height/2-(18 if detail else 0)
        artist = text(x+width/2, title_y, title, 17, 'bold', color)
        cell_text.append((artist, (x, y, width, height)))
        if detail:
            artist = text(x+width/2, y+height/2+18, detail, 13.5, color=color)
            cell_text.append((artist, (x, y, width, height)))

    def arrow(start, end, color=muted, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=18,
                     linewidth=1.8, color=color, linestyle=(0,(5,4)) if dashed else '-'))

    text(60, 31, 'Part2Action: Unified architecture', 23, 'bold', ha='left')
    inputs = [(60, 'RGB history', 'Two recent frames'),
              (435, 'Task instruction', 'Language describing the task'),
              (810, 'Scene point cloud', 'Visible scene geometry'),
              (1185, 'Robot state', 'TCP pose + finger opening')]
    for x, title, detail in inputs:
        box(x, 83, 355, 100, title, detail, fill='white')
        arrow((x+177.5, 183), (x+177.5, 236))

    ax.add_patch(FancyBboxPatch((60, 236), 1480, 169,
                 boxstyle='round,pad=0,rounding_size=14',
                 edgecolor=border, linewidth=1.5, facecolor=blue_bg))
    text(800, 267, 'Shared perception and skill context', 18, 'bold')
    for x, title, detail in [
        (310, 'Vision', 'Frozen DINOv2 + temporal encoder\nPredicted visual part grounding'),
        (800, 'Language and skills', 'Frozen Flan-T5 + skill plan\nPhase selector + termination head'),
        (1290, 'Scene geometry', 'Point-cloud encoder\nScene features + TCP geometry'),
    ]:
        text(x, 307, title, 16, 'bold')
        text(x, 355, detail, 13.5)
    # These boxes group interacting modules; they are not independent chains.
    text(800, 390, 'Skill selection uses visual and scene context; fusion combines the selected skill with observation features.',
         12.5, color=muted)

    arrow((445, 405), (445, 475))
    arrow((1245, 405), (1245, 475))
    box(160, 475, 570, 96, 'Base action features',
        'Fusion + finger opening + preserved control bias')
    box(1035, 475, 420, 96, 'Contact prediction', 'Predicted 3D point on the scene')
    arrow((445, 571), (445, 644))
    arrow((1245, 571), (1245, 611), color=teal)

    box(1000, 611, 490, 132, 'Contact conditioning',
        '(predicted contact − TCP) / 0.1 m\nProjection → remove bias\nNormalize × learned global weight',
        fill=teal_bg, color=teal)
    # Optional forward contribution, not a distance gate or phase-based switch.
    arrow((1000, 677), (474, 677), color=teal, dashed=True)
    text(742, 649, 'Optional contact contribution', 14, color=teal)
    ax.add_patch(Circle((445, 677), 28, facecolor='white', edgecolor=navy, linewidth=1.6))
    text(445, 675, '+', 25)
    arrow((445, 705), (445, 754))
    box(160, 754, 570, 96, 'MLP action head',
        '8 actions × [XYZ, wrist rotation, gripper]')
    arrow((445, 850), (445, 883))
    text(445, 909, 'Execute the first action → observe again → repeat', 14.5, 'bold')

    text(1245, 802, 'Unified contact: contribution enabled\nControl / timing runs: contribution set to zero\nContact prediction remains active in both',
         14, color=teal)
    text(1245, 898, 'Part/contact labels supervise training only.\nAuxiliary heatmap and approach outputs are omitted.',
         12.5, color=muted)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bounds = fig.bbox
    for artist in text_boxes:
        bbox = artist.get_window_extent(renderer)
        if not (figure_bounds.x0<=bbox.x0 and bbox.x1<=figure_bounds.x1
                and figure_bounds.y0<=bbox.y0 and bbox.y1<=figure_bounds.y1):
            raise ValueError(f'Diagram text clipped: {artist.get_text()}')
    for artist, (x, y, width, height) in cell_text:
        bbox = artist.get_window_extent(renderer)
        a = ax.transData.transform((x+10, y+6))
        b = ax.transData.transform((x+width-10, y+height-6))
        if not (a[0]<=bbox.x0 and bbox.x1<=b[0] and b[1]<=bbox.y0 and bbox.y1<=a[1]):
            raise ValueError(f'Diagram text exceeds its box: {artist.get_text()}')
    png, svg = output/'slide04_unified_architecture.png', output/'slide04_unified_architecture.svg'
    fig.savefig(png, dpi=200, facecolor='white')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)
    with Image.open(png) as image:
        image.load()
        dimensions = list(image.size)
    record = {
        'slides': [1], 'file': png.name, 'svg': svg.name,
        'title': 'Slide 1 — Unified architecture', 'kind': 'architecture_diagram',
        'description': 'Simplified forward dataflow for the current two-frame unified MLP. The teal contact contribution is enabled for normalized unified contact and disabled for unified control and timing runs. The skill hierarchy, visual part grounding and contact prediction remain active.',
        'dimensions': dimensions, 'bytes': png.stat().st_size,
        'sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
        'svg_sha256': hashlib.sha256(svg.read_bytes()).hexdigest(),
        'source_config': str(CONFIG.relative_to(ROOT)),
        'source_modules': ['part2action/models/part2action_model.py', 'part2action/models/heads.py'],
        'generator': str(Path(__file__).resolve().relative_to(ROOT)),
        'conditioning_expression': 'features + P(0) + weight * LayerNorm(P((contact - TCP)/0.1) - P(0))',
        'simplification': 'Shared encoders, hierarchy, and fusion are grouped rather than drawing all internal attention connections. Explicit contact geometry is detached from action-loss gradients. This is a forward architecture diagram, not a performance claim.',
    }
    manifest = json.loads(manifest_path.read_text())
    manifest['diagram_images'] = [record]
    for slide in record['slides']:
        files = manifest['slide_mapping'][str(slide)]
        for name in [png.name, svg.name]:
            if name not in files:
                files.append(name)
    manifest.get('unavailable', {}).pop('4', None)
    manifest.setdefault('generators', {})['architecture'] = record['generator']
    print(json.dumps(write_bundle(output, manifest), indent=2))


if __name__ == '__main__':
    main()
