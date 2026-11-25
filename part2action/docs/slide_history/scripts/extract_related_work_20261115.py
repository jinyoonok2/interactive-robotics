"""Extract cited paper figures for the November 15 presentation.

Uses the versioned local PDFs, Ghostscript, and Pillow. Coordinates are PDF
points measured from the page's top-left corner. Captions stay editable in the
slide document; the diagram contents are reproduced without annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageOps, PngImagePlugin


ROOT = Path(__file__).resolve().parents[4]
HISTORY = ROOT / 'part2action/docs/slide_history'
PAPERS = HISTORY / 'papers/2026-11-15'
DEFAULT_OUT = HISTORY / 'media/2026-11-15/related_work'

FIGURES = (
    dict(file='ghost_fig01_overview.png', pdf='GHOST_RSS2026.pdf',
         figure='1', page=1, bbox_points=(97, 194, 520, 518),
         title='GHOST high-level goal prediction and low-level execution',
         credit='Krishna et al., GHOST, RSS 2026, Figure 1',
         source_url='https://arxiv.org/pdf/2606.10025v1#page=1'),
    dict(file='ghost_fig06_goal_conditioned_policy.png', pdf='GHOST_RSS2026.pdf',
         figure='6', page=5, bbox_points=(307, 329, 569, 506),
         title='GHOST goal-conditioned visual diffusion policy',
         credit='Krishna et al., GHOST, RSS 2026, Figure 6',
         source_url='https://arxiv.org/pdf/2606.10025v1#page=5'),
    dict(file='diffuser_actor_fig02_overview.png', pdf='3D_Diffuser_Actor_CoRL2024.pdf',
         figure='2', page=4, bbox_points=(106, 60, 508, 325),
         title='3D Diffuser Actor overview, denoising, and real-world examples',
         credit='Ke et al., 3D Diffuser Actor, CoRL 2024, Figure 2',
         source_url='https://arxiv.org/pdf/2402.10885v3#page=4'),
    dict(file='diffuser_actor_fig02ab_tokens_and_denoising.png',
         pdf='3D_Diffuser_Actor_CoRL2024.pdf', figure='2(a-b)', page=4,
         bbox_points=(106, 60, 508, 253),
         title='3D Diffuser Actor token architecture and denoising panels',
         credit='Ke et al., 3D Diffuser Actor, CoRL 2024, Figure 2(a-b)',
         source_url='https://arxiv.org/pdf/2402.10885v3#page=4'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 144:
        raise ValueError('Use at least 144 DPI for readable paper diagrams')
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to replace an existing figure extraction')
    source_manifest = json.loads((PAPERS / 'data/references.json').read_text())
    expected = {Path(row['local_pdf']).name: row for row in source_manifest['papers']}
    for name in {row['pdf'] for row in FIGURES}:
        assert sha(PAPERS / name) == expected[name]['sha256'], name

    records = []
    with tempfile.TemporaryDirectory(prefix='p2a_related_work_pages_') as tmp:
        pages = {}
        for row in FIGURES:
            key = (row['pdf'], row['page'])
            if key in pages:
                continue
            rendered = Path(tmp) / f'page_{len(pages):02d}.png'
            command = ['gs', '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE',
                       '-sDEVICE=png16m', '-dTextAlphaBits=4', '-dGraphicsAlphaBits=4',
                       f'-r{args.dpi}', f'-dFirstPage={row["page"]}',
                       f'-dLastPage={row["page"]}', f'-sOutputFile={rendered}',
                       str(PAPERS / row['pdf'])]
            subprocess.run(command, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=60)
            pages[key] = rendered
        out.mkdir(parents=True, exist_ok=True)
        for row in FIGURES:
            with Image.open(pages[(row['pdf'], row['page'])]) as page:
                box = tuple(round(value * args.dpi / 72) for value in row['bbox_points'])
                assert 0 <= box[0] < box[2] <= page.width
                assert 0 <= box[1] < box[3] <= page.height
                crop = page.crop(box).convert('RGB')
            ink = ImageChops.difference(crop, Image.new('RGB', crop.size, 'white')).getbbox()
            assert ink is not None, row['file']
            assert ink[0] > 0 and ink[1] > 0 and ink[2] < crop.width and ink[3] < crop.height, row['file']
            image = ImageOps.expand(crop, border=16, fill='white')
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text('Source', row['source_url'])
            metadata.add_text('Credit', row['credit'])
            metadata.add_text('Extraction', 'PDF figure crop; no diagram annotations; 16-pixel white border')
            destination = out / row['file']
            image.save(destination, dpi=(args.dpi, args.dpi), pnginfo=metadata, optimize=True)
            with Image.open(destination) as check:
                check.load()
                assert check.size == image.size
            records.append({**row, 'bbox_origin': 'top-left PDF page', 'dpi': args.dpi,
                            'dimensions': list(image.size), 'bytes': destination.stat().st_size,
                            'sha256': sha(destination),
                            'source_pdf': str((PAPERS / row['pdf']).relative_to(ROOT)),
                            'source_pdf_sha256': sha(PAPERS / row['pdf'])})
            print(row['file'], image.size, flush=True)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'generator': str(Path(__file__).resolve().relative_to(ROOT)),
                'generator_sha256': sha(Path(__file__).resolve()),
                'figure_numbers': 'Numbers in the downloaded PDFs; GHOST HTML numbers differ',
                'extraction': 'Diagram crops without original prose captions; credit is retained in slides and PNG metadata',
                'figures': records}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
