"""Write the shared preview and portable archive for slide media."""
from pathlib import Path
import hashlib
import html
import json
import zipfile


def write_bundle(root: Path, manifest: dict) -> dict:
    root = Path(root)
    date = manifest['meeting_date']
    archive_name = f'{date}_slide_media.zip'
    cards = []
    for row in manifest.get('assets', []):
        file = root / row['file']
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Media differs from recorded provenance: {file}')
        reference = row.get('ground_truth_reference') or row.get('provenance', {}).get('ground_truth_reference')
        caption = row['description']
        if reference:
            caption += (f' Expert reference: {reference["demo_key"]}, source episode '
                        f'{reference["episode_id"]}; {reference["match_type"]}.')
        cards.append(
            f'<article><h2>{html.escape(row["title"])}</h2><p>{html.escape(caption)}</p>'
            f'<p>Panels: {html.escape(" | ".join(row["panels"]))}</p>'
            f'<img src="{row["file"]}" loading="lazy" alt="{html.escape(row["title"], quote=True)}">'
            f'<p><a download href="{row["file"]}">Download GIF</a> · '
            f'{row["bytes"]/1024**2:.1f} MiB · {row["dimensions"][0]}×{row["dimensions"][1]}</p></article>')
    table_cards = []
    for row in manifest.get('table_images', []):
        file = root / row['file']
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Table image differs from recorded provenance: {file}')
        table_cards.append(
            f'<article><h2>{html.escape(row["title"])}</h2>'
            '<p>Exact table values from the slide document. High-resolution PNG with a white background.</p>'
            f'<img src="{row["file"]}" loading="lazy" alt="{html.escape(row["title"], quote=True)}">'
            f'<p><a href="{row["file"]}" download>Download PNG</a> · '
            f'{row["dimensions"][0]}×{row["dimensions"][1]}</p></article>')
    table_link = '<a href="#tables">View PNG tables</a> · ' if table_cards else ''
    tables = '<section id="tables"><h1>Table images</h1>' + ''.join(table_cards) + '</section>' if table_cards else ''
    diagram_cards = []
    for row in manifest.get('diagram_images', []):
        for name, checksum in [(row['file'], row['sha256']), (row['svg'], row['svg_sha256'])]:
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f'Diagram differs from recorded provenance: {name}')
        diagram_cards.append(
            f'<article><h2>{html.escape(row["title"])}</h2><p>{html.escape(row["description"])}</p>'
            f'<img src="{row["file"]}" alt="{html.escape(row["title"], quote=True)}">'
            f'<p><a href="{row["file"]}" download>Download PNG</a> · '
            f'<a href="{row["svg"]}" download>Download editable SVG</a></p></article>')
    diagrams = '<section id="diagrams"><h1>Architecture diagram</h1>' + ''.join(diagram_cards) + '</section>' if diagram_cards else ''
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Part2Action {date} slide media</title>'
        '<style>body{font:16px system-ui;background:#17191d;color:#eee;max-width:2000px;margin:auto;padding:24px}'
        'article{background:#24272d;padding:18px;margin:22px 0}img{display:block;max-width:100%;height:auto}'
        'a{color:#9bd0ff}h2{font-size:21px}</style>'
        f'<h1>Part2Action {date} slide media</h1>'
        '<p>Download each GIF, table, or diagram for its slide, or download the complete portable bundle.</p>'
        f'<p>{table_link}<a href="{archive_name}">Download all media</a> · '
        '<a href="manifest.json">Source provenance and slide mapping</a></p>'
        f'<p>{html.escape(manifest.get("reference_note", ""))}</p>'
        + diagrams + ''.join(cards) + tables + '</html>')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'index.html').write_text(page)
    temporary = root / (archive_name + '.tmp')
    included = {'index.html', 'manifest.json'}
    included.update(manifest.get('supporting_files', []))
    included.update(row['file'] for row in manifest.get('assets', []))
    included.update(row['file'] for row in manifest.get('table_images', []))
    for row in manifest.get('diagram_images', []):
        included.update([row['file'], row['svg']])
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(included):
            if Path(name).name != name:
                raise ValueError(f'Slide media must use flat filenames: {name}')
            bundle.write(root / name, name)
    archive = root / archive_name
    temporary.replace(archive)
    return {'folder': str(root), 'gifs': len(manifest.get('assets', [])),
            'table_pngs': len(manifest.get('table_images', [])),
            'diagrams': len(manifest.get('diagram_images', [])),
            'zip_size_mib': round(archive.stat().st_size / 1024**2, 1)}
