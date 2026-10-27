"""Export slide tables to dated PNGs and replace visible tables with image links.

Table cells are retained in media/YYYY-MM-DD/tables.json for regeneration after
the Markdown tables have been replaced. Edit that JSON to revise an exported
table, then rerun this command with --slides pointing to the same document.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from media_bundle import write_bundle

ROOT = Path(__file__).resolve().parents[4]
FONTS = Path('/usr/share/fonts/truetype/dejavu')


def render(rows, fractions, output):
    full_width, margin, pad, line_height = 2600, 32, 38, 68
    fonts = [ImageFont.truetype(str(FONTS / name), 48) for name in
             ['DejaVuSans-Bold.ttf', 'DejaVuSans.ttf']]
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    widths = [round((full_width - margin * 2) * f) for f in fractions]
    widths[-1] = full_width - margin * 2 - sum(widths[:-1])
    layout, heights = [], []
    for row_index, row in enumerate(rows):
        cells = []
        font = fonts[0 if row_index == 0 else 1]
        for column, text in enumerate(row):
            text = text.replace('`', '').replace('**', '')
            lines, current = [], ''
            for word in text.split():
                if measure.textlength(word, font=font) > widths[column] - 2 * pad:
                    raise ValueError(f'Table column too narrow: {word}')
                candidate = f'{current} {word}'.strip()
                if measure.textlength(candidate, font=font) <= widths[column] - 2 * pad:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            cells.append(lines + [current] if current else lines or [''])
        layout.append(cells)
        heights.append(max(124, max(map(len, cells)) * line_height + 60))
    image = Image.new('RGB', (full_width, sum(heights) + margin * 2), 'white')
    draw, y = ImageDraw.Draw(image), margin
    for row_index, cells in enumerate(layout):
        background = '#17324d' if row_index == 0 else '#ffffff' if row_index % 2 else '#f0f5fa'
        color = 'white' if row_index == 0 else '#172b3d'
        font = fonts[0 if row_index == 0 else 1]
        draw.rectangle((margin, y, full_width - margin, y + heights[row_index]), fill=background)
        x = margin
        for column, lines in enumerate(cells):
            for number, line in enumerate(lines):
                px = x + pad
                py = y + (heights[row_index] - len(lines) * line_height) / 2 + number * line_height
                box = draw.textbbox((px, py), line, font=font, anchor='lt')
                if box[2] > x + widths[column] - 16 or box[3] > y + heights[row_index] - 12:
                    raise ValueError(f'Clipped table text: {line}')
                draw.text((px, py), line, font=font, fill=color, anchor='lt')
            if column:
                draw.line((x, y, x, y + heights[row_index]), fill='#b9c9d7', width=2)
            x += widths[column]
        if row_index:
            draw.line((margin, y, full_width - margin, y), fill='#d4dee8', width=2)
        y += heights[row_index]
    draw.rectangle((margin, margin, full_width - margin, image.height - margin), outline='#b9c9d7', width=2)
    image.save(output, optimize=True, dpi=(200, 200))
    with Image.open(output) as check:
        check.load()
        assert check.size == image.size and check.mode == 'RGB'
    return list(image.size)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slides', type=Path, required=True)
    args = parser.parse_args()
    document = args.slides.resolve(strict=True)
    date = re.match(r'\d{4}-\d{2}-\d{2}', document.name)
    if not date:
        raise ValueError('Slide document must start with YYYY-MM-DD')
    date = date.group()
    output = document.parent / 'media' / date
    output.mkdir(parents=True, exist_ok=True)
    source = document.read_text()
    slide, title, blocks, block = 0, '', [], []
    for line in source.splitlines() + ['']:
        if not line.startswith('|') and block:
            raw = '\n'.join(block)
            cells = [[cell.strip() for cell in row.strip().strip('|').split('|')] for row in block]
            if len(cells) < 2 or not all(re.fullmatch(r':?-+:?', cell) for cell in cells[1]):
                raise ValueError('Invalid Markdown table')
            rows = cells[:1] + cells[2:]
            columns = len(rows[0])
            if not all(len(row) == columns for row in rows):
                raise ValueError('Inconsistent table columns')
            fractions = {2: [.4, .6], 3: [.40, .30, .30], 4: [.23, .43, .17, .17]}[columns]
            if slide == 6 and columns == 3:
                fractions = [.28, .36, .36]
            slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
            number = sum(row['slide'] == slide for row in blocks) + 1
            suffix = '' if number == 1 else f'_{number}'
            blocks.append({'slide': slide, 'title': title, 'rows': rows, 'column_fractions': fractions,
                           'file': f'slide{slide:02d}_{slug}_table{suffix}.png', 'markdown_source': raw})
            block = []
        if line.startswith('## Slide '):
            match = re.match(r'## Slide (\d+) — Part2Action: (.+)', line)
            if not match:
                raise ValueError('Unexpected slide title')
            slide, title = int(match.group(1)), match.group(2)
        if line.startswith('|'):
            block.append(line)
    sources = output / 'tables.json'
    if blocks:
        sources.write_text(json.dumps(blocks, indent=2) + '\n')
    else:
        blocks = json.loads(sources.read_text())
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'meeting_date': date, 'assets': [], 'slide_mapping': {}, 'generators': {}}
    records = []
    for row in blocks:
        file = output / row['file']
        dimensions = render(row['rows'], row['column_fractions'], file)
        records.append({'slides': [row['slide']], 'file': row['file'],
                        'title': f'Slide {row["slide"]} — {row["title"]}',
                        'kind': 'table_png', 'markdown_cells': row['rows'],
                        'dimensions': dimensions, 'bytes': file.stat().st_size,
                        'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
        files = manifest['slide_mapping'].setdefault(str(row['slide']), [])
        if row['file'] not in files:
            files.append(row['file'])
        link = f'media/{date}/{row["file"]}'
        replacement = (f'![Slide {row["slide"]} table: {row["title"]}]({link})\n\n'
                       f'Use this image: [{row["file"]}]({link}).')
        if row['markdown_source'] in source:
            source = source.replace(row['markdown_source'], replacement, 1)
        elif link not in source:
            raise ValueError(f'Table image not referenced by slide document: {link}')
        print(row['file'], dimensions, flush=True)
    manifest['table_images'] = records
    manifest['supporting_files'] = ['tables.json']
    manifest['generators']['tables'] = str(Path(__file__).resolve().relative_to(ROOT))
    preview_link = f'Slide media: [PNG preview](media/{date}/index.html) · [download all table images](media/{date}/{date}_slide_media.zip).'
    if preview_link not in source:
        heading, remainder = source.split('\n', 1)
        source = heading + '\n\n' + preview_link + '\n' + remainder
    document.write_text(source)
    print(json.dumps(write_bundle(output, manifest), indent=2))


if __name__ == '__main__':
    main()
