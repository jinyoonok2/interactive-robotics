"""Render October 12 Markdown tables as PNGs and update the media bundle."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import argparse
import json
import re
from media_bundle import write_bundle

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT/'part2action/docs/slide_history/2026-10-12_part2action_meeting_status.md'
OUT = DOC.parent/'media/2026-10-12'
font_dir = Path('/usr/share/fonts/truetype/dejavu')
body_font = ImageFont.truetype(str(font_dir/'DejaVuSans.ttf'), 48)
header_font = ImageFont.truetype(str(font_dir/'DejaVuSans-Bold.ttf'), 46)
code_font = ImageFont.truetype(str(font_dir/'DejaVuSansMono.ttf'), 44)
measure = ImageDraw.Draw(Image.new('RGB',(1,1)))
layouts = [
    (2, 'slide02_contact_localization_table.png', [.33,.42,.25], 'Slide 2 — Contact localization table'),
    (3, 'slide03_contact_strength_table.png', [.32,.20,.48], 'Slide 3 — Contact strength table'),
    (4, 'slide05_normalized_results_table.png', [.56,.22,.22], 'Slide 4 — Normalized policy results table'),
    (5, 'slide07_recording_order_table.png', [.30,.35,.35], 'Slide 5 — Recording order example table'),
    (5, 'slide08_action_alignment_table.png', [.24,.34,.42], 'Slide 5 — Action alignment table'),
    (6, 'slide06_timing_results_table.png', [.56,.22,.22], 'Slide 6 — Timing pilot results table'),
]


def width(text, font):
    return measure.textlength(text,font=font)


def wrap(text, font, limit):
    words = text.split()
    lines = []
    current = ''
    for word in words:
        trial = f'{current} {word}'.strip()
        if width(trial,font)<=limit:
            current=trial
        else:
            if current:
                lines.append(current)
                current=''
            if width(word,font)>limit:
                raise ValueError(f'Cell token exceeds its column: {word}')
            current=word
    if current:
        lines.append(current)
    return lines or ['']
def main():
    global DOC, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slides', type=Path, default=DOC)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()
    DOC = args.slides.resolve()
    OUT = args.out_dir.resolve() if args.out_dir else DOC.parent/'media/2026-10-12'
    if not (OUT/'manifest.json').is_file():
        raise FileNotFoundError('Build the slide GIF media first so its manifest and preview exist.')
    source = DOC.read_text()
    tables = []
    slide = 0
    block = []
    for line in source.splitlines()+['']:
        if line.startswith('## Slide '):
            slide=int(line.split()[2])
        if line.startswith('|'):
            block.append(line)
        elif block:
            tables.append((slide,'\n'.join(block)))
            block=[]
    if [s for s,_ in tables] != [row[0] for row in layouts]:
        raise ValueError('Slide tables differ from expected selection')

    records=[]
    for (slide, raw), (_, filename, fractions, title) in zip(tables, layouts):
        markdown_rows=[[cell.strip() for cell in line.strip().strip('|').split('|')] for line in raw.splitlines()]
        if not all(re.fullmatch(r':?-+:?',cell) for cell in markdown_rows[1]):
            raise ValueError('Unexpected Markdown table divider')
        markdown_rows = markdown_rows[:1]+markdown_rows[2:]
        rows=[[cell.replace('`','') for cell in row] for row in markdown_rows]
        full_width=2400
        margin=32
        inner_width=full_width-2*margin
        widths=[round(inner_width*f) for f in fractions]
        widths[-1]=inner_width-sum(widths[:-1])
        laid_out=[]
        heights=[]
        for r,row in enumerate(rows):
            cells=[]
            for c,text in enumerate(row):
                font=header_font if r==0 else code_font if '`' in markdown_rows[r][c] else body_font
                lines=wrap(text,font,widths[c]-84)
                cells.append((lines,font))
            height=max(118,max(len(lines) for lines,font in cells)*64+58)
            heights.append(height)
            laid_out.append(cells)
        height=sum(heights)+2*margin
        image=Image.new('RGB',(full_width,height),'white')
        draw=ImageDraw.Draw(image)
        y=margin
        text_boxes=[]
        for r,cells in enumerate(laid_out):
            x=margin
            background='#17324d' if r==0 else '#ffffff' if r%2 else '#f0f5fa'
            draw.rectangle((margin,y,full_width-margin,y+heights[r]),fill=background)
            for c,(lines,font) in enumerate(cells):
                color='white' if r==0 else '#172b3d'
                line_height=64
                baseline_top=y+(heights[r]-len(lines)*line_height)/2
                for j,line in enumerate(lines):
                    line_width=width(line,font)
                    px=x+42 if c==0 else x+(widths[c]-line_width)/2
                    py=baseline_top+j*line_height
                    draw.text((px,py),line,font=font,fill=color,anchor='lt')
                    bbox=draw.textbbox((px,py),line,font=font,anchor='lt')
                    if not (x+20<=bbox[0] and bbox[2]<=x+widths[c]-20 and y+10<=bbox[1] and bbox[3]<=y+heights[r]-10):
                        raise ValueError(f'Clipped cell at slide {slide}, row {r}, col {c}: {bbox}')
                    text_boxes.append({'row':r,'column':c,'text':line,'bbox':list(bbox)})
                if c>0:
                    draw.line((x,y,x,y+heights[r]),fill='#304a65' if r==0 else '#d4dee8',width=2)
                x+=widths[c]
            if r>0:
                draw.line((margin,y,full_width-margin,y),fill='#d4dee8',width=2)
            y+=heights[r]
        draw.rectangle((margin,margin,full_width-margin,height-margin),outline='#b9c9d7',width=2)
        file=OUT/filename
        image.save(file,optimize=True,dpi=(200,200))
        with Image.open(file) as check:
            check.load()
            if check.size!=(full_width,height) or check.mode!='RGB':
                raise ValueError('Unexpected PNG export')
        records.append({'slides':[slide],'file':filename,'title':title,'kind':'markdown_table_png',
                        'source_document':str(DOC.relative_to(ROOT)),'markdown_cells':markdown_rows,
                        'rendered_cells':rows,'dimensions':[full_width,height],'dpi':200,
                        'background':'white','bytes':file.stat().st_size,
                        'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'text_layout':text_boxes})
        link=f'Table image: [download PNG](media/2026-10-12/{filename}).'
        if filename not in source:
            source=source.replace(raw,raw+'\n\n'+link,1)
        print(filename,full_width,height,flush=True)

    source=source.replace('Slide GIFs: [preview and download]', 'Slide media: [preview and download]')
    source=source.replace('The media folder contains standalone copies and selected comparisons rendered\nfrom existing recordings, with source provenance in its manifest.',
                          'The media folder contains GIF comparisons and PNG tables, with source provenance\nin its manifest.')
    DOC.write_text(source)
    mp=OUT/'manifest.json'
    manifest=json.loads(mp.read_text())
    manifest['table_images']=records
    for slide, filename, _, _ in layouts:
        files=manifest['slide_mapping'][str(slide)]
        if filename not in files:
            files.append(filename)
    manifest.setdefault('generators', {}).update({
        'tables': str(Path(__file__).resolve().relative_to(ROOT)),
        'bundle': 'part2action/docs/slide_history/scripts/media_bundle.py',
    })
    print(json.dumps(write_bundle(OUT, manifest), indent=2))


if __name__ == '__main__':
    main()
