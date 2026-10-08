"""Render synchronized threshold comparisons from the offline review gallery."""
import argparse
import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, default=ROOT/'results/diagnostics/contact_label_audit/review/index.html')
    parser.add_argument('--out', type=Path, default=ROOT/'results/diagnostics/contact_label_audit/gate_gifs')
    args = parser.parse_args()
    payload = args.gallery.read_text().split('const data=', 1)[1].split(';let idx=', 1)[0]
    examples = json.loads(payload)
    args.out.mkdir(parents=True, exist_ok=True)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def font(size):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            return ImageFont.load_default()
    title_font, label_font, small_font = font(20), font(18), font(14)
    records = []
    for example in examples:
        timeline = {row['t']: row for row in example['timeline']}
        frames = []
        statuses = []
        for sample in example['frames']:
            row = timeline[sample['t']]
            distance = row['legacy_cm']
            canvas = Image.new('RGB', (800, 940), '#111923')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 10), 'Contact distance gate: 3 / 5 / 7 / 10 cm', font=title_font, fill='white')
            draw.text((16, 42), example['instruction'], font=label_font, fill='white')
            draw.text((16, 69), f"{example['object']} / {example['demo']} | frame {sample['t']} | legacy derived label", font=small_font, fill='#c7d6e2')
            image = Image.open(io.BytesIO(base64.b64decode(sample['images'][1]))).convert('RGB')
            image = image.resize((384, 384), Image.Resampling.LANCZOS)
            state = {}
            for index, threshold in enumerate([3, 5, 7, 10]):
                x, y = 8+(index % 2)*400, 100+(index // 2)*400
                active = distance is not None and distance <= threshold
                status = 'UNAVAILABLE' if distance is None else 'ACTIVE' if active else 'INACTIVE'
                color = '#43dc94' if active else '#a3b4c4' if distance is not None else '#ffc76b'
                canvas.paste(image, (x, y))
                draw.rectangle((x, y, x+383, y+57), fill='#182532', outline=color, width=3)
                draw.text((x+10, y+5), f'{threshold} cm: {status}', font=label_font, fill=color)
                dist_text = 'unavailable' if distance is None else f'{distance:.2f} cm'
                draw.text((x+10, y+32), f'TCP to target: {dist_text}', font=small_font, fill='white')
                state[str(threshold)] = status
            draw.text((16, 904), 'Yellow: TCP | Red: target | ACTIVE if distance <= threshold', font=small_font, fill='white')
            draw.text((16, 925), 'Recorded demo; diagnostic gate only. Not predicted actions or confirmed contact.', font=small_font, fill='#c7d6e2')
            frames.append(canvas)
            statuses.append(dict(frame=sample['t'], distance_cm=distance, thresholds=state))
        filename = f"{example['object']}_{example['kind']}_{example['demo']}_thresholds.gif"
        durations = [170]*len(frames)
        durations[-1] = 1000
        frames[0].save(args.out/filename, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
        # Export a slide-friendly still near the estimated interaction event.
        event_index = min(range(len(example['frames'])), key=lambda i: abs(example['frames'][i]['t']-example['skill_event']))
        frames[event_index].save(args.out/filename.replace('.gif', '_event.png'))
        records.append(dict(file=filename, object=example['object'], demo=example['demo'], skill=example['kind'], instruction=example['instruction'], label_mode='legacy', thresholds_cm=[3,5,7,10], frame_statuses=statuses))
        print(filename, flush=True)
    (args.out/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
