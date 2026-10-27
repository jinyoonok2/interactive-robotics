"""Add recorded TCP and derived contact geometry to the profile-page GIF."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'part2action'))
sys.path.insert(0, str(ROOT / 'part2action/scripts'))
from data.geometry import tcp_position_numpy
from data.targets import derive_contact_xyz, derive_skill_contact_target
from audit_contact_labels import _segments
from visualize_contact_projection import project

CONTACT = '#00ddff'  # Same as the skill-specific contact audit.
TCP = '#ffe25c'
PART = '#41dc64'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path,
                        default=ROOT / 'assets/part2action/part2action_bottle_neck.json')
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    output = args.metadata.parent / metadata['gif']
    indices = metadata['source_frames']
    crop = metadata['crop_xyxy']
    size = tuple(metadata['output_resolution'])
    sx, sy = size[0] / (crop[2] - crop[0]), size[1] / (crop[3] - crop[1])
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    def screen(xyz):
        uv = project(xyz, *metadata['source_rgb_resolution'])
        if uv is None:
            return None
        return ((uv[0] - crop[0]) * sx, (uv[1] - crop[1]) * sy)

    images, records = [], []
    with h5py.File(metadata['source_hdf5'], 'r') as handle:
        demo = handle['data'][metadata['demo']]
        actions = demo['actions'][:]
        poses = demo['obs/tcp_pose'][:]
        clouds = demo['obs/agentview_part_pcd'][:]
        start, end, instruction = _segments(demo['skill_instructions'][:])[0]
        target = derive_skill_contact_target(
            actions=actions, part_pcd=clouds, tcp_pose=poses, skill_kind='grasp_obj',
            segment_start=start, segment_end=end, current_t=start)
        event = int(target['contact_t'])
        if event < 0 or not instruction.lower().startswith('grasp '):
            raise ValueError('This thumbnail requires a grasp demonstration with a valid contact event')
        # Subsequent move skills have no contact labels. Keep this illustration
        # inside the audited grasp segment rather than inventing missing targets.
        indices = [t for t in indices if start <= t <= end]
        if not indices:
            raise ValueError('No requested frames fall inside the grasp segment')
        if indices[-1] != end:
            indices.append(end)
        durations = [100] * len(indices)
        durations[0] = 300
        durations[-1] = 1000
        for t in indices:
            rgb = np.asarray(demo['obs/agentview_rgb'][t]).copy()
            mask = np.asarray(demo['obs/agentview_part_mask'][t]).squeeze() > 0
            rgb[mask] = (.65 * rgb[mask] + .35 * np.array([65, 220, 100])).astype(np.uint8)
            image = Image.fromarray(rgb).crop(tuple(crop)).resize(size, Image.Resampling.LANCZOS)
            contact, valid = derive_contact_xyz(clouds, poses, event, current_t=t)
            contact_uv = screen(contact) if valid else None
            tcp_uv = screen(tcp_position_numpy(poses[t]))
            draw = ImageDraw.Draw(image)
            # Recorded recent TCP history, using the same 12-frame window as the audit.
            trail = [screen(tcp_position_numpy(poses[k])) for k in range(max(start, t - 12), t + 1)]
            trail = [uv for uv in trail if uv is not None]
            if len(trail) > 1:
                draw.line(trail, fill='#101820', width=4)
                draw.line(trail, fill=TCP, width=2)
            for uv, color, radius in [(tcp_uv, TCP, 8), (contact_uv, CONTACT, 5)]:
                if uv is None:
                    continue
                x, y = uv
                draw.ellipse((x-radius-2, y-radius-2, x+radius+2, y+radius+2),
                             outline='#101820', width=4)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)
                if color == CONTACT:
                    draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            # Compact legend remains inside the single scene, with no diagnostic panels.
            draw.rounded_rectangle((10, size[1]-36, 384, size[1]-6), radius=8, fill='#101820')
            for x, label, color in [(23, 'Contact target', CONTACT), (178, 'TCP', TCP),
                                     (247, 'Target part', PART)]:
                y = size[1]-21
                draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
                draw.text((x+11, y-9), label, fill='white', font=font)
            images.append(image)
            records.append({'source_frame': t, 'contact_valid': bool(valid),
                            'contact_world_xyz_m': contact.tolist() if valid else None,
                            'contact_output_xy_px': contact_uv, 'tcp_output_xy_px': tcp_uv})

    palette_samples = Image.new('RGB', (1200, ((len(images) + 9) // 10) * 80))
    for i, image in enumerate(images):
        palette_samples.paste(image.resize((120, 80)), ((i % 10) * 120, (i // 10) * 80))
    palette = palette_samples.quantize(colors=251, method=Image.Quantize.MEDIANCUT)
    # Tiny markers would otherwise lose their vivid colors to the much larger
    # brown table/background during GIF palette allocation.
    reserved = [0, 221, 255, 255, 226, 92, 65, 220, 100, 16, 24, 32, 255, 255, 255]
    palette.putpalette(palette.getpalette()[:251 * 3] + reserved)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    temporary = output.with_name(output.name + '.tmp')
    frames[0].save(temporary, format='GIF', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    # Decode the complete animation before replacing the existing asset.
    with Image.open(temporary) as gif:
        duration = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
            duration += gif.info['duration']
        if gif.size != size or duration != sum(durations) or gif.info['loop'] != 0:
            raise ValueError('Exported GIF dimensions, playback duration or loop setting differ')
        decoded_frames = gif.n_frames
    temporary.replace(output)
    metadata.pop('poster', None)
    metadata.update(
        bytes=output.stat().st_size, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        decoded_frames=decoded_frames,
        source_frames=indices, frame_duration_ms=durations, duration_seconds=duration / 1000,
        description='Franka Panda approaching a bottle neck and closing its gripper in PartGym.',
        selection=f'Start at source frame {indices[0]} and stop at the grasp segment end ({end}); subsequent move skills do not provide contact labels.',
        suggested_alt='A Franka Panda approaches and grasps a bottle neck. Cyan marks the derived contact target, yellow marks the recorded TCP and recent path, and green highlights the dataset target part.',
        overlay={
            'contact_color': CONTACT, 'tcp_color': TCP, 'part_color': PART,
            'contact_kind': 'Derived skill-specific demonstration label; not a learned model prediction',
            'contact_method': target['method'], 'contact_event_frame': event,
            'grasp_segment': [start, end],
            'projection': 'Same fixed-camera projection and world geometry as the contact audit',
            'scope': 'Only the initial grasp segment; no target extrapolation into later skills',
            'tcp_trail': 'Recorded measured TCP positions over the preceding 12 source frames',
            'part_mask': 'Recorded agentview_part_mask, tinted green', 'frames': records},
        exporter=str(Path(__file__).resolve().relative_to(ROOT)))
    args.metadata.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'gif': str(output), 'frames': decoded_frames, 'duration_seconds': duration / 1000,
                      'size_mib': round(output.stat().st_size / 1024**2, 2),
                      'valid_contacts': sum(row['contact_valid'] for row in records)}))


if __name__ == '__main__':
    main()
