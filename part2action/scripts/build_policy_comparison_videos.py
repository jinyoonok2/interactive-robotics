"""Shared multi-panel renderer and compatibility entry point for GIF reviews."""
import argparse
import json
import textwrap
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from visualize_contact_projection import project


COLORS = {"prediction": (255, 180, 35), "truth": (70, 225, 90)}  # BGR


def text(frame, label, x, y, scale=.55, color=(235, 235, 235)):
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def annotated_panel(frame, event):
    result = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_LINEAR)
    alignment = event.get("initial_target_part_alignment", {}) if event else {}
    if alignment.get("supported"):
        points = {}
        for name, field in [("prediction", "predicted_contact_xyz"), ("truth", "nearest_part_to_contact_xyz")]:
            uv = project(alignment[field], 480, 480)
            if uv is not None and 0 <= uv[0] < 480 and 0 <= uv[1] < 480:
                points[name] = tuple(int(round(v)) for v in uv)
        if len(points) == 2:
            cv2.line(result, points["prediction"], points["truth"], (255, 255, 255), 1, cv2.LINE_AA)
        for name, uv in points.items():
            cv2.circle(result, uv, 7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(result, uv, 5, COLORS[name], 2, cv2.LINE_AA)
    return result, alignment


def render_pair(left, right, output, speed, show_action_state=False):
    from ground_truth_reference import GroundTruthReferences
    cache = Path(output).parent.parent / 'ground_truth'
    truth = GroundTruthReferences(cache).trial(left)
    return render_panels([truth, left, right], ["GROUND-TRUTH REFERENCE", "UNIFIED CONTROL", "UNIFIED CONTACT"],
                         output, speed, show_action_state)


def render_panels(trials, labels, output, speed=.5, show_action_state=False):
    """Synchronize any number of recorded policies by simulation frame."""
    if not trials or len(trials) != len(labels) or speed <= 0:
        raise ValueError("Provide one label per trial and a positive playback speed")
    left = next((t for t in trials if not t.get('reference_kind')), trials[0])
    width = 480 * len(trials)
    captures = [cv2.VideoCapture(r["rollout_video"]) for r in trials]
    fps = [c.get(cv2.CAP_PROP_FPS) for c in captures]
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]
    if not all(c.isOpened() for c in captures) or min(counts) < 1 or fps[0] <= 0 or any(abs(fps[0] - f) > .001 for f in fps):
        for cap in captures:
            cap.release()
        raise ValueError("Input videos must be decodable and use matched frame rates")
    events = [{e["inference_step"]: e for e in r.get("phase_events", [])} for r in trials]
    writer = imageio_ffmpeg.write_frames(str(output), (width, 720), fps=fps[0] * speed,
                                       codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                       output_params=["-crf", "20", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"])
    writer.send(None)
    last = [None] * len(trials)
    try:
        for index in range(max(counts)):
            canvas = np.full((720, width, 3), 22, dtype=np.uint8)
            title = f"{left['obj_class']} | task {left['task_type']} | trial {left['trial_index']} | episode {left['info']['ep_id']}"
            text(canvas, title, 12, 22, .45 if len(trials) == 1 else .55)
            for line_index, line in enumerate(textwrap.wrap(left["instruction"], width=max(45, (width-24)//10))[:2]):
                text(canvas, line, 12, 45 + line_index * 20, .52)
            for column, (cap, trial) in enumerate(zip(captures, trials)):
                x = column * 480
                if index < counts[column]:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Unexpected decode failure at frame {index}")
                    last[column] = frame
                label = labels[column].upper()
                is_reference = trial.get('reference_kind') == 'expert_demonstration'
                outcome = "EXPERT DEMO" if is_reference else "SUCCESS" if trial["success"] else "FAILURE"
                short_label = label if len(label) <= 40 else label[:37] + '...'
                heading = f"{short_label} | {outcome}"
                heading_width = cv2.getTextSize(heading, cv2.FONT_HERSHEY_SIMPLEX, .57, 1)[0][0]
                text(canvas, heading, x + 12, 103, min(.57, .57*456/max(1, heading_width)))
                event = events[column].get(index) if index < counts[column] else None
                panel, alignment = annotated_panel(last[column], event)
                canvas[120:600, x:x + 480] = panel
                if is_reference:
                    reference = trial['ground_truth_reference']
                    text(canvas, f"{reference['demo_key']} | source episode {reference['episode_id']}", x+12, 624, .5)
                    match = 'Same evaluation episode' if reference['match_type'] == 'exact_episode' else 'Same skill chain; different episode'
                    text(canvas, match, x+12, 647, .48)
                    instructions = trial['reference_instructions']
                    instruction = instructions[min(index, len(instructions)-1)]
                    text(canvas, instruction[:62], x+12, 670, .43)
                    instance = 'same object' if reference['same_object_instance'] else 'different object instance'
                    text(canvas, f"Green tint: demo target | {instance}", x+12, 691, .43)
                    continue
                if alignment.get("supported"):
                    text(canvas, f"Contact -> true part: {100 * alignment['predicted_contact_to_part_min_distance_m']:.2f} cm", x + 12, 624)
                    text(canvas, f"TCP -> true part: {100 * alignment['tcp_to_part_min_distance_m']:.2f} cm", x + 12, 647)
                    text(canvas, f"Command -> true part: {100 * alignment['action_target_to_part_min_distance_m']:.2f} cm", x + 12, 670)
                    if show_action_state and index < len(trial['diagnostics']):
                        from scipy.spatial.transform import Rotation
                        d = trial['diagnostics'][index]
                        error = (Rotation.from_euler('xyz', trial['actions'][index][3:6]).inv()*Rotation.from_quat(d['tcp_quat'])).magnitude()*180/np.pi
                        label = f"Finger cmd/meas: {1000*trial['actions'][index][6]:.1f}/{1000*d['measured_finger_position_m']:.1f} mm | Wrist err: {error:.0f} deg"
                        text(canvas, label, x+12, 691, .43)
                    else:
                        text(canvas, f"Target part: {alignment.get('part_name', '')} | phase {event['phase']}", x + 12, 691, .48)
                else:
                    text(canvas, "Finished; final frame held" if index >= counts[column] else "No geometry diagnostic on this frame", x + 12, 628, .48)
            text(canvas, f"Step {index} | simulated time {index / fps[0]:.2f}s | playback {speed:g}x", 12, 714, .43)
            if width >= 960:
                text(canvas, "BLUE: predicted contact  GREEN: nearest true-part point", 425, 714, .43)
            writer.send(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
        for cap in captures:
            cap.release()
    return {"frames": max(counts), "input_fps": fps[0], "output_fps": fps[0] * speed,
            "input_frame_counts": counts}


def key(trial):
    return trial["obj_class"], str(trial["task_type"]), trial["trial_index"], trial["seed"], str(trial["info"]["ep_id"])


def main():
    """Compatibility entry point; new exports follow the flat GIF convention."""
    import sys
    from export_evaluation_gifs import main as export
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=.5)
    args = parser.parse_args()
    source = args.source_dir
    if (source / "_raw" / "unified_control").is_dir():
        source = source / "_raw"
    sys.argv = [sys.argv[0], "--source", f"control={source / 'unified_control'}",
                "--source", f"contact={source / 'unified_contact'}",
                "--out-dir", str(args.out_dir), "--speed", str(args.speed)]
    export()


if __name__ == "__main__":
    main()
