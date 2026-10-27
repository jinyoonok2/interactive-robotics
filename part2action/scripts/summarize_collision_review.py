"""First-contact robot pose, finger state and bottle disturbance in matched replays."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from build_contact_intervention_report import VARIANTS, render


def enrich(trial):
    samples = trial["physics_collision_trace"]
    initial = samples[0]
    initial_rotation = Rotation.from_quat(initial["object_quat_xyzw"])
    initial_axis = initial_rotation.apply([0, 0, 1])
    rows = []
    for sample in samples:
        contacts = [c for c in sample["contacts"] if c["normal_force_n"] > 1e-6]
        fingers = [c for c in contacts if c["is_finger"]]
        other = [c for c in contacts if not c["is_finger"]]
        rotation = Rotation.from_quat(sample["object_quat_xyzw"])
        tilt = np.degrees(np.arccos(np.clip(np.dot(initial_axis, rotation.apply([0,0,1])), -1, 1)))
        command = sample["executed_action"]
        orientation_error = (Rotation.from_euler("xyz", command[3:6]).inv()*Rotation.from_quat(sample["tcp_quat_xyzw"])).magnitude()*180/np.pi if command is not None else None
        rows.append({
            "elapsed_s": sample["elapsed_s"], "action_index": sample["action_index"],
            "tcp_x": sample["tcp_xyz"][0], "tcp_y": sample["tcp_xyz"][1], "tcp_z": sample["tcp_xyz"][2],
            "tcp_quat_xyzw": sample["tcp_quat_xyzw"],
            "commanded_xyz": command[:3] if command is not None else None,
            "commanded_rpy": command[3:6] if command is not None else None,
            "commanded_finger_cm": command[6]*100 if command is not None else None,
            "left_finger_cm": sample["finger_positions_m"][0]*100,
            "right_finger_cm": sample["finger_positions_m"][1]*100,
            "orientation_tracking_error_deg": orientation_error,
            "position_tracking_error_cm": float(np.linalg.norm(np.array(command[:3])-sample["tcp_xyz"])*100) if command is not None else None,
            "finger_normal_force_n": sum(c["normal_force_n"] for c in fingers),
            "nonfinger_normal_force_n": sum(c["normal_force_n"] for c in other),
            "contact_links": sorted(set(c["robot_link_name"] for c in contacts)),
            "geometric_contact_links": sorted(set(c["robot_link_name"] for c in sample["contacts"])),
            "both_fingers_in_contact": len(set(c["robot_link"] for c in fingers)) == 2,
            "object_axis_tilt_from_start_deg": float(tilt),
            "object_translation_from_start_cm": float(np.linalg.norm(np.array(sample["object_xyz"])-initial["object_xyz"])*100),
            "object_height_change_cm": (sample["object_xyz"][2]-initial["object_xyz"][2])*100,
        })
    return rows


def plot(rows, fields, title):
    colors = ["#5dafff", "#ffbd66", "#68dd91"]
    values = [r[field] for r in rows for field in fields if r[field] is not None]
    lo, hi = min(0, min(values)), max(values)
    hi = max(lo+1e-3, hi)
    end = max(rows[-1]["elapsed_s"], .001)
    lines = []
    for i, field in enumerate(fields):
        points = ' '.join(f'{45+r["elapsed_s"]/end*560:.2f},{165-(r[field]-lo)/(hi-lo)*120:.2f}' for r in rows if r[field] is not None)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="1.5"/><text x="45" y="{202+i*20}" fill="{colors[i]}">{html.escape(field)}</text>')
    contact = next((r for r in rows if r["contact_links"]), None)
    if contact:
        x = 45+contact["elapsed_s"]/end*560
        lines.append(f'<line x1="{x}" x2="{x}" y1="35" y2="165" stroke="white" stroke-dasharray="4,4"/>')
    return f'<svg viewBox="0 0 650 270"><text x="45" y="22" fill="white">{html.escape(title)}</text><rect x="45" y="35" width="560" height="130" fill="none" stroke="#666"/><text x="3" y="45" fill="#bbb">{hi:.1f}</text><text x="3" y="165" fill="#bbb">{lo:.1f}</text><text x="45" y="185" fill="#bbb">0 s</text><text x="555" y="185" fill="#bbb">{end:.2f} s</text>'+''.join(lines)+'</svg>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--skip-video", action="store_true", help="Refresh metrics/HTML while preserving an existing comparison video.")
    args = parser.parse_args()
    root = args.source_dir
    trials, summaries, sections = [], {}, []
    checkpoints = []
    for name, title in VARIANTS:
        payload = json.loads(next((root/name/"hierarchical").glob("rollout_results_*.json")).read_text())
        assert len(payload["results"]) == 1 and payload["collision_diagnostics"]
        checkpoints.append(payload["ckpt"])
        trial = payload["results"][0]; trials.append(trial)
        rows = enrich(trial)
        first = next((r for r in rows if r["contact_links"]), None)
        fingers = next((r for r in rows if r["finger_normal_force_n"] > 0), None)
        both = next((r for r in rows if r["both_fingers_in_contact"]), None)
        close_command = next((r.copy() for r in rows if r["commanded_finger_cm"] is not None and r["commanded_finger_cm"] <= .1), None)
        measured_closed = next((r for r in rows if max(r["left_finger_cm"], r["right_finger_cm"]) <= .1), None)
        if close_command is not None:
            event = trial["phase_events"][close_command["action_index"]]
            close_command["pre_action_tcp_to_target_part_cm"] = 100*event["initial_target_part_alignment"]["tcp_to_part_min_distance_m"]
        summaries[name] = {"success": trial["success"], "first_forceful_contact": first,
            "first_finger_contact": fingers, "first_both_fingers_contact": both,
            "first_nearly_closed_command": close_command,
            "first_measured_nearly_closed_fingers": measured_closed,
            "max_object_axis_tilt_from_start_deg": max(r["object_axis_tilt_from_start_deg"] for r in rows),
            "final_object_axis_tilt_from_start_deg": rows[-1]["object_axis_tilt_from_start_deg"],
            "final_object_height_change_cm": rows[-1]["object_height_change_cm"], "physics_samples": len(rows)}
        with (root/f'{name}_physics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        window = [r for r in rows if first and abs(r["elapsed_s"]-first["elapsed_s"]) <= .25]
        (root/f'{name}_first_contact_window.json').write_text(json.dumps(window,indent=2)+'\n')
        if first:
            note = f'First forceful contact at {first["elapsed_s"]:.3f} s, action {first["action_index"]}: '+', '.join(first["contact_links"])+f'. Finger target {first["commanded_finger_cm"]:.2f} cm per finger; measured left/right {first["left_finger_cm"]:.2f}/{first["right_finger_cm"]:.2f} cm. Orientation tracking error {first["orientation_tracking_error_deg"]:.1f}°.'
        else:
            note = 'No forceful robot–object contact recorded.'
        if close_command is not None:
            note += f' A nearly closed command (at most 1 mm per finger) first occurred at action {close_command["action_index"]}, when pre-action TCP-to-target-part distance was {close_command["pre_action_tcp_to_target_part_cm"]:.1f} cm.'
        if measured_closed is not None:
            note += f' Both measured finger joints first reached at most 1 mm at {measured_closed["elapsed_s"]:.3f} s.'
        plots = plot(rows,["nonfinger_normal_force_n","finger_normal_force_n"],"Robot–object normal forces (N)")+plot(rows,["commanded_finger_cm","left_finger_cm","right_finger_cm"],"Finger command and measured joints (cm per finger)")+plot(rows,["object_axis_tilt_from_start_deg","orientation_tracking_error_deg"],"Object axis tilt and wrist tracking error (degrees)")
        sections.append(f'<section><h2>{title}</h2><p>{html.escape(note)}</p><p>Task success: {trial["success"]}. Maximum bottle-axis tilt change: {summaries[name]["max_object_axis_tilt_from_start_deg"]:.1f}°.</p><div class="plots">{plots}</div><a href="{name}_physics.csv">Full physics trace CSV</a></section>')
    assert len(set(checkpoints)) == 1
    for t in trials[1:]:
        assert (t["info"]["ep_id"],t["seed"],t["instruction"]) == (trials[0]["info"]["ep_id"],trials[0]["seed"],trials[0]["instruction"])
    if args.skip_video:
        assert (root/'comparison.mp4').is_file()
    else:
        render(trials,root/'comparison.mp4')
    (root/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Bottle collision review</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#17191d;color:#eee}video{width:100%}.plots{display:grid;grid-template-columns:repeat(3,1fr)}svg{width:100%}section{background:#24272d;padding:16px;margin:20px 0}a{color:#9bd0ff}</style><h1>Bottle first-collision review</h1><p>One matched bottle task-1 trial per contact-input strength; same checkpoint and seed 42. Physics traces are sampled after every simulator substep (240 Hz) using read-only queries. Forces above 1e-6 N identify forceful contact; all geometric contacts also remain in the raw rollout JSON. Dashed vertical lines mark first forceful robot–object contact. Finger values are individual joint positions; sum left and right for the approximate opening. Axis tilt is change in the object’s local Z axis relative to its initial direction, not a task-success criterion. Tracking error measures disagreement with the commanded wrist orientation, not agreement with an expert grasp pose.</p><video controls muted playsinline preload="none" src="comparison.mp4"></video>'+''.join(sections))
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()
