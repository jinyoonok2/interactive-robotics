"""Frozen-checkpoint intervention metrics and expert-first GIF review."""
import argparse
import csv
import html
import json
import statistics
import textwrap
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from build_policy_comparison_videos import annotated_panel, key, text, render_panels
from export_evaluation_gifs import encode_gif, write_bundle
from ground_truth_reference import GroundTruthReferences
from summarize_contact_motion import summarize


VARIANTS = [("normal", "CONTACT 100%"), ("reduced", "CONTACT 25%"), ("zero", "CONTACT 0%")]


def render(trials, output):
    captures = [cv2.VideoCapture(t["rollout_video"]) for t in trials]
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]
    fps = [c.get(cv2.CAP_PROP_FPS) for c in captures]
    if not all(c.isOpened() for c in captures) or len(set(fps)) != 1:
        raise ValueError("Missing media or inconsistent input FPS")
    events = [{e["inference_step"]: e for e in t["phase_events"]} for t in trials]
    writer = imageio_ffmpeg.write_frames(str(output), (1440, 720), fps=fps[0]*.5,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"])
    writer.send(None)
    last = [None]*3
    try:
        for index in range(max(counts)):
            canvas = np.full((720, 1440, 3), 22, np.uint8)
            t = trials[0]
            text(canvas, f"SAME CONTACT CHECKPOINT | {t['obj_class']} task {t['task_type']} trial {t['trial_index']} | episode {t['info']['ep_id']}", 12, 24)
            for i, line in enumerate(textwrap.wrap(t["instruction"], 145)[:2]):
                text(canvas, line, 12, 46+i*20, .52)
            for col, (cap, trial) in enumerate(zip(captures, trials)):
                x = col*480
                if index < counts[col]:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Decode failure: {output}, frame {index}")
                    last[col] = frame
                text(canvas, VARIANTS[col][1] + (" | SUCCESS" if trial["success"] else " | FAILURE"), x+12, 103)
                event = events[col].get(index) if index < counts[col] else None
                panel, alignment = annotated_panel(last[col], event)
                canvas[120:600, x:x+480] = panel
                if alignment.get("supported"):
                    for row, (label, field) in enumerate([
                        ("Contact -> true part", "predicted_contact_to_part_min_distance_m"),
                        ("TCP -> true part", "tcp_to_part_min_distance_m"),
                        ("Command -> true part", "action_target_to_part_min_distance_m")]):
                        text(canvas, f"{label}: {100*alignment[field]:.2f} cm", x+12, 624+23*row)
                else:
                    label = "Recorded preview; geometry traces pending" if trial.get("preview_only") else "Final frame held; no next prediction"
                    text(canvas, label, x+12, 625, .5)
            legend = "Recorded preview; overlays available in full gallery" if trials[0].get("preview_only") else "Blue: predicted contact; green: nearest true-part point"
            text(canvas, f"Step {index} | playback 0.5x | {legend}", 12, 711, .48)
            writer.send(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
        for cap in captures:
            cap.release()
    check = cv2.VideoCapture(str(output))
    actual = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    assert check.isOpened() and actual == max(counts)
    assert int(check.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1440
    for frame in [0, actual-1]:
        check.set(cv2.CAP_PROP_POS_FRAMES, frame)
        assert check.read()[0]
    check.release()
    return {"frames": actual, "input_frame_counts": counts, "input_fps": fps[0], "output_fps": fps[0]*.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_dir
    inputs = root/'_raw' if (root/'_raw'/'normal').is_dir() else root
    data, summary, probes = {}, {}, []
    for name, label in VARIANTS:
        source = next((inputs/name/"hierarchical").glob("rollout_results_*.json"))
        payload = json.loads(source.read_text())
        assert payload["gripper_command_bounds_m"] == [0, .04]
        data[name] = {key(t): t for t in payload["results"]}
        summary[name] = summarize(payload)[0]
        summary[name]["strength"] = payload["contact_intervention_strength"]
        for trial in payload["results"]:
            for d in trial["diagnostics"]:
                probe = d["contact_intervention"]
                for strength, candidate in probe["candidates"].items():
                    probes.append({"trajectory": name, "object": trial["obj_class"], "task": trial["task_type"],
                        "trial": trial["trial_index"], "seed": trial["seed"], "episode_id": trial["info"]["ep_id"],
                        "step": d["inference_step"], "candidate_strength": strength,
                        "relative_input_norm": probe["relative_input_norm"], "base_pooled_feature_rms": probe["base_pooled_feature_rms"],
                        **{k: v for k, v in candidate.items() if k != "action0"},
                        **dict(zip(["x", "y", "z", "roll", "pitch", "yaw", "gripper_raw"], candidate["action0"]))})
    sources = [json.loads(next((inputs/name/"hierarchical").glob("rollout_results_*.json")).read_text()) for name, _ in VARIANTS]
    assert len({s["ckpt"] for s in sources}) == 1
    for field in ["config", "split", "termination_mode", "min_phase_cycles"]:
        assert len({str(s[field]) for s in sources}) == 1, field
    assert all(set(data[name]) == set(data["normal"]) for name, _ in VARIANTS)
    assert len(data["normal"]) == 39
    with (root/"same_observation_probes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(probes[0]))
        writer.writeheader(); writer.writerows(probes)
    # Report paired output sensitivity on NORMAL trajectories, avoiding different-state confounds.
    probe_summary = {}
    for strength in ["0.0", "0.25", "1.0"]:
        rows = [p for p in probes if p["trajectory"] == "normal" and p["candidate_strength"] == strength]
        trial_means = []
        for episode in data["normal"]:
            obj, task, trial, seed, ep = episode
            selected = [r for r in rows if (r["object"],str(r["task"]),r["trial"],r["seed"],str(r["episode_id"])) == episode]
            trial_means.append({f: statistics.mean(r[f] for r in selected) for f in [
                "xyz_change_from_zero_cm", "rotation_change_from_zero_deg", "projection_to_base_rms_ratio", "contact_signal_to_base_rms_ratio"]})
        probe_summary[strength] = {f: statistics.mean(t[f] for t in trial_means) for f in trial_means[0]}
    (root/"summary.json").write_text(json.dumps({"rollouts": summary, "same_observation_normal_trajectory": probe_summary}, indent=2)+"\n")
    media = root/"_raw/composites"
    media.mkdir(parents=True, exist_ok=True)
    references=GroundTruthReferences(root/'_raw'/'ground_truth')
    manifest, cards = [], []
    for episode in sorted(data["normal"]):
        trials = [data[name][episode] for name, _ in VARIANTS]
        assert len({t["instruction"] for t in trials}) == 1
        assert all(np.allclose(trials[0]["diagnostics"][0]["tcp_xyz"], t["diagnostics"][0]["tcp_xyz"]) for t in trials)
        obj, task, trial, seed, ep = episode
        filename = f"{obj}_task{task}_trial{trial:02d}_episode{ep}.mp4"
        truth=references.trial(trials[0])
        result = render_panels([truth]+trials,['GROUND-TRUTH REFERENCE']+[label for _,label in VARIANTS],media/filename,.5)
        gif = root/Path(filename).with_suffix('.gif')
        result.update(encode_gif(media/filename, gif, 5, 1+len(VARIANTS)))
        manifest.append({"object": obj, "task": task, "trial": trial, "seed": seed, "episode_id": ep,
                         "ground_truth_reference":truth['ground_truth_reference'],
                         "panel_order":['ground_truth_reference']+[name for name,_ in VARIANTS],
                         "gif": gif.name, "video": "_raw/composites/"+filename, "success": {name: t["success"] for (name,_),t in zip(VARIANTS,trials)}, **result})
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        cards.append(f'<article data-object="{obj}"><h2>{obj} task {task} trial {trial} — episode {ep}</h2><p>{html.escape(trials[0]["instruction"])}</p><img loading="lazy" src="{gif.name}" alt="Contact 100%, 25%, and 0%"><p><a href="{gif.name}" download>Download GIF</a></p></article>')
        print(f"[{len(manifest)}/39] {filename}", flush=True)
    rows = ''.join(f'<tr><td>{label}</td><td>{summary[name]["successes"]}/39</td><td>{summary[name]["last_third_mean_tcp_to_part_cm"]:.2f}</td><td>{summary[name]["last_third_mean_contact_to_part_cm"]:.2f}</td></tr>' for name,label in VARIANTS)
    page = '<!doctype html><meta charset="utf-8"><title>Frozen contact-policy interventions</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#17191d;color:#eee}article{background:#24272d;padding:18px;margin:20px 0}video{width:100%}a{color:#9bd0ff}td,th{padding:10px;text-align:left}</style><h1>Same trained contact policy: 100%, 25%, and 0% contact input</h1><p>39 matched test1 trials, same checkpoint, seeds, initial states, 120-step budget and bounded gripper. The relative contact vector entering its projection is multiplied by 1, 0.25 or 0. The learned projection bias is preserved. No retraining. This tests inference sensitivity; zero input is outside the model’s normal training distribution.</p><table><tr><th>Condition</th><th>Success</th><th>Final-third TCP distance (cm)</th><th>Final-third contact error (cm)</th></tr>'+rows+'</table><p>Left to right: ground-truth reference, 100%, 25%, 0%. The expert reference matches the complete skill chain but is a different episode and may have a different starting scene. Half-speed playback aligned by elapsed sample time, not task phase. Blue: predicted contact; green: nearest point on the first true target part. Ground-truth geometry is diagnostic only. Earlier finishes hold the last frame.</p><label>Object <select id="object"><option value="">All</option><option>bottle</option><option>mug</option><option>pliers</option><option>scissors</option></select></label>'+''.join(cards)+'<script>document.querySelector("#object").onchange=e=>document.querySelectorAll("article").forEach(a=>a.hidden=e.target.value&&a.dataset.object!==e.target.value)</script>'
    (root/"index.html").write_text(page.replace('video{width:100%}', 'img{width:100%}'))
    write_bundle(root)
    print(json.dumps({"rollouts": summary, "paired_probes": probe_summary}, indent=2))


if __name__ == "__main__":
    main()
