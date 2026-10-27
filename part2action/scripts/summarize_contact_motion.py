"""Contact localization and action alignment over complete evaluation trajectories."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def mean(values):
    return statistics.mean(values) if values else None


def write_csv(path, rows):
    if rows:
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(payload):
    steps, trials = [], []
    for trial in payload["results"]:
        trial_steps = []
        for event in trial["phase_events"]:
            for reference, key in [("initial_task_part", "initial_target_part_alignment"),
                                   ("selected_skill_part", "policy_target_alignment")]:
                alignment = event.get(key, {})
                supported = bool(alignment.get("supported", False))
                row = {"reference": reference, "object": trial["obj_class"], "task_type": trial["task_type"],
                       "trial_index": trial["trial_index"], "seed": trial["seed"],
                       "episode_id": trial["info"].get("ep_id"), "step": event["inference_step"],
                       "predicted_phase": event["phase"], "target_skill_index": alignment.get("skill_index"),
                       "part": alignment.get("part_name"), "supported": supported,
                       "reason": alignment.get("reason", "missing measurement")}
                for column, field in [("contact_to_part_cm", "predicted_contact_to_part_min_distance_m"),
                                      ("tcp_to_part_cm", "tcp_to_part_min_distance_m"),
                                      ("action_to_part_cm", "action_target_to_part_min_distance_m")]:
                    row[column] = 100 * alignment[field] if supported else None
                steps.append(row)
                if reference == "initial_task_part" and supported:
                    trial_steps.append(row)
        row = {"object": trial["obj_class"], "task_type": trial["task_type"],
               "trial_index": trial["trial_index"], "seed": trial["seed"],
               "episode_id": trial["info"].get("ep_id"), "success": trial["success"],
               "recorded_steps": len(trial["phase_events"]), "supported_steps": len(trial_steps),
               "phase_changes": sum(e["advanced"] for e in trial["phase_events"]),
               "gripper_clipped_count": trial.get("gripper_clipped_count", 0)}
        for field in ["contact_to_part_cm", "tcp_to_part_cm", "action_to_part_cm"]:
            values = [s[field] for s in trial_steps]
            row["mean_" + field] = mean(values)
            row["start_" + field] = values[0] if values else None
            row["end_" + field] = values[-1] if values else None
            third = max(1, len(values) // 3)
            row["last_third_mean_" + field] = mean(values[-third:])
        contacts = [s["contact_to_part_cm"] for s in trial_steps]
        row["fraction_contact_within_1cm"] = mean([v <= 1 for v in contacts])
        row["fraction_contact_within_3cm"] = mean([v <= 3 for v in contacts])
        trials.append(row)
    summary = {"trials": len(trials), "steps": len(steps) // 2,
               "supported_initial_part_steps": sum(t["supported_steps"] for t in trials),
               "successes": sum(t["success"] for t in trials),
               "gripper_command_bounds_m": payload.get("gripper_command_bounds_m"),
               "metric_reference": "First ground-truth task part, tracked with the object throughout motion"}
    for key in trials[0] if trials else []:
        if key.startswith(("mean_", "start_", "end_", "last_third_mean_", "fraction_")):
            summary[key] = mean([r[key] for r in trials if r[key] is not None])
    summary["selected_skill_unsupported_steps"] = sum(not s["supported"] for s in steps if s["reference"] == "selected_skill_part")
    return summary, trials, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    source = next((args.run_dir / "hierarchical").glob("rollout_results_*.json"))
    payload = json.loads(source.read_text())
    assert payload.get("contact_diagnostics"), "Contact logging must be enabled"
    summary, trials, steps = summarize(payload)
    out = args.run_dir / "contact_motion_report"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out / "per_trial.csv", trials)
    write_csv(out / "per_step.csv", steps)
    lines = ["# Contact localization throughout motion", "",
             "This run uses bounded gripper commands. Ground-truth part geometry is measured before each action and never supplied to the policy. Main aggregates track the first instructed part throughout motion, even after the model changes phase. A second trace scores the part associated with the selected skill; unsupported move/rotation skills remain explicitly marked.", "",
             "Numbers are averages of per-trial statistics, giving each supported trial equal weight. Distances are to sampled part surfaces, not a unique physical contact or grasp pose. Initial approach distance is expected to be large.", "",
             f"Trials: {summary['trials']}; success count: {summary['successes']}; supported initial-part measurements: {summary['supported_initial_part_steps']}/{summary['steps']}.", "",
             "| Distance | Mean across motion (cm) | Start (cm) | End (cm) | Last third (cm) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, field in [("Predicted contact to true part", "contact_to_part_cm"),
                         ("TCP to true part", "tcp_to_part_cm"), ("Commanded position to true part", "action_to_part_cm")]:
        values = [summary[k + field] for k in ["mean_", "start_", "end_", "last_third_mean_"]]
        lines.append("| " + label + " | " + " | ".join(f"{v:.2f}" if v is not None else "n/a" for v in values) + " |")
    lines += ["", "`per_step.csv` contains both target references with object, task, episode, phase, and unsupported-case reasons. `per_trial.csv` contains individual trajectory averages and end distances. `summary.json` contains aggregate values. Recorded media is in `../hierarchical/videos/`."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
