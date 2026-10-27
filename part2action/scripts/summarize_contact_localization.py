"""Summarize frozen-checkpoint contact localization on initial test observations."""
import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def summarize(rows):
    supported = [r for r in rows if r["supported"]]
    distances = [r["contact_to_true_part_cm"] for r in supported]
    return {
        "trials": len(rows), "supported": len(supported), "unsupported": len(rows) - len(supported),
        "mean_contact_to_part_cm": statistics.mean(distances) if distances else None,
        "median_contact_to_part_cm": statistics.median(distances) if distances else None,
        "within_1cm": sum(d <= 1 for d in distances),
        "within_3cm": sum(d <= 3 for d in distances),
        "within_5cm": sum(d <= 5 for d in distances),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", type=Path, default=ROOT / "results/part2action/evaluations/contact_localization_submissions_20261008.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/diagnostics/contact_motion_20261008/reference_reports/contact_localization_20261008")
    args = parser.parse_args()
    rows, pending, sources = [], [], {}
    for entry in json.loads(args.submissions.read_text()):
        source = list(Path(entry["out_dir"]).rglob("rollout_results_*.json"))
        if not source:
            pending.append(entry["model"])
            continue
        payload = json.loads(source[0].read_text())
        assert payload["num_episodes"] == 39, "Expected all 39 diagnostic trials"
        assert payload.get("contact_diagnostics"), "Ground-truth contact logging was not enabled"
        sources[entry["model"]] = str(source[0])
        for trial in payload["results"]:
            alignment = trial["phase_events"][0].get("policy_target_alignment", {})
            supported = bool(alignment.get("supported", False))
            row = {"model": entry["model"], "object": trial["obj_class"],
                   "task_type": trial["task_type"], "trial_index": trial["trial_index"],
                   "seed": trial["seed"], "episode_id": trial["info"].get("ep_id"),
                   "instruction": trial["instruction"], "supported": supported,
                   "part": alignment.get("part_name"), "skill": alignment.get("skill_name"),
                   "reason": alignment.get("reason", "missing diagnostic"),
                   "predicted_phase": trial["phase_trace"][0],
                   "ground_truth_skill_index": alignment.get("skill_index")}
            for name, key in [("contact_to_true_part_cm", "predicted_contact_to_part_min_distance_m"),
                              ("tcp_to_true_part_cm", "tcp_to_part_min_distance_m"),
                              ("action_to_true_part_cm", "action_target_to_part_min_distance_m")]:
                row[name] = 100 * alignment[key] if supported else None
            rows.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {model: summarize([r for r in rows if r["model"] == model]) for model in sources}
    per_object = {f"{model}/{obj}": summarize([r for r in rows if r["model"] == model and r["object"] == obj])
                  for model in sources for obj in ["bottle", "mug", "pliers", "scissors"]}
    (args.out_dir / "summary.json").write_text(json.dumps({"models": summary, "per_object": per_object,
                                                          "pending": pending, "sources": sources}, indent=2) + "\n")
    if rows:
        with (args.out_dir / "per_trial.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Initial contact localization against ground truth", "",
             "Frozen checkpoints are scored on the same 39 test1 episodes before the first action. Ground-truth target geometry comes from skill 0 in the simulator task chain, independently of the model's predicted phase. It is logging-only and never enters policy inputs.", "",
             "Distance is to the nearest sampled point on the intended target part. This checks part localization, not agreement with a unique physical contact, a collision-free grasp pose, or task success. Proximity thresholds account for region distance; they do not establish physical contact.", "",
             "| Model | Supported / trials | Mean (cm) | Median (cm) | Within 1 cm | Within 3 cm | Within 5 cm |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, values in summary.items():
        mean = f"{values['mean_contact_to_part_cm']:.2f}" if values["supported"] else "n/a"
        median = f"{values['median_contact_to_part_cm']:.2f}" if values["supported"] else "n/a"
        lines.append(f"| {model} | {values['supported']}/{values['trials']} | {mean} | {median} | {values['within_1cm']} | {values['within_3cm']} | {values['within_5cm']} |")
    lines += ["", "Threshold counts use only supported measurements. Unsupported cases and reasons are retained in per_trial.csv; per-object statistics are in summary.json.", "",
              "Pending outputs: " + (", ".join(pending) if pending else "none")]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print("Pending:", pending)


if __name__ == "__main__":
    main()
