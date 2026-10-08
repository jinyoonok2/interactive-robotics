"""Aggregate diagnostic-only PartGym predicate traces by phase and skill.

Usage:
    python scripts/analyze_predicate_diagnostics.py \
      --input baseline=/path/to/baseline.json \
      --input split_gripper=/path/to/split.json \
      --out results/predicate_diagnostic_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _new_skill_stats() -> dict[str, Any]:
    return {
        "phase_visits": 0,
        "physically_completed_visits": 0,
        "advanced_visits": 0,
        "advanced_while_predicate_false": 0,
        "advanced_without_ever_complete": 0,
        "complete_with_next_slot_but_not_advanced": 0,
        "unsupported_visits": 0,
        "reach_distance_visits": 0,
        "reach_min_distance_sum_m": 0.0,
        "reached_within_3cm_visits": 0,
        "reached_within_5cm_visits": 0,
        "reached_within_5cm_but_incomplete": 0,
        "policy_alignment_visits": 0,
        "predicted_contact_on_part_3cm_visits": 0,
        "action_target_on_part_5cm_visits": 0,
        "action_target_near_contact_5cm_visits": 0,
        "predicted_contact_error_sum_m": 0.0,
        "action_target_error_sum_m": 0.0,
    }


def _new_predicate_stats() -> dict[str, Any]:
    return {
        "phase_visits": 0,
        "event_checks": 0,
        "event_matches": 0,
        "ever_matched_visits": 0,
        "final_matched_visits": 0,
        "achieved_then_lost_visits": 0,
        "expected_true_visits": 0,
        "expected_false_visits": 0,
    }


def _phase_visit(events: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        event["predicate_diagnostic"]
        for event in events
        if "predicate_diagnostic" in event
    ]
    supported = bool(diagnostics) and all(
        diagnostic.get("supported", False) for diagnostic in diagnostics
    )
    skill_name = (
        diagnostics[0].get("skill_name") if diagnostics else "missing_diagnostic"
    )
    complete_sequence = [
        bool(diagnostic.get("complete", False))
        for diagnostic in diagnostics
        if diagnostic.get("supported", False)
    ]
    advanced_events = [event for event in events if event.get("advanced", False)]
    advanced = bool(advanced_events)
    advanced_while_false = any(
        not event.get("predicate_diagnostic", {}).get("complete", False)
        for event in advanced_events
    )
    valid_skill_count = max(
        (int(event.get("valid_skill_count", 1)) for event in events),
        default=1,
    )
    phase = int(events[0].get("phase", 0))
    next_slot_exists = phase + 1 < valid_skill_count

    predicate_sequences: dict[str, list[bool]] = defaultdict(list)
    predicate_expected: dict[str, bool] = {}
    for diagnostic in diagnostics:
        expected = diagnostic.get("predicate_expected", {})
        observed = diagnostic.get("predicate_results", {})
        for predicate, expected_value in expected.items():
            predicate_expected[predicate] = bool(expected_value)
            if predicate in observed:
                predicate_sequences[predicate].append(
                    bool(observed[predicate]) == bool(expected_value)
                )

    reach_distances = [
        float(reach["tcp_to_part_min_distance_m"])
        for event in events
        for reach in [event.get("reach_diagnostic", {})]
        if reach.get("supported", False)
        and "tcp_to_part_min_distance_m" in reach
    ]
    alignments = [
        alignment
        for event in events
        for alignment in [event.get("policy_target_alignment", {})]
        if alignment.get("supported", False)
    ]

    def alignment_min(key: str) -> float | None:
        values = [
            float(alignment[key])
            for alignment in alignments
            if key in alignment
        ]
        return min(values) if values else None

    return {
        "phase": phase,
        "skill_name": skill_name,
        "supported": supported,
        "complete_ever": any(complete_sequence),
        "complete_final": complete_sequence[-1] if complete_sequence else False,
        "advanced": advanced,
        "advanced_while_false": advanced_while_false,
        "next_slot_exists": next_slot_exists,
        "predicate_sequences": dict(predicate_sequences),
        "predicate_expected": predicate_expected,
        "reach_supported": bool(reach_distances),
        "min_tcp_to_part_distance_m": (
            min(reach_distances) if reach_distances else None
        ),
        "min_predicted_contact_to_part_distance_m": alignment_min(
            "predicted_contact_to_part_min_distance_m"
        ),
        "min_action_target_to_part_distance_m": alignment_min(
            "action_target_to_part_min_distance_m"
        ),
        "min_action_target_to_predicted_contact_distance_m": alignment_min(
            "action_target_to_predicted_contact_distance_m"
        ),
    }


def _failure_label(visits: list[dict[str, Any]], success: bool) -> str:
    if success:
        return "success"
    if not visits:
        return "missing_predicate_trace"
    unsupported = next(
        (visit for visit in visits if not visit["supported"]), None
    )
    if unsupported is not None:
        return f"unsupported:{unsupported['skill_name']}"
    unresolved = next(
        (visit for visit in visits if not visit["complete_ever"]), None
    )
    if unresolved is None:
        return "task_failed_despite_all_phase_predicates"
    min_distance = unresolved["min_tcp_to_part_distance_m"]
    if min_distance is not None:
        contact_error = unresolved[
            "min_predicted_contact_to_part_distance_m"
        ]
        action_error = unresolved["min_action_target_to_part_distance_m"]
        if contact_error is not None and contact_error > 0.03:
            return f"{unresolved['skill_name']}:contact_prediction_off_part"
        if action_error is not None and action_error > 0.05:
            return f"{unresolved['skill_name']}:action_target_off_part"
        if min_distance <= 0.05:
            return f"{unresolved['skill_name']}:reached_5cm_but_predicate_failed"
        return f"{unresolved['skill_name']}:targeted_part_but_never_reached_5cm"
    never_matched = sorted(
        predicate
        for predicate, sequence in unresolved["predicate_sequences"].items()
        if not any(sequence)
    )
    predicate_suffix = ",".join(never_matched) or "compound_predicates"
    return f"{unresolved['skill_name']}:{predicate_suffix}"


def analyze(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    skill_stats: dict[str, dict[str, Any]] = defaultdict(_new_skill_stats)
    predicate_stats: dict[str, dict[str, Any]] = defaultdict(
        _new_predicate_stats
    )
    failure_labels: Counter[str] = Counter()
    unsupported_reasons: Counter[str] = Counter()
    unsupported_reach_reasons: Counter[str] = Counter()
    total_events = predicate_events = 0
    phase_visits_total = 0

    for episode in data.get("results", []):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in episode.get("phase_events", []):
            total_events += 1
            if "predicate_diagnostic" in event:
                predicate_events += 1
                diagnostic = event["predicate_diagnostic"]
                if not diagnostic.get("supported", False):
                    unsupported_reasons[str(diagnostic.get("reason", ""))] += 1
            reach = event.get("reach_diagnostic")
            if reach is not None and not reach.get("supported", False):
                unsupported_reach_reasons[str(reach.get("reason", ""))] += 1
            grouped[int(event.get("phase", 0))].append(event)

        visits = [
            _phase_visit(grouped[phase])
            for phase in sorted(grouped)
        ]
        failure_labels[_failure_label(visits, bool(episode.get("success")))] += 1

        for visit in visits:
            phase_visits_total += 1
            skill = str(visit["skill_name"])
            stats = skill_stats[skill]
            stats["phase_visits"] += 1
            stats["physically_completed_visits"] += int(visit["complete_ever"])
            stats["advanced_visits"] += int(visit["advanced"])
            stats["advanced_while_predicate_false"] += int(
                visit["advanced_while_false"]
            )
            stats["advanced_without_ever_complete"] += int(
                visit["advanced"] and not visit["complete_ever"]
            )
            stats["complete_with_next_slot_but_not_advanced"] += int(
                visit["complete_ever"]
                and visit["next_slot_exists"]
                and not visit["advanced"]
            )
            stats["unsupported_visits"] += int(not visit["supported"])
            min_distance = visit["min_tcp_to_part_distance_m"]
            if min_distance is not None:
                stats["reach_distance_visits"] += 1
                stats["reach_min_distance_sum_m"] += min_distance
                stats["reached_within_3cm_visits"] += int(
                    min_distance <= 0.03
                )
                stats["reached_within_5cm_visits"] += int(
                    min_distance <= 0.05
                )
                stats["reached_within_5cm_but_incomplete"] += int(
                    min_distance <= 0.05 and not visit["complete_ever"]
                )
            contact_error = visit[
                "min_predicted_contact_to_part_distance_m"
            ]
            action_error = visit["min_action_target_to_part_distance_m"]
            action_contact_error = visit[
                "min_action_target_to_predicted_contact_distance_m"
            ]
            if contact_error is not None and action_error is not None:
                stats["policy_alignment_visits"] += 1
                stats["predicted_contact_error_sum_m"] += contact_error
                stats["action_target_error_sum_m"] += action_error
                stats["predicted_contact_on_part_3cm_visits"] += int(
                    contact_error <= 0.03
                )
                stats["action_target_on_part_5cm_visits"] += int(
                    action_error <= 0.05
                )
                stats["action_target_near_contact_5cm_visits"] += int(
                    action_contact_error is not None
                    and action_contact_error <= 0.05
                )

            for predicate, sequence in visit["predicate_sequences"].items():
                pred = predicate_stats[predicate]
                pred["phase_visits"] += 1
                pred["event_checks"] += len(sequence)
                pred["event_matches"] += sum(sequence)
                pred["ever_matched_visits"] += int(any(sequence))
                pred["final_matched_visits"] += int(
                    sequence[-1] if sequence else False
                )
                pred["achieved_then_lost_visits"] += int(
                    any(sequence)
                    and any(
                        not matched
                        for matched in sequence[
                            sequence.index(True) + 1 :
                        ]
                    )
                )
                expected = visit["predicate_expected"].get(predicate)
                pred["expected_true_visits"] += int(expected is True)
                pred["expected_false_visits"] += int(expected is False)

    finalized_skills = {}
    for skill, stats in sorted(skill_stats.items()):
        finalized_skills[skill] = {
            **stats,
            "physical_completion_rate": _rate(
                stats["physically_completed_visits"], stats["phase_visits"]
            ),
            "advance_rate": _rate(
                stats["advanced_visits"], stats["phase_visits"]
            ),
            "premature_advance_rate": _rate(
                stats["advanced_without_ever_complete"],
                stats["advanced_visits"],
            ),
            "mean_phase_min_tcp_to_part_distance_m": _rate(
                stats["reach_min_distance_sum_m"],
                stats["reach_distance_visits"],
            ),
            "reached_within_3cm_rate": _rate(
                stats["reached_within_3cm_visits"],
                stats["reach_distance_visits"],
            ),
            "reached_within_5cm_rate": _rate(
                stats["reached_within_5cm_visits"],
                stats["reach_distance_visits"],
            ),
            "mean_phase_min_predicted_contact_error_m": _rate(
                stats["predicted_contact_error_sum_m"],
                stats["policy_alignment_visits"],
            ),
            "mean_phase_min_action_target_error_m": _rate(
                stats["action_target_error_sum_m"],
                stats["policy_alignment_visits"],
            ),
            "predicted_contact_on_part_3cm_rate": _rate(
                stats["predicted_contact_on_part_3cm_visits"],
                stats["policy_alignment_visits"],
            ),
            "action_target_on_part_5cm_rate": _rate(
                stats["action_target_on_part_5cm_visits"],
                stats["policy_alignment_visits"],
            ),
            "action_target_near_contact_5cm_rate": _rate(
                stats["action_target_near_contact_5cm_visits"],
                stats["policy_alignment_visits"],
            ),
        }

    finalized_predicates = {}
    for predicate, stats in sorted(predicate_stats.items()):
        finalized_predicates[predicate] = {
            **stats,
            "event_match_rate": _rate(
                stats["event_matches"], stats["event_checks"]
            ),
            "ever_matched_visit_rate": _rate(
                stats["ever_matched_visits"], stats["phase_visits"]
            ),
            "final_matched_visit_rate": _rate(
                stats["final_matched_visits"], stats["phase_visits"]
            ),
            "achieved_then_lost_rate": _rate(
                stats["achieved_then_lost_visits"], stats["phase_visits"]
            ),
        }

    episodes = int(data.get("num_episodes", len(data.get("results", []))))
    return {
        "source": str(path),
        "rollout": {
            "episodes": episodes,
            "successes": sum(
                bool(result.get("success")) for result in data.get("results", [])
            ),
            "success_rate": float(data.get("success_rate", 0.0)),
            "mean_completion_rate": float(
                data.get("mean_completion_rate", 0.0)
            ),
        },
        "trace_quality": {
            "total_phase_events": total_events,
            "predicate_events": predicate_events,
            "predicate_event_coverage": _rate(
                predicate_events, total_events
            ),
            "phase_visits": phase_visits_total,
            "unsupported_event_reasons": dict(unsupported_reasons.most_common()),
            "unsupported_reach_reasons": dict(
                unsupported_reach_reasons.most_common()
            ),
        },
        "by_skill": finalized_skills,
        "by_predicate": finalized_predicates,
        "episode_outcomes": dict(failure_labels.most_common()),
    }


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=/path/to/results.json")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid input: {value}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=_parse_input,
        required=True,
        help="Repeatable LABEL=/path/to/rollout_results.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "models": {
            label: analyze(path)
            for label, path in args.input
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[predicate-analysis] wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
