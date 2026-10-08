"""Shared phase-advance rules for legacy and oracle termination."""
from __future__ import annotations


def should_advance_phase(
    *,
    skill_complete: bool,
    phase_steps: int,
    min_phase_cycles: int,
    current_phase: int,
    valid_skill_count: int,
) -> bool:
    """Return True iff the controller should move to the next skill slot."""
    if not skill_complete:
        return False
    if phase_steps < int(min_phase_cycles):
        return False
    if int(current_phase) + 1 >= int(valid_skill_count):
        return False
    return True
