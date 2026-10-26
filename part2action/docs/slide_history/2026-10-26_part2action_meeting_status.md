# Part2Action research update — October 26, 2026

Date: 2026-10-26 (1026)

Four slides for Google Slides; speaker notes are separate. Proposed research is
distinguished from current implementation.

## Slide 1 — Part mask versus contact point

Part mask: which image region belongs to the target part?
Contact point: which 3D surface location is associated with the demonstrated interaction?
Current label: select a gripper-close timestep, read its TCP position, and choose
  the nearest annotated target-part point from the current frame in world coordinates.
The mask supervises visual gating; the derived point supervises contact prediction.
Both are predicted at deployment; ground-truth part annotations are training labels.

Speaker notes: The active legacy rule selects the first commanded opening ≤0.02
at or after the sample; already closed selects the current step, and no close
selects the final frame. This is a proxy, not measured contact. Comparing
current-frame points with an event-frame TCP can introduce object-motion error.

## Slide 2 — A spatial target anchored to an interaction event

The contact point is derived using an event time; the model predicts XYZ, not time.
The derived event timestep is training metadata, not a temporal prediction.
TCP pose contains position/orientation; measured gripper opening is separate.
The action sequence already specifies when to close through per-step commands.
A separate closing-time head alone would not introduce new supervision.

Speaker notes: Distinguish contact location, commanded closure, and actual
finger–object contact. Evaluate closing events from executed actions; physical
contact timing needs validated replay/contact records. XYZ alone cannot establish it.

## Slide 3 — Why contact geometry might help the current policy

Part gating emphasizes the relevant region; contact relative to TCP adds metric guidance.
Unified contact projects the detached vector (predicted contact − TCP) / 0.1 m
  into action features throughout motion.
Unified control uses zeros in the same projection and retains contact supervision.
Compare matched rollouts to test whether explicit geometry improves success.
A surface point alone does not specify wrist pose, gripper width, or a valid approach.

Speaker notes: The MLP outputs eight actions; rollout executes one and observes
again. Contact prediction averages scene points and may fall between surfaces.
Localization errors can harm control; improved rollout success is not established here.

## Slide 4 — Proposed contribution: Learn interaction tolerances

Learn which position, wrist-orientation, and closing-time changes preserve task success.
Represent acceptable interactions as a part-, task-, and state-conditioned set.
Keep current sensors; add successful/failed simulator perturbations as supervision.
Use learned tolerances to train action selection, beyond fitting one demonstrated point.
Test spatial versus joint spatial/temporal tolerances against matched-data controls;
  report held-out success, robustness, latency, and uncertainty across seeds.

Speaker notes: This is proposed, not implemented. Success depends on the starting
state and continuation controller; validate replay and define replanning consistently.
PartInstruct and KITE already use part/keypoint guidance; ReKep and GLiDE overlap
with constraints and perturbation learning. Novelty must lie in how pose–timing
success sets are learned and used, supported by evidence.

References: [PartInstruct](https://www.cs.jhu.edu/~alanlab/Pubs25/yin2025PartInstruct.pdf),
[KITE](https://arxiv.org/abs/2306.16605),
[ReKep](https://arxiv.org/abs/2409.01652),
[GLiDE](https://arxiv.org/abs/2403.17124).

Code: [label derivation](../../data/targets.py),
[loader](../../data/partinstruct_loader.py),
[contact conditioning](../../models/heads.py),
[unified contact config](../../configs/architecture_update/hierarchical_world_geometry_30epoch_unified_contact.yaml).
