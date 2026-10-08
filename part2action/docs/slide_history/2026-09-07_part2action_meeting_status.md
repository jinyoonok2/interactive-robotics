# Part2Action Meeting Status

Date: 2026-09-07

## Slide 1 — What Is Running

Baseline world-geometry policy: 8/39 PartGym success (20.5%). No training or full rollout is running today.

Finished ablations that did not beat this: skill gate, oracle phase termination, delta actions, split gripper, skill-specific contact labels.

Active work is the point-part head. The policy was frozen; only this head trained for 10 epochs to mark which of 1,024 scene points belong to the requested part.

Training looked strong: 69% precision / 96% recall / 80% F1. The 39-trial held-out check on kuorobot01 is complete: one step per scene, actions unchanged.

Held-out recognition is weaker: 33% precision / 62% recall / 43% F1 at 5 mm. Touch is worse than grasp. This is localization only, not a new success rate.

Next: add part scores to contact attention and rerun 39 physical trials. Also rerun a corrected baseline; earlier rollouts used 3 point-cloud points instead of 1,024.

## Slide 2 — Problems We Need to Fix

Physical success is 8/39. Offline metrics do not match closed-loop contact and grasp.

Touch grounding is the main perception problem: only 6/18 touch contacts were within 3 cm of the requested part. Thirteen failures aimed at the wrong region.

Near the part is not enough. Touch completed 1/18 and grasp 10/21. Grasp geometry looked good (19/21 within 3 cm), but eight failures reached within 5 cm and still failed the physical check.

Some trials never get close: four never reached 5 cm, and three aimed the arm at the wrong place.

Knowing the part is not enough. An oracle with true part geometry raised touch only from 1/18 to 6/18. Last-centimeter approach, orientation, and fingertip contact remain broken.

One exact contact label is a bad target. Changing that label dropped success from 8/39 to 4/39. Predicting the whole part region is better in principle, but held-out precision is still low.

Phase switching, the 2D gate, delta actions, and a separate gripper head are not the main bottleneck. Remaining work: part-region grounding, then last-centimeter execution.
