# Part2Action research update — October 5, 2026

Copy each slide's bullets into Google Slides. Speaker notes and asset links are
provided separately. GIFs are in the unified root results folder.

## Slide 1 — Part2Action: DP3 pilot results

Trained official DP3-S on scissors demonstrations; evaluated checkpoint 1500.
Five test1 trials across five task types, with ground-truth part-cloud input.
One success: grasp the scissors at its left part, completed in 83 steps.
Four trials reached the 250-step limit; three achieved partial task completion.
These results motivated checking the data, labels, and evaluation pipeline.

Speaker notes: This is a small pilot, not a reproduction of the paper's full
benchmark. Poor performance does not establish that the dataset is broken.
The successful task was previously misidentified as screw touch; video metadata
confirms left-part grasp.

Visuals:
[Successful grasp](../../../results/partinstruct/visualizations/dp3_success/dp3_scissors_left_grasp_success.gif)
[Failed screw touch](../../../results/partinstruct/visualizations/dp3_failures/dp3_scissors_touch_screw_failed.gif)
[Failed grasp/move/release](../../../results/partinstruct/visualizations/dp3_failures/dp3_scissors_grasp_move_release_failed.gif)

## Slide 2 — Part2Action: Contact-gate diagnosis

Reviewed recorded demonstrations and derived surface-contact targets.
Our contact distance gate activates when TCP-to-target distance is at most 3 cm.
In the first 32 demonstrations per object: no sampled touch segment activated
  the gate (0/57); scissors grasp also had no activation (0/16).
Bottle and mug grasps activated more often: 22/23 and 21/24 segments.
Comparing 3, 5, 7, and 10 cm showed that larger thresholds activate farther away.

Speaker notes: This is the contact distance gate, not the skill selector or
visual part gate. TCP is a gripper reference point, not the fingertip. A finger
may touch while TCP remains farther than 3 cm from the surface. The observation
suggests the distance criterion and surface-to-TCP geometry need reconsideration.
It does not prove that a larger action offset is the solution. Threshold and
bounded action offset are separate parameters. These are diagnostic comparisons,
not retrained threshold ablations or model rollout results.

Visuals:
[Bottle touch: synchronized thresholds](../../../results/diagnostics/contact_label_audit/gate_gifs/bottle_touch_obj_demo_11_thresholds.gif)
[Scissors grasp: synchronized thresholds](../../../results/diagnostics/contact_label_audit/gate_gifs/scissors_grasp_obj_demo_1_thresholds.gif)

## Slide 3 — Part2Action: Rethinking contact switching

A learned selector could use vision, task, robot state, and recent motion.
It could learn when to use regular versus contact-relative actions.
However, physical interaction is not defined by one fixed TCP-to-surface distance.
Learning the existing 3 cm rule would reproduce its limitation.
A more complex gate would not fix an unsuitable contact-relative action representation.

Speaker notes: Phase labels are also estimates, not measured physical contacts.
Before adding a selector, ask whether explicit switching is necessary. A unified
policy can receive predicted contact geometry throughout the motion and learn
how it relates to actions. We retain learned routing as a later comparison.

## Slide 4 — Part2Action: Contact-conditioned actions

Keep visual part grounding, contact prediction, skill hierarchy, and temporal encoding.
Replace the hard contact switch and separate bounded-offset branch in a new variant.
Feed predicted contact position relative to TCP into the MLP action head.
One action head predicts XYZ, wrist, and gripper throughout the motion.
Use predicted geometry in both training and deployment; ground-truth part/contact
  labels remain auxiliary supervision only.

Diagram:

```text
RGB history + scene cloud + task + TCP + finger opening
                  ↓
Part grounding + contact prediction + skill hierarchy
                  ↓
Observation features + predicted contact relative to TCP
                  ↓
Unified MLP → eight-action chunk
```

Speaker notes: Earlier full-path contact-relative output was contact plus a
learned offset. The new variant uses contact as an input instead of an output
anchor. Explicit contact-conditioning features are detached from action-loss
backpropagation in the first experiment. We have not added calibrated contact
confidence. Existing variants/checkpoints are preserved.

## Slide 5 — Part2Action: Controlled experiments

Implementation is ready; seven CPU checks passed, including model-forward
  checks with stub encoders and existing near-contact behavior tests.
Train matched two-frame models: without versus with contact conditioning.
Keep data split, labels, losses, training budget, and rollout protocol fixed.
Create an episode-level validation split and run a short GPU smoke test first.
If conditioning improves closed-loop behavior, compare diffusion or flow matching
  against the MLP with matched action-chunk horizons.

Speaker notes: The old baseline used one frame, so it does not isolate contact
conditioning from temporal-history effects. Our MLP already predicts eight-action
chunks; diffusion is a decoder change, not the first introduction of chunking.
No unified-model training has been submitted. CPU checks are not evidence of
improved robot performance.

## Slide 6 — Part2Action: Runs and dataset expansion

Baseline and near-contact 39-trial evaluations were pending; after GPU eligibility
  was broadened, both started but failed during setup. No new scores are available.
Failure: stale PartInstruct config path. Fix the path and rerun the matched protocol.
Stopped DP training to prioritize the contact-policy investigation; checkpoint 2500 retained.
DP3 provides an initial comparison, but it is not enough to establish DP's performance.
Full released demonstration download is verified complete: 11 training object
  categories, approximately 72.23 GB of HDF5 demonstrations.
Initial architecture ablations retain the original four objects; expand to all
  released categories as a separate experiment.

Speaker notes: Stopping DP was a project-priority decision, not a conclusion that
DP3 establishes the performance of both methods. Dataset files are on durable
liverobotics storage. Existing training configs do not automatically include new
categories. Official five evaluation splits remain unchanged; download expansion
does not automatically schedule broader evaluations.

## Slide 7 — Part2Action: Next steps

Repair the evaluation setup and recover baseline/near-contact reference results.
Freeze a train/validation episode split for the four-object controlled experiment.
Run unified-policy smoke tests, then matched control/contact training.
Evaluate success by object and skill, contact localization, motion errors, and latency.
Expand dataset coverage and test a stronger action decoder after the initial comparison.

Sources of local evidence: [research plan](../RESEARCH_PLAN.md),
[contact audit](../CONTACT_LABEL_AUDIT.md),
[DP3 visualization provenance](../../../results/partinstruct/visualizations/dp3_success/manifest.json),
[download manifest](../../../results/data/full_dataset_download/manifest.json).
