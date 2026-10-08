# Part2Action Meeting Status

Date: 2026-09-21

Speaker note: Real policy GIFs only (not oracle). Copy-paste paths into Google Slides via Insert → Image → Upload.

Order: current limit (broken from approach; even “successes” fail contact) → why contact-relative → new baseline vs contact-relative → same-trial GIFs → next.
---

## Slide 1 — Current model limit: broken from approach

The failure starts at approach, not only at the last centimeter. Many trials never put the hand on the object: far miss, hover, residual pegged at ±2 cm. Phase stays 0. Movement never starts.

The few PartGym “successes” are still a contact problem. Task-1 hold fires on proximity. The gripper stays almost open. It does not look like a touch or a hold.

So: most runs die on the way in. The ones that score still do not make contact.

Say: “Approach first. The wins are not real contact either.”

Approach already fails
`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/scissors_2_test1_trial00_seed1242/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/bottle_2_test1_trial00_seed142/rollout.gif`

Scored success — still not contact
`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/mug_1_test1_trial02_seed444/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/pliers_1_test1_trial02_seed844/rollout.gif`

Expert contact
`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/scissors_demo_0_touch_the_scissors_at_its_screw/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/bottle_demo_1000_touch_the_bottle_at_its_lid/rollout.gif`

---

## Slide 2 — Why contact-relative

Intent: make contact the thing the hand is defined from, not a ±2 cm afterthought.

We wanted “fingers 3 cm above the neck” instead of “go to world `(0.05, 0.02, 0.21)`.” If the predicted part moves, the same offset should follow it. That is last-centimeter geometry, not another perception head.

So we replaced cloned XYZ:

hand XYZ = predicted contact + learned offset
Wrist / gripper still cloned. Same 1024-in, one-contact-out head. No PartGym oracle at rollout. Training uses demo contact labels only.

We did not init from 8/39. That would have been a new module on residual-trained weights. Contact-relative is 30 epochs from scratch, matched to the new no-residual baseline.

---

## Slide 3 — New baseline vs contact-relative

Two matched 30-epoch from-scratch runs. Freeze DINO/T5 only. Same data. 8/39 is not in this pair — that checkpoint was residual trained three times (buggy 15 → residual 15 → residual 30).

New baseline. 8/39 stack minus the ±2 cm head. Cloned world XYZ only. 2/39.
Contact-relative. Same recipe. XYZ = predicted contact + offset (40 cm tanh). 2/39.
Same score. Not a method win. Do not compare either to 8/39.

New baseline successes
`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_no_residual_6570908_eval/hierarchical/videos/mug_1_test1_trial02_seed444/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_no_residual_6570908_eval/hierarchical/videos/pliers_1_test1_trial02_seed844/rollout.gif`

---

## Slide 4 — Same trial: what contact-relative did

Scissors screw, seed 1242. New baseline still walks. Contact-relative stops far away and vibrates.

Predicted contact is on the scissors. Not a 1024-point input change. Offset saturates at +40 cm per axis (home is ~80 cm out). Command = contact + 40 cm → park and chatter.

We put a last-centimeter formula on the whole path. Offline losses looked fine. Physical eval 2/39, same as the new baseline. The two “successes” are task-1 proximity; gripper stays open.

New baseline
`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_no_residual_6570908_eval/hierarchical/videos/scissors_2_test1_trial00_seed1242/rollout.gif`

Contact-relative
`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_contact_relative_6571099_eval/hierarchical/videos/scissors_2_test1_trial00_seed1242/rollout.gif`

Contact-relative “success” (does not look like a hold)
`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_contact_relative_6571099_eval/hierarchical/videos/mug_1_test1_trial02_seed444/rollout.gif`

---

## Slide 5 — Next

Keep cloned XYZ for the walk in. Use contact only near the part (last centimeter). That was the original intent.

Not: raise the 40 cm cap (that is cloning again).
Not: more epochs of the same full-path offset.
Not: another architecture ablation.
Not: treat 8/39 as the fair baseline.

---

## Appendix — Copy blocks

```
# Limit: approach fails; scored wins still not contact
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/scissors_2_test1_trial00_seed1242/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/bottle_2_test1_trial00_seed142/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/mug_1_test1_trial02_seed444/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_baseline_videos/hierarchical/videos/pliers_1_test1_trial02_seed844/rollout.gif

# GT contact
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/scissors_demo_0_touch_the_scissors_at_its_screw/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/bottle_demo_1000_touch_the_bottle_at_its_lid/rollout.gif

# New baseline
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_no_residual_6570908_eval/hierarchical/videos/mug_1_test1_trial02_seed444/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_no_residual_6570908_eval/hierarchical/videos/pliers_1_test1_trial02_seed844/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_no_residual_6570908_eval/hierarchical/videos/scissors_2_test1_trial00_seed1242/rollout.gif

# Contact-relative
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_contact_relative_6571099_eval/hierarchical/videos/scissors_2_test1_trial00_seed1242/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_contact_relative_6571099_eval/hierarchical/videos/mug_1_test1_trial02_seed444/rollout.gif
```
