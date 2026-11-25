# Part2Action Meeting Status

Date: 2026-09-13

Speaker note: All failure GIFs are **oracle eval** of the same 30-epoch checkpoint (`world_geometry_30epoch_6529250`). Oracle gives the policy **true part geometry**. They are not the missing baseline videos. Ground-truth HDF5 demos succeed; show them only as the comparison after each fail slide.

Copy-paste paths into Google Slides via Insert → Image → Upload, or drag the file from the file explorer.

Suggested order for each problem: **explain → failed GIFs → matching GT GIFs**.

---

## Slide 1 — Where We Are

Baseline world-geometry policy: **8/39** PartGym success (20.5%).

Discarded: skill gate, oracle phase termination, delta actions, split gripper, skill-specific contact labels, and the extra point-part head (held-out 33% P / 62% R / 43% F1; physical eval 6/39).

Queued now: 15-epoch **contact-attention part loss** (job `6562294`). Same contact head. No new module. A100 or H100, whichever frees first.

A40 job `6562370` is re-recording the missing **baseline** 39-trial videos (not these oracle clips).

---

## Slide 2 — Problem 1: Looking at the wrong place

The robot often does not put its hand on the **requested part**, especially for **touch**.

On the 8/39 baseline: touch contact was within 3 cm of the part in only **6/18** visits. Thirteen failures aimed at the wrong region.

This is the perception / grounding problem. The queued training is meant to help **this**.

We **do not** have baseline GIFs of “wrong part.” Those videos were deleted. The next slide is the closest visual: a failed **touch** where the hand never arrives.

---

## Slide 3 — Failed GIFs for Problem 1 (never reaches)

Say: “Policy fail. Oracle still never gets the hand to the part.”

**Touch, never reaches**

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_2_test1_trial00_seed142/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/mug_2_test1_trial01_seed543/rollout.gif`

---

## Slide 4 — Ground-truth GIFs for Problem 1

Say: “This is the expert demo, not the policy. Touch actually happens.”

**Expert touch**

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/bottle_demo_1000_touch_the_bottle_at_its_lid/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/mug_demo_1007_touch_the_mug_at_its_right/rollout.gif`

---

## Slide 5 — Problem 2: Close is not enough

Even when the hand is **near** the part, grasp/touch still fails: fingers slip, or the gripper never really clamps.

Baseline: grasp contact within 3 cm in **19/21** visits, but physical grasp only **10/21**. Eight failures reached within 5 cm and still failed the physical check.

Oracle (true part given) only raised touch from **1/18 to 6/18**. Knowing the part does not fix last-centimeter angle, fingertips, and close timing.

The ±2 cm residual does **not** rotate the wrist or close the gripper.

---

## Slide 6 — Failed GIFs for Problem 2 (near, still fails)

Say: “Hand gets close, then slides off or never clamps.”

**Close, still fails**

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/mug_3_test1_trial02_seed644/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_3_test1_trial02_seed244/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/pliers_1_test1_trial01_seed843/rollout.gif`

---

## Slide 7 — Ground-truth GIFs for Problem 2

Say: “Expert grasp: the hand arrives and actually closes.”

**Expert grasp / pick-and-move**

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/02_grasp/mug_demo_1005_grasp_the_mug_at_its_handle/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/bottle_demo_0_grasp_the_bottle_at_its_neck/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/pliers_demo_104_grasp_the_pliers_at_its_leg/rollout.gif`

---

## Slide 8 — Problem 3: The ±2 cm residual cannot save a big miss

We still use a **contact residual**: after the main 7D action, add a small XYZ nudge, clipped to **±2 cm**. Orientation and gripper are unchanged.

We added it because unbounded contact steering made the arm jump. PartInstruct does **not** have this module. It is our safety patch, not a measured 8/39 win.

If the main command is already 10–15 cm off, 2 cm cannot recover it. In failing trials the residual is often **pegged at the cap**.

---

## Slide 9 — Failed GIFs for Problem 3 (far miss)

Say: “Lid grasp. Hand stays far. Residual maxed. Gripper stays open.”

**Far miss**

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_1_test1_trial02_seed44/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_1_test1_trial01_seed43/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/mug_1_test1_trial00_seed442/rollout.gif`

---

## Slide 10 — Ground-truth GIFs for Problem 3

Say: “Expert lid / mug grasp. This is what the cloned trajectory looks like when it works.”

**Expert grasp lid / mug**

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/bottle_demo_1_grasp_the_bottle_at_its_lid/rollout.gif`

`/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/mug_demo_0_grasp_the_mug_at_its_right/rollout.gif`

---

## Slide 11 — What is training now

**Not** “more epochs of the same recipe.” Offline losses already fit (phase ~99%, contact L1 tiny). More of that will not fix slipping.

Current job: **15 epochs**, init from the 8/39 checkpoint.

Same 1024 scene points in, same contact XYZ out. **New loss only:** part-restricted Gaussian on `contact_attention` (3 cm). Labels stay; no new head; `use_point_part_head: false`.

Goal: make attention look at the requested part, then rerun the same **39** physical trials.

Job `6562294`, pending, 1 GPU, A100 or H100, 24 h limit.

---

## Slide 12 — Next step after that eval

If 39-trial success does not move: contact attention was not the bottleneck; do not stack another perception head.

If success moves but grasp still slips: **last-centimeter execution** — aim the TCP at the contact, better wrist, close only when near, hold. Replace the 2 cm residual with contact-relative actions. That is step 2, not another architecture ablation.

Not next: more epochs, phase switching, 2D gate, delta actions, split gripper. Already checked.

---

## Appendix — Copy blocks

```
# Problem 1 FAIL — touch never reaches
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_2_test1_trial00_seed142/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/mug_2_test1_trial01_seed543/rollout.gif

# Problem 1 GT
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/bottle_demo_1000_touch_the_bottle_at_its_lid/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/01_touch/mug_demo_1007_touch_the_mug_at_its_right/rollout.gif

# Problem 2 FAIL — close, still fails
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/mug_3_test1_trial02_seed644/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_3_test1_trial02_seed244/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/pliers_1_test1_trial01_seed843/rollout.gif

# Problem 2 GT
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/02_grasp/mug_demo_1005_grasp_the_mug_at_its_handle/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/bottle_demo_0_grasp_the_bottle_at_its_neck/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/pliers_demo_104_grasp_the_pliers_at_its_leg/rollout.gif

# Problem 3 FAIL — far miss
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_1_test1_trial02_seed44/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/bottle_1_test1_trial01_seed43/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/part2action/archive/evaluations/world_geometry_30epoch_6529250_oracle_eval/hierarchical/videos/mug_1_test1_trial00_seed442/rollout.gif

# Problem 3 GT
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/bottle_demo_1_grasp_the_bottle_at_its_lid/rollout.gif
/u/xna8aw/workspace/research_projects/interactive-robotics/results/data/gt_demo_videos/03_pick_and_move/mug_demo_0_grasp_the_mug_at_its_right/rollout.gif
```
