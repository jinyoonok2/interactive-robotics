# Part-to-Action Direction (Professor Feedback)

Date: 2026-04-27  
Project: Interactive Robotics / affordance pipeline

## Why this note

Current plan separates:
1) learn part-aware affordance map, then  
2) train policy on top.

Professor feedback is important: affordance heatmaps may be too coarse for action policy.

Example:
- "lid" may be detected correctly,
- but successful manipulation may require "grasp lid from side edge" (contact geometry + approach direction),
- which is different from just knowing the lid region.

So the key question is:
Can we train from data that maps part understanding directly to executable actions, instead of only heatmap supervision?

---

## Core hypothesis

For fine-grained interaction, representation should encode:
- part identity (what part),
- interaction role (how to use the part),
- action geometry (where to contact, from which direction, with what pose),
- temporal order for multi-step tasks.

A pure 2D affordance map is often missing the last two.

---

## Dataset options beyond AGD20K

AGD20K is valuable for part-aware grounding, but it is mostly perception supervision (heatmaps), not policy-level action supervision.

### 1) PartInstruct (strong match to this problem)

What it adds:
- Part-level language instructions for manipulation tasks.
- Demonstrations for action learning (not only masks).
- Goal predicates such as part grasping/contacts/orientation.
- Includes part labels and part-level task structure.

Why useful here:
- Directly addresses "lid vs side of lid" style ambiguity.
- Supports learning the mapping: instruction + scene -> action sequence.

Notes:
- Simulation benchmark; morphology/domain may differ from Fetch + Habitat setup.

Links:
- [PartInstruct project page](https://partinstruct.github.io/)
- [PartInstruct paper](https://arxiv.org/abs/2505.21652)

### 2) GAPartManip / GAPartNet-style articulated datasets

What they add:
- Part-centric articulated manipulation signals.
- Actionable interaction pose style annotations for parts.

Why useful:
- Helpful for learning contact/approach priors on articulated parts (handles, lids, doors, drawers).

Link:
- [GAPartManip paper](https://arxiv.org/abs/2411.18276)

### 3) ManiSkill2 (action-centric simulation benchmark)

What it adds:
- Large-scale demonstrations for manipulation tasks.
- Good infrastructure for IL/RL and articulated object tasks.
- Useful for training policies with richer action supervision than static heatmaps.

Why useful:
- Practical environment for curriculum-style training and policy debugging.

Link:
- [ManiSkill2 paper](https://arxiv.org/abs/2302.04659)

### 4) Open X-Embodiment / BridgeData V2 (broad policy pretraining)

What they add:
- Large language-conditioned action datasets for visuomotor policy pretraining.
- Better general manipulation priors.

Limit:
- Usually not strongly part-annotated at the granularity needed for "specific side of specific part."

Why still useful:
- Great for initializing policy backbone, then fine-tune on part-specific datasets.

Links:
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata)

---

## Recommended training strategy (curriculum, simple -> hard)

Professor suggestion aligns with a strong curriculum:

1) **Stage A: Part grounding pretrain**
- Keep AGD20K-style supervision for part-awareness.
- Learn robust language-part alignment.

2) **Stage B: Part-to-action imitation**
- Train policy on part-centric demonstration datasets (prefer PartInstruct-like data).
- Objective: instruction + observation -> action trajectory.
- Add auxiliary losses:
  - contact point prediction,
  - approach vector prediction,
  - grasp pose prediction.

3) **Stage C: Habitat adaptation**
- Transfer to your Fetch/Habitat stack.
- Use domain randomization + small expert demo set + RL fine-tuning.

4) **Stage D: Hard-case curriculum**
- Start with single-step grasp tasks.
- Move to articulated and multi-step tasks (open lid -> grasp side -> move).

---

## Practical idea for your codebase

Instead of replacing the current pipeline immediately, add an action-oriented branch:

- Keep current outputs:
  - part heatmap,
  - part point cloud.
- Add new outputs:
  - contact candidates on part,
  - approach direction distribution,
  - grasp pose proposal conditioned on instruction.

Then train policy with these action-oriented targets (or distill them into policy features).

This creates a bridge from "where" to "how."

---

## Immediate next steps (low risk)

- [ ] Build a small benchmark set in your existing Habitat scenes with "part is right but grasp fails" cases.
- [ ] Re-label these failures with action-specific targets (contact side, approach vector, grasp orientation).
- [ ] Evaluate whether adding action losses improves success over heatmap-only baseline.
- [ ] Pilot one external part-action dataset first (PartInstruct), then decide integration depth.

---

## Bottom line

Yes, the professor's point is valid and important:
- affordance-only supervision can be insufficient for precise manipulation,
- especially when multiple valid points exist on the same part label.

Best direction is not "affordance OR policy", but:
- **affordance grounding + direct part-to-action supervision + curriculum training**.

---

---

## Overall project flow: from affordance pipeline to part2action

```
Affordance Pipeline (existing)
│
│  Goal: robot understands WHICH part to interact with.
│  Method: AGD20K-style heatmap supervision on top of DINOv2 features.
│  Output: 2D part affordance heatmap ("here is the lid region").
│  Gap: heatmap tells WHERE, not HOW (contact geometry, approach direction).
│
└──► Professor feedback (2026-04-27)
       "affordance map may not be accurate enough to refine action policy.
        robot knows 'lid' but needs 'grasp lid from the side edge' –
        that is a different signal."
       
       Parallel to VLA curriculum insight:
         VLAs train simpler tasks first, then harder ones.
         Similarly: part grounding → part-conditioned action.
         
└──► Part2Action concept validation (part2action/)
       Goal: test whether adding contact/approach/action supervision
             on top of the same frozen backbone improves over heatmap-only.
       If yes → integrate into Habitat with Fetch robot.
       If no  → the bottleneck is representation scale, not supervision type.
```

---

## Dataset: PartInstruct (SCAI-JHU)

**Why chosen:** Only publicly available dataset that gives per-timestep
part-level language instructions alongside full action demonstrations and
part segmentation masks in simulation — exactly the supervision needed to
test the professor's hypothesis.

**What it contains (per demo):**
- `agentview_rgb` — 300×300 RGB images from a fixed overhead camera
- optional additional observations in the full benchmark, including depth / point clouds
  such as `agentview_pcd`, `agentview_part_pcd`, `wrist_rgb`, and `wrist_pcd`
- `agentview_part_mask` — binary part segmentation mask (which pixels belong to the target part)
- `actions` — (T, 7) end-effector deltas: Δpos(3) + Δaxis-angle(3) + gripper(1)
- `skill_instructions` — per-timestep language string, e.g. `"Touch the scissors at its screw"`
- `gripper_state` — binary open/closed per timestep
- `joint_states` — robot joint angles per timestep

**Subset used for concept validation:**
- `scissors.hdf5` — 369 demos, 1.8 GB
- `pliers.hdf5` — ~300 demos, 2.1 GB
- Total: ~36,600 training samples at stride=2 (one sample = one timestep)

These were chosen as the two smallest categories in the dataset (~4 GB combined
vs 83 GB full), sufficient to test the hypothesis on a single RTX 4060.

**Reference: PartInstruct's own baselines**
- PartInstruct evaluates diffusion-policy style baselines, including DP-S and DP3.
- Their configs use short temporal observation history (`n_obs_steps: 2`).
- DP-S uses a ResNet-style RGB image encoder + T5-small language encoder.
- DP3 uses point-cloud / 3D features + T5-small language encoder.
- Their focus is action-policy benchmarking; they do not expose a separate
  heatmap/contact/approach head structure like this project.

This project intentionally differs by keeping DINOv2 patch tokens and explicit
spatial heads so that the model remains connected to the affordance pipeline.

---

## Architecture

```
Input: RGB image (300×300) + language instruction (string)
         │                          │
         ▼                          ▼
 FrozenDINOv2                 FrozenFlanT5
 ViT-S/14 reg4                base encoder
 (21M params,                 (250M params,
  frozen)                      frozen)
 Output: (B, 324, 384)        Output: (B, T_text, 512)
  324 = 18×18 patch tokens     T_text = tokenized instruction length
         │                          │
  Optional TemporalEncoder          │
  (for n_obs_steps > 1)             │
  (B,T,324,384) → (B,324,384)       │
         │                          │
         └──────────┬───────────────┘
                    ▼
         CrossAttentionFusion
         (visual patches attend to text tokens)
         2 layers, hidden_dim=256, 4 heads
         ~1.5M trainable parameters
         Output: (B, 324, 256)  ← task-grounded patch features
                    │
         ┌─────────┼──────────────────┬──────────────────────────┐
          ▼         ▼                  ▼                          ▼
    HeatmapHead  ContactHead2D    ApproachHead             Action head
    2D conv      MLP + sigmoid    MLP + L2-norm            selected by config:
    upsample     Output: (B,2)    Output: (B,3)            - MLP ActionChunkHead
    Output:      normalized       3D unit                  - DiffusionActionHead
    (B,252,252)  image coords     approach vec             Output: (B,8,7)
```

**Key design choices:**
- DINOv2 and T5 are fully frozen — only the fusion + heads are trained (~1.5M params).
- This keeps GPU memory low and allows fast training on an 8 GB GPU.
- The frozen DINOv2 features are already strong part discriminators; the fusion module
  learns to modulate them based on the instruction.
- The action head is modular: `action_head_type: mlp` keeps the simple baseline,
  while `action_head_type: diffusion` trains a small denoising head from scratch
  on action chunks. This is not an external pretrained diffusion model.
- The heatmap head does not feed into contact / approach / action heads directly.
  All heads read the same shared cross-attention features. Heatmap supervision is
  an auxiliary grounding loss that keeps the representation spatially part-aware.
- The temporal encoder is modular: `temporal_encoder_type: none` keeps current
  single-frame behavior, while `temporal_encoder_type: transformer` uses a short
  RGB history (`n_obs_steps: 2`) and compresses DINOv2 frame-token grids back to
  the same patch-token shape expected by cross-attention.

**Why patch tokens instead of one fixed vector:** DINOv2 outputs one token per
image patch, so the model preserves "where" information. This is important for
heatmaps and contact localization. A fixed global vector, common in many policy
baselines, is compact and useful for action generation but weaker for producing
inspectable spatial predictions.

---

## Training

**Five tracks are available to test the hypothesis:**

| | Heatmap | PartAction-MLP | PartAction-Diffusion | Temporal-PartAction-MLP | Temporal-PartAction-Diffusion |
|---|---|---|---|---|---|
| Config | `heatmap_real.yaml` | `part_action_mlp_real.yaml` | `part_action_diffusion_real.yaml` | `temporal_part_action_mlp_real.yaml` | `temporal_part_action_diffusion_real.yaml` |
| Temporal input | single frame | single frame | single frame | 2-frame history | 2-frame history |
| Active heads | heatmap only | heatmap + contact + approach + MLP action | heatmap + contact + approach + diffusion action | heatmap + contact + approach + MLP action | heatmap + contact + approach + diffusion action |
| Action loss | none | smooth-L1(action chunk) | MSE(noise prediction) | smooth-L1(action chunk) | MSE(noise prediction) |
| Purpose | heatmap-only grounding baseline | does action supervision help? | does diffusion action improve over MLP? | does temporal history help MLP action? | strongest modular track: temporal + diffusion |

**Training setup:**
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-5
- Batch size: 8, gradient clip: 1.0
- Mixed precision: bfloat16 (not float16 — avoids overflow with DINOv2/T5)
- Epochs: 30, ~4,575 batches/epoch, ~16 it/s on RTX 4060
- Estimated time: ~2.5 hours per track

**Loss functions:**
```
heatmap:  BCE(predicted_logits, part_mask_resized)
contact:  L1(predicted_xy, derived_contact_xy)
approach: 1 - cosine_similarity(predicted_dir, derived_approach_dir)
MLP action:        smooth_L1(predicted_chunk, gt_action_chunk)
diffusion action:  MSE(predicted_noise, sampled_noise)
```

**Temporal support:** PartInstruct demonstrations are temporal sequences, and
PartInstruct's own DP / DP3 baselines use observation history (`n_obs_steps: 2`).
The current branch now supports the same style of short history through a modular
`TemporalEncoder`. It consumes multiple DINOv2 frame-token grids
`(B, T, N_patches, D)` and compresses them back to `(B, N_patches, D)` before
`CrossAttentionFusion`. That keeps the rest of the architecture unchanged while
adding motion context.

---

## How evaluation ground truth is derived

PartInstruct does **not** provide explicit contact point or approach direction
annotations. These are derived deterministically from the raw demonstration data
in `part2action/data/targets.py`:

**Contact point (`contact_xy_norm`):**
1. Find the timestep `contact_t` where the gripper first closes (gripper signal transitions from open → closed).
2. At that timestep, integrate the cumulative position deltas from the action sequence to estimate the end-effector position in 3D.
3. Project to normalized 2D image coordinates [0, 1] × [0, 1].

**Approach direction (`approach_dir`):**
1. Take the 4 timesteps just before `contact_t` (the pre-contact window).
2. Compute the mean of the positional action deltas over that window (Δx, Δy, Δz).
3. L2-normalize to get a unit vector representing the direction the robot was moving when it reached the part.

**Why this works:** Expert demonstrations in PartInstruct are short, goal-directed trajectories. The pre-contact motion reliably encodes how the robot was oriented when approaching the part, which is exactly the "grasp from the side" vs "grasp from above" signal the professor described.

**Offline evaluation metrics (no simulator needed):**
- Heatmap: IoU between predicted heatmap (thresholded at 0.5) and ground truth part mask
- Contact: L1 distance in normalized image coordinates
- Approach: cosine similarity between predicted and derived approach vector
- Action: smooth-L1 between predicted 8-step chunk and ground truth

---

## Implementation status (2026-05-04)

Concept validation scaffolding is implemented on real PartInstruct data. Long
training was paused by the user and can be restarted later.

**Completed:**
- `part2action` conda env, isolated from `habitat-grasp`.
- PartInstruct HDF5 adapter verified against real dataset schema.
- Derived ground truth (contact + approach) confirmed numerically sound.
- Model architecture smoke-tested in fp32 and bfloat16.
- Fixed fp16 AMP overflow (switched to bfloat16; fp32 loss = 0.693 at init as expected).
- Dataset downloaded: `scissors.hdf5` (1.8 GB) + `pliers.hdf5` (2.1 GB), 36,607 samples.
- `heatmap_real.yaml` was smoke-tested on real data; early loss dropped
  `0.614 → ~0.004` during epoch 1 before the run was intentionally stopped.
- Added modular action-head support:
  - `action_head_type: mlp` keeps the original direct action chunk head.
  - `action_head_type: diffusion` enables a small DDPM-style denoising head.
- Added `configs/part_action_diffusion_real.yaml` for the diffusion-head track.
- Added modular temporal encoder support:
  - `n_obs_steps: 1` + `temporal_encoder_type: none` keeps single-frame behavior.
  - `n_obs_steps: 2` + `temporal_encoder_type: transformer` enables short observation history.
- Added `configs/temporal_part_action_mlp_real.yaml` and
  `configs/temporal_part_action_diffusion_real.yaml`.
- Updated training / evaluation code to instantiate the selected action head.
- Python compile, action-head smoke test, and linter checks passed.

**Still pending:**
- Full real-data training runs for all five tracks.
- Offline evaluation of all checkpoints.
- Optional point-cloud branch to add explicit 3D geometry.
- Go / no-go decision for Habitat integration.

---

## Go / no-go decision rule

After training the tracks on real PartInstruct demos, decide based on
this rule:

| Outcome on hard-subset rollouts | Decision |
|---|---|
| `part_action_mlp_real` beats `heatmap_real` by ≥10 pp task success | **Go**: start Habitat integration phase (action-space adapter for Fetch, domain randomization, fine-tuning). |
| `part_action_mlp_real` beats `heatmap_real` by 3–10 pp | **Investigate**: try larger fusion + more demos before committing to Habitat. |
| `part_action_mlp_real` ties or loses to `heatmap_real` | **No-go**: bottleneck is elsewhere (representation or data scale). Revise [PROPOSAL.md](PROPOSAL.md) before more engineering. |

For the added diffusion track, compare:
- `part_action_mlp_real` vs `heatmap_real`: does action-geometry supervision help over heatmap-only?
- `part_action_diffusion_real` vs `part_action_mlp_real`: does distributional action generation improve over direct MLP action prediction?
- `temporal_part_action_mlp_real` vs `part_action_mlp_real`: does temporal history improve the MLP action track?
- `temporal_part_action_diffusion_real` vs diffusion / temporal MLP tracks: does combining temporal history with diffusion give the strongest action prediction?
