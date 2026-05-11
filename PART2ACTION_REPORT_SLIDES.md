# Part2Action Progress Report Slides

Concise slide draft for explaining the PartInstruct pivot and the current Part2Action framework.

---

## Slide 1: Motivation

**Header:** From Part Localization to Part-Aware Action

- Existing affordance pipeline localizes useful object parts.
- But manipulation also needs contact point, approach direction, and action sequence.
- Professor feedback: heatmaps are useful, but too coarse for policy refinement.

**Takeaway:** Extend affordance grounding from "where is the part?" to "how should the robot interact?"

---

## Slide 2: Why PartInstruct?

**Header:** Dataset That Connects Parts to Actions

PartInstruct provides:
- RGB observations: `agentview_rgb`
- Part masks: `agentview_part_mask`
- Language instructions: `skill_instructions`
- Robot actions: `actions`
- Gripper state: `gripper_state`

Current subset:
- `scissors.hdf5` + `pliers.hdf5`
- ~36,607 timestep samples

**Takeaway:** PartInstruct gives structured supervision from part understanding to robot action.

---

## Slide 3: What We Learn From PartInstruct

**Header:** Heatmap Alone Is Not Enough

From the same demonstrations, we can supervise:
- **Heatmap:** where is the instructed part?
- **Contact:** where should the robot first touch?
- **Approach:** from what direction should it move?
- **Action chunk:** what are the next 8 robot actions?

**Takeaway:** Part-aware action learning needs both grounding and action geometry.

---

## Slide 4: PartInstruct Baselines vs Ours

**Header:** Same Dataset, Different Model Question

PartInstruct baselines:
- DP-S: ResNet RGB encoder + T5-small + diffusion policy
- DP3: point-cloud encoder + T5-small + diffusion policy
- Uses temporal history (`n_obs_steps=2`)

Our direction:
- Frozen DINOv2 patch tokens
- Frozen Flan-T5-base
- Cross-attention fusion
- Explicit heatmap/contact/approach/action heads

**Takeaway:** Their focus is policy benchmarking; ours is interpretable part-grounded action learning.

---

## Slide 5: Architecture Diagram

**Header:** Modular Part2Action Architecture

```text
RGB frame(s)
   │
   ▼
Frozen DINOv2
   │
   ├─ single frame:    (B, N, 384)
   └─ temporal frames: (B, T, N, 384)
             │
             ▼
     Optional TemporalEncoder
     (B, T, N, 384) → (B, N, 384)

Instruction
   │
   ▼
Frozen Flan-T5 → text tokens

visual tokens + text tokens
        │
        ▼
CrossAttentionFusion
        │
        ▼
shared part-grounded features
        │
        ├── HeatmapHead
        ├── ContactHead2D
        ├── ApproachHead
        └── ActionHead
              ├── MLPActionHead
              └── DiffusionActionHead
```

**Takeaway:** DINOv2/T5 are frozen; fusion, temporal encoder, and heads are trainable.

---

## Slide 6: Why Keep Heatmap?

**Header:** Heatmap as Grounding, Not the Final Policy

- Heatmap output does **not** feed into other heads.
- All heads read the same shared cross-attention features.
- Heatmap loss keeps the shared representation spatially part-aware.
- It also gives an interpretable debugging signal.

**Takeaway:** Heatmap supervision anchors the action heads to the correct part.

---

## Slide 7: Modular Experiment Tracks

**Header:** Config-Based Architecture Variants

| Config | Input | Action Head | Purpose |
|---|---|---|---|
| `heatmap_real.yaml` | single frame | none | grounding baseline |
| `part_action_mlp_real.yaml` | single frame | MLP | action supervision |
| `part_action_diffusion_real.yaml` | single frame | diffusion | distributional action |
| `temporal_part_action_mlp_real.yaml` | 2 frames | MLP | temporal context |
| `temporal_part_action_diffusion_real.yaml` | 2 frames | diffusion | strongest current track |

**Takeaway:** We can compare modules without rewriting the model.

---

## Slide 8: Heads and Losses

**Header:** What Each Head Predicts

| Head | Output | Supervision | Loss |
|---|---|---|---|
| Heatmap | 2D part map | part mask | BCE |
| Contact | `(x, y)` | derived contact point | L1 |
| Approach | 3D unit vector | derived approach direction | 1 - cosine |
| MLP Action | 8×7 action chunk | demo actions | Smooth-L1 |
| Diffusion Action | 8×7 action chunk | demo actions + noise | MSE noise loss |

**Takeaway:** Each head captures a different part-to-action signal.

---

## Slide 9: Derived Targets

**Header:** Contact and Approach From Demonstrations

PartInstruct does not directly label contact/approach, so we derive them:

- **Contact:** first gripper-close timestep → estimate end-effector contact point → project to 2D.
- **Approach:** average robot motion over 4 timesteps before contact → normalize to 3D direction.
- **Action:** directly slice next 8 actions from the demonstration.

**Takeaway:** Raw expert trajectories become explicit action-geometry supervision.

---

## Slide 10: Evaluation

**Header:** Offline Metrics Before Simulator Integration

Metrics:
- Heatmap IoU
- Contact L1 error
- Approach cosine similarity
- Action Smooth-L1

Current training chain:

```text
heatmap_real → evaluate
part_action_mlp_real → evaluate
part_action_diffusion_real → evaluate
```

**Takeaway:** Validate grounding/action geometry before expensive simulator integration.

---

## Slide 11: Project Meaning

**Header:** Extending the Affordance Pipeline

Original:

```text
pixels → DINOv2 features → part / affordance grounding
```

Now:

```text
pixels + language → part-grounded features
                 → heatmap + contact + approach + action
```

**Takeaway:** Part2Action extends the pipeline from perception-only grounding to action-aware interaction.

---

## Slide 12: Next Steps

**Header:** What We Validate Next

Immediate:
- Finish first 3 training tracks.
- Compare offline metrics.
- Check whether action-aware tracks beat heatmap-only.

Next:
- Train temporal tracks.
- Add point-cloud / 3D geometry branch.
- Complete PartGym rollout adapter.

**Decision:** If action-aware tracks clearly improve, proceed toward Habitat/Fetch integration.
