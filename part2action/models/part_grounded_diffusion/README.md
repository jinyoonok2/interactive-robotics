# Part-grounded Diffusion: existing policy family

This name describes the existing visual policy's part grounding and diffusion
action prediction. It is an organizational name for the current code, not a new
algorithm or a change to previously evaluated models.

| Entry point | Meaning | Original implementation |
| --- | --- | --- |
| `baseline.PartGroundedDiffusionPolicy` | Original visual baseline | [`../part2action_model.py`](../part2action_model.py), `Part2ActionModel` |
| `gt_pose.PartGroundedDiffusionWithPose` | Original policy with additive pose conditioning; GT source in the evaluated study | [`../visual_pose_policy_v1.py`](../visual_pose_policy_v1.py), `VisualPosePolicyV1` |
| `learned_pose_prototype.PartGroundedDiffusionPosePrototype` | Earlier learned target-pose branch with additive conditioning; one-step GPU smoke, not full training | [`../grounded_visual_pose_policy_v1.py`](../grounded_visual_pose_policy_v1.py), `GroundedVisualPosePolicyV1` |

The named entry points are **aliases to the exact existing classes**. They do not
wrap modules, change parameter prefixes, rename checkpoint architecture tags, or
add another policy. Package exports are lazy so importing the preservation guard
does not load model dependencies. For example:

```python
from models.part_grounded_diffusion import PartGroundedDiffusionPolicy
from models.part_grounded_diffusion.gt_pose import PartGroundedDiffusionWithPose
```

Existing flat files remain the implementation and compatibility paths. Their
bytes, original class names, trainer imports, config files, and old checkpoints
are preserved. The new [Pose-token Diffusion](../pose_token_diffusion/README.md)
workspace contains the proposed architecture and evolves separately.

The original three active recipes remain in `part2action/configs/`. The proposed
model keeps its scaffold config in its own package so it cannot be confused with
an existing runnable training recipe.

`preserved_sources.json` retains the original separation snapshot. From the
repository root, check it without loading Torch or the simulator:

```bash
PYTHONPATH=part2action python3 -m models.part_grounded_diffusion.verify_sources
```

The check covers the 30 recorded existing model/config/entrypoint files. It does
not hash result directories or all historical tools. Keep this snapshot fixed
during new-model implementation; review any intentional original-model change
separately.
