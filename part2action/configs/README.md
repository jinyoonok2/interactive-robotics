# Current Part2Action configurations

Three regular configuration files define the current visual-policy work:

| File | Role | Code that implements the structure |
| --- | --- | --- |
| [Original visual baseline](architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_transformer.yaml) | Shared original Transformer recipe and GT-study base settings | `models/part2action_model.py` |
| [GT-pose study](pose_learning/visual_oracle_pose_v1.json) | Experiment manifest: base YAML, initial checkpoint, matched two-epoch training and evaluation seeds | `scripts/run_visual_oracle_pose_v1.py`, `models/visual_pose_policy_v1.py` |
| [Shared grounding and predicted pose](pose_learning/grounded_visual_pose_v1.yaml) | Full model/data/loss/fine-tuning recipe for the new architecture | `scripts/train_grounded_visual_pose_v1.py`, `models/grounded_visual_pose_policy_v1.py` |

[Machine-readable index](index.json). Check it with
`python part2action/scripts/audit_active_configs.py`; add `--check-files` on the
research server to check installed demonstrations and initialization weights.

Configs select available model options, widths/layer counts, inputs, losses,
training and rollout settings. **Python implements the actual modules and
forward path; a checkpoint stores learned parameter values.** Reproducing a run
requires its resolved config, source version, initialization and trained weights.
A config alone is not a complete saved model or computation-graph specification.

The GT-study JSON generates `control.yaml` and `oracle_pose.yaml` in its run
directory. Those resolved snapshots are frozen inputs to the queued jobs; the
JSON is not itself a complete model YAML. Both arms retain vision and the same
original action decoder. The pose-conditioning module changes feature inputs.

The predicted-pose recipe retains some baseline construction flags for compatible
pretrained loading. Its versioned model explicitly omits unused heatmap/approach
and contact-token heads after construction. Read that model class alongside the
recipe; the list under `heads` alone does not describe its final active modules.
Frozen recipes are preserved instead of silently relabeling already run code.

Eighteen earlier recipes are now in the
[historical config archive](../docs/archive/2026-11-11_config_history/README.md),
with an additional baseline reference copy for historical regression tests.
One old thirty-epoch config path remains as a **compatibility symlink**, because
frozen rollout/trainer defaults still name it. It is not a fourth current recipe.
Historical checks and launchers point explicitly to the archive.
