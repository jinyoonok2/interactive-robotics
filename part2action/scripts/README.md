# Scripts by model generation

Files stay flat because frozen source hashes and launchers record their paths.

| Group | Scripts | Launchers in `../slurm/` |
| --- | --- | --- |
| **V3 (latest)** | `train_/evaluate_/run_pose_token_diffusion_v3.py`, `run_pose_token_v3_phase_diagnostic.py`, `run_v3_grasp_probe.py` | `pose_token_diffusion_v3`, `evaluate_pose_token_v3_phase`, `evaluate_v3_grasp_probe`, `project_storage_env.sh` |
| V2 | `train_/evaluate_/run_pose_token_diffusion_v2.py`, `run_pose_token_v2_development.py` | `pose_token_diffusion_v2`, `evaluate_pose_token_v2_development` |
| V1 | `train_/evaluate_pose_token_diffusion.py`, `run_pose_token_evaluation.py` | `pose_token_diffusion`, `smoke_pose_token_diffusion`, `evaluate_pose_token_diffusion` |
| Visual policy + GT or predicted pose | `*visual_oracle*`, `train_visual_pose_v1.py`, `*grounded_visual_pose_v1.py` | `visual_oracle_pose_v1`, `evaluate_visual_oracle_pose_v2/v3`, `grounded_visual_pose_v1` |
| Original policy | `train.py`, `evaluate.py`, `validate_configs.py`, `smoke_unified_diffusion.py` | `train_diffusion_screening` |
| Rollout and review | `rollout_partgym.py`, `submit_evaluation_comparison.py`, `compare_rollout_metrics.py`, `export_evaluation_gifs.py`, `build_policy_comparison_videos.py`, `run_policy_diagnostic.py`, `summarize_contact_*.py` | `rollout_bigtemp_full_eval_a40`, `smoke_partgym_gpu`, `evaluate_policy_diagnostic`, `export_evaluation_gifs` |
| Released DP/DP3 | [`official_partinstruct/`](official_partinstruct/README.md) | `evaluate_official_*` |
| Shared helpers | `_common.py`, `register_result.py`, `download_full_partinstruct.py`, `audit_active_configs.py`, `generate_logged_references.py`, `run_interaction_tolerance_pilot.py`, `ground_truth_reference.py`, `visualize_contact_projection.py` | |
| Figures | `draw_*.py` | |

`generate_logged_references.py` and `run_interaction_tolerance_pilot.py` are
older entry points still imported by the v1 and GT-pose evaluators.
