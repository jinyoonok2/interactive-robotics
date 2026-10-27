# Released PartInstruct checkpoints

`evaluate.py` runs the official DP-S/DP3-S checkpoint using the public native
PartInstruct evaluator, runner, observation preprocessing, EMA policy, eight-action
execution blocks, and 250-step test1 budget. It does not retrain or load optimizer
state. Evaluation uses ground-truth target-part observations and oracle skill
progression, not SAM/VLM grounding. A separate scissors rollout precedes the full
native pass: 16 sampled rollouts per category across all 11 test1 categories
(176 per model). Native sampling can repeat scenarios and is not the fixed
149-scenario x three-seed Part2Action evaluation. This first pass is not an exact
reproduction of every setting or repeated run in the paper.

Download source: https://huggingface.co/datasets/SCAI-JHU/PartInstruct

The two released checkpoint files live under
`/p/part2action/data/official_checkpoints/PartInstruct/checkpoints/`.
`download_manifest.json` records the immutable source revision and file sizes.

The release contains legacy `lgm_bc` Hydra class names and a stale DP input-channel
setting. `compat.py` maps those class names to their public equivalents and adds
the mask-name alias required by the public encoder. DP's convolution channel count
is read from the weights (five: RGB plus the public encoder's repeated mask).
No learned weight is reshaped, dropped, or randomly substituted. Strict EMA state
loading must pass. Inference dimensions come from the checkpoint configuration.
All adjustments are separate from the framework and are recorded in provenance.

Submit `part2action/slurm/evaluate_official_partinstruct.sbatch` with `MODEL=DP-S`
or `MODEL=DP3-S` and an absolute `OUT` directory. The job stops if its preflight
fails. It writes `smoke/` and `full/`, with videos, native rollout logs, resolved
configuration, provenance, and success summaries. Existing output directories
are never overwritten.
