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

## Fixed-scenario privileged reference

On November 16, the superseded 120/1 fixed evaluations and the earlier
self-trained DP3 pilot outputs were deleted at the user's request. The
completed `official_release_20261022` native evaluations above are retained.
The fixed-scenario launcher is available for a future fresh run using the
native 250-action/eight-action execution settings; no new run has been submitted.

`fixed_scenarios.py` reuses the same strict release loader and each native
runner's observation preprocessing, but pins the exact 149 scenarios and seeds
42/43/44 from our saved `evaluation_scenarios.json`. The manifest is checked
against dataset metadata before any GPU model loads. Full runs contain 447
trials per model. A four-case, seed-42 preflight covers one case per category.

The launcher `slurm/evaluate_official_fixed_scenarios.sbatch` now uses a 250-action
budget, executes eight actions per prediction, clips finger commands to [0,0.04] m,
and retains the released native camera, robot and task thresholds. Policy
weights, two-frame checkpoint observation preprocessing, native output chunks
and denoising settings are retained. The separate T5 preprocessing encoder is
explicitly put in evaluation mode. A wrapper time limit is never counted as
success; the final native `Success` field supplies the score.

**This is not a matched architecture ranking.** Released models still receive
GT part observations and oracle skill progression. Their `skill_mode=True`
sequential-skill success differs from the full-task predicate scorer used in our
standard runs. Observation modalities/training histories also differ. These
remaining differences are recorded in provenance and reports. Equal trial counts
must not be presented as equal information access or equal success definitions.
A future controlled architecture study needs a common task scorer and a defined
grounding/progression interface, with separate privileged and predicted-input
conditions. No such adapter is silently substituted here.

Required launch variables: `MODEL`, fresh `OUT`, submission `MANIFEST` containing
`source_sha256`, and `SCENARIOS_MANIFEST`. Optional `SCENARIO_INDICES` and
`EVAL_SEEDS` select explicit preflight subsets; full runs leave both empty.
Keep checkpoints and retained native-evaluation outputs intact. Per-trial actions,
initial states, CSV, JSON, protocol provenance and dated PNG tables are written
incrementally. Exceptions fail the job rather than becoming policy failures.
