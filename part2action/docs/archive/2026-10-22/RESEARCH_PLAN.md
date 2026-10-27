# Part2Action Research Plan

Updated October 18, 2026. The matched decoder comparison completed.
Future experiments use the diffusion action head by default, per the user's
decision. Retain completed MLP models and results as historical references;
do not queue further MLP training unless explicitly requested.
The completed matched evaluation yielded diffusion control 33/447 successes
versus MLP control 3/447. This supports choosing diffusion for ongoing work,
without establishing that it will outperform MLP under every setting or seed.
Current screening protocol: 10 epochs from scratch with previous diffusion as
the baseline, two frames, next-index targets, stride 1, and contact disabled.
Compare this against the Transformer denoiser with all other settings matched.
This baseline choice follows the user's direction; the independent 30-epoch
timing evaluation and early-contact evaluation still provide separate evidence.
Paths and job status live in
[PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md); commands live in
[SETUP.md](SETUP.md). Slide history is a dated record and is not rewritten.

## Research question and next decision

Does predicted part-contact geometry help a policy perform part-specific
manipulation when it conditions actions throughout motion? Can a diffusion
decoder improve the approach, grasp, and interaction behavior of the current
MLP policy? Treat these as separate questions: a better decoder does not by
itself prove that explicit contact conditioning helps.

Finish the experiments already submitted. Compare success by object and task,
approach initiation, grasp closure, orientation, contact localization, and
inference latency. Use the matched expert-left GIFs to select the next targeted
fix. The earlier MLP continuation comparison remains limited evidence. The new
diffusion timing comparison starts both policies from the same untrained weights.

## Architecture and experiment definitions

All current variants use frozen DINOv2 and Flan-T5 encoders, deployable scene
geometry and robot state, learned visual part grounding, skill planning and
phase selection, and auxiliary contact supervision. The contact head predicts
a scene point. Dataset part masks and derived contacts are labels, not deployed
policy inputs. Finger opening is predicted with position and wrist orientation.

| Variant | Observation history | How explicit contact affects actions |
| --- | --- | --- |
| Original baseline | One frame | Auxiliary contact prediction; no explicit XYZ input to action head |
| Near-contact | Two frames | Inside3cm, XYZ becomes predicted contact plus a bounded offset |
| Unified MLP control | Two frames | One action policy throughout motion; contact contribution disabled |
| Original unified MLP contact | Two frames | Additive predicted TCP-relative geometry in action features |
| Normalized-global contact | Two frames | Bias-subtracted, normalized contact features with a learned global scalar |
| Diffusion control/contact pair | Two frames | Same optional normalized-global conditioner before a diffusion decoder |

Unified variants remove distance-based XYZ switching and the bounded contact
output offset. They retain the visual part gate and skill hierarchy. Current
configs enable visual part gating and disable skill conditioning of that gate.
The hierarchy's phase selector remains separate from contact strength.

The normalized contact feature expression is
`features + P(0) + weight * LayerNorm(P(r) - P(0))`, where
`r = (predicted_contact - TCP) / 0.1`. LayerNorm has no affine parameters.
The scalar is global and signed, initialized at0.01; it does not automatically
increase near the object. The explicit geometry path is detached from contact
coordinate gradients. Control disables the contact-dependent term and retains
`P(0)`. Ground-truth contact geometry never replaces predicted action-conditioning
geometry during training.

Both MLP and diffusion predict eight-action chunks and execute the first action
before observing again. Diffusion changes the modeled action distribution and
decoder; it is not the first introduction of trajectories or action chunks.

## Completed evidence and its limits

| Investigation | Finding | Interpretation |
| --- | --- | --- |
| Derived contact-label audit | Touch legacy events select motion onset; skill-specific events occur during the final approach in reviewed examples | Event heuristics need care; this does not establish broken demonstrations |
| Initial contact localization | Original model predictions were roughly1cm from the intended part surface | A nearby point is not a valid grasp pose, physical contact, or evidence the hand is close |
| Frozen additive-contact intervention | Reducing or removing contact input improved approach in some matched rollouts | Action sensitivity to contact matters beyond where the dot is located |
| Normalized MLP continuations | Control4/39; contact 0/39 | Normalization plus a learned global scalar has not demonstrated a contact benefit |
| Recording-order audit | All4,053 valid demos show first approach motion at the command's same saved index; eight CPU prefix replays confirm post-action TCP at that transition | Test alignment, but do not infer all native frame spacing or replay behavior from those prefixes |
| Timing MLP continuations | Same-index1/39; next-action5/39 | Encouraging continuation evidence; few successes, repeated scenarios, one training seed, and limited task coverage |

The old39-rollout results sampled three trials per object–task group with task
types1–4. They covered13 groups drawn from65 available test1 scenarios, not all
149. They are historical pilot scores and are not directly comparable to the
expanded protocol below.

The five next-action successes involved mug and pliers, three object–task
combinations, and four unique episodes. Two successful mug trials repeated the
same episode with different seeds. Do not interpret five successful rollouts
as five independent task types.

The released PartInstruct loader also uses same-index alignment. The next-action
shift is an experiment, rather than proof that the upstream loader is wrong.
Repeated absolute commands can remain useful after motion starts. Native touch
frame spacing/controller history remains unresolved, and missing pre-action
reset images cannot be recovered by shifting labels.

Private label details remain in [CONTACT_LABEL_AUDIT.md](CONTACT_LABEL_AUDIT.md).
Keep this personal analysis separate from public project descriptions.

## Active experiments

### Full-test1 timing comparison

Evaluate the same-index6705831 and next-action6705832 checkpoints on the identical
full test1 scenario list. Both are two-frame MLPs with contact disabled, stride 1,
and ten extra epochs from the same30-epoch control checkpoint with fresh
optimizers. Only their action-target offset differs: `t` versus `t+1`.
They are continuation models, rather than an isolated from-scratch alignment
comparison. That additional experiment is skipped.

Completed evaluation jobs: same-index6714294 and next-action6714295, plus
expert-left export6714296. Full-set scores are9/447 versus24/447. Outputs:
`results/part2action/archive/mlp/evaluations/action_timing_full_test1_20261012/`.
Single-seed replacement jobs6714288/6714289 and export 6714290 were cancelled
before starting, and their unused outputs were removed.

### Two-frame diffusion comparison

Training control 6712229 and contact 6712230 was submitted from scratch for30
epochs on the four-object train set, seed 42, stride 2, same-index targets,
two-frame history, and eight-action chunks. Both use the same architecture and
normalized-global conditioner; only contact-dependent conditioning is enabled
for contact. These runs are not using the next-action shift.

The control matches the original30-epoch two-frame MLP training protocol.
Existing MLP contact references used additive conditioning or continuation
training, so their comparison also differs in conditioning or training history.
Label these limitations instead of attributing every difference to the decoder.

The custom small DDPM decoder uses100 cosine steps and fixed action-unit
normalization; it passed a real-data H100 optimization/sampling preflight.
Compare rollout behavior and latency, not raw MLP action loss against diffusion
noise-prediction loss. See [DIFFUSION_ABLATION.md](DIFFUSION_ABLATION.md) for exact
scales, clipping, tests, and implementation limits. Diffusion evaluations6717932/6717933 now use the standard full-set protocol,
alongside the matched30-epoch two-frame MLP control6717931; export6717934 follows.

## Evaluation and reporting protocol

Use all 149 unique test1 scenarios for bottle, mug, pliers, and scissors, across
all 32 object–task groups. Select each scenario explicitly once for evaluation
seeds 42, 43, 44:447 rollouts per model. Match the exact scenario and seed between
policies, with max 120 actions, execute 1, legacy termination, bounded gripper
commands, CPU FPS, and GPU inference.

Report pooled success counts, each seed's149-scenario success rate, and the
mean with sample standard deviation across evaluation seeds. Also show per-object
and per-task counts. These seeds measure evaluation randomness. Current
checkpoints still come from one training seed each, so the report does not
measure variability from model initialization and training. Identical
near-deterministic behavior across seeds is possible and should be reported.

Keep one flat gallery containing a combined GIF per scenario/seed, expert
reference left and matched policies beside it. Retain raw videos, traces,
scenario lists, checkpoint/config references, and code/metadata fingerprints
under the same experiment. Task-matched training expert videos can have different
initial scenes and are illustrative references, not exact test ground truth.

Use the official train/test separation; no validation split is requested.
Training loss and offline training-data metrics are fitting measurements,
not held-out task success. Fixed seeds and full coverage make comparisons
reproducible, but repeatedly choosing models using the test score still makes
those choices part of test-guided development. Describe that use accurately.

## Bottlenecks to inspect after rollouts

| Question | Evidence to examine | Possible next experiment |
| --- | --- | --- |
| Does the hand initiate approach? | TCP displacement from reset, first approach action, commanded versus measured motion | Frozen-policy gripper/command intervention if waiting persists |
| Is it aiming at the intended part throughout motion? | Contact-to-part distance by timestep, predicted dot, moving object pose | Improve contact candidates or correspondence if localization drifts |
| Does contact conditioning disrupt actions? | Same-observation position/orientation changes and feature scales | Revisit conditioning only with a matched control |
| Does the gripper close at the right time and pose? | Raw/executed command, both measured finger joints, target distance, wrist orientation | Separate closure loss/representation experiment if supported by traces |
| Does diffusion help or just add cost? | Success, stalls, collisions, repeatability, per-cycle sampling latency | Adjust decoder only after locating its failure mode |
| Is the skill hierarchy selecting the right stage? | Planned slots, selected phase, termination logs | Targeted hierarchy ablation instead of assuming low-level actions are responsible |

A finger not reaching zero does not by itself prove incorrect closure: an object
between the fingers can prevent it. Surface proximity is not a substitute for
hand orientation, contact forces, or task success. Do not increase distance
gates or restore whole-path bounded offsets simply to hide approach failures.

## Four longer-term candidate methods

| Method | Proposed change | Why test it | Evidence to collect |
|---|---|---|---|
| 1. Contact candidates and confidence | Score scene points, retain several candidates, predict candidate-specific pose/width; supervise valid neighborhoods | Avoid averaging distinct contact modes; preserve alternative interactions | Surface distance, target localization, candidate recall, confidence calibration, rollout success |
| 2. Contact-and-pose approach | Predict a pre-contact pose and interaction pose; condition approach/local control on geometry and skill | Make contact guidance useful before the hand is already close | Pre-contact reach rate, collision rate, final pose error, grasp/touch success, switch stability |
| 3. Spatially grounded action features | Associate point tokens with calibrated image features; compare structured 3D action representations if needed | Preserve which image region belongs to each scene point | Part/contact localization, pose error, success under object-pose variation, compute cost |
| 4. Multimodal action generation | Compare MLP with a contact-conditioned diffusion action head using repeated observations | Represent several valid motion sequences instead of their average | Success, motion consistency, inference latency, recovery after perturbation |


Diffusion (method4) is implemented and training; candidate selection,
contact-and-pose targets, and finer point/image correspondence remain separate
future experiments. Develop them incrementally and compare matched controls.
A learned phase-dependent contact selector is another option if evidence supports
it; the current learned scalar is global, not that selector.

## Dataset scope and later expansion

[DATASET.md](DATASET.md) explains the complete release, the four-object scope,
and the benchmark's five generalization splits. Current evaluation is restricted
to test1; testing new instances, part combinations, and task categories requires
separate test2–test4 results. Do not call the current comparison the complete
four-object test suite.

Expanding training categories changes the training problem and should follow
the current decoder comparison, with explicit configuration and matching
evaluation coverage. Existing files and server paths remain documented in
[PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md#dataset-storage).

Keep the current frozen encoders while testing action decoding. A newer vision
or language backbone, a different dataset, or a VLA can be explored afterward;
combining those changes now would make the comparison difficult to interpret.

## Research precedents

These papers motivate representations; none validates our exact 3 cm switch
or guarantees gains on PartGym.

- [Contact-GraspNet](https://arxiv.org/abs/2103.14127): roots grasp proposals in observed surface contacts to reduce the search space.
- [Where2Act](https://openaccess.thecvf.com/content/ICCV2021/papers/Mo_Where2Act_From_Pixels_to_Actions_for_Articulated_3D_Objects_ICCV_2021_paper.pdf): predicts point actionability, interaction orientations, and proposal success.
- [VAT-Mart](https://hyperplane-lab.github.io/vat-mart/): predicts affordances and contact-conditioned trajectory proposals for articulated objects.
- [PerAct](https://peract.github.io/) and [RVT](https://proceedings.mlr.press/v229/goyal23a.html): structured spatial representations for action prediction.
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/): multimodal action-sequence generation and receding-horizon execution.

## Completed timing results and queued decoder evaluation — October 13, 2026

Expanded timing results: same-index9/447 (2.0%), next-action24/447 (5.4%).
Each seed scored3/149 versus8/149, with identical respective success episodes
and evaluation-seed SD0. The next-action advantage persists, but overall
success is low and these are single-training-seed continuation policies.
All447 expert-left GIFs are ready. Keep historical39 results separate.

Both diffusion checkpoints completed30 epochs. Queue MLP control6717931,
diffusion control6717932, and diffusion contact6717933, with export6717934.
They match training data, seed42, stride2, same-index targets, two frames,
and30 epochs; the timing continuations are not the matched-budget baseline.
Evaluate149 scenarios at seeds42,43,44 and compare success, approach, grasping,
and hardware-identified policy-forward latency. Diffusion noise loss is not
comparable to MLP action loss. No further training is needed before collecting
these results. Outputs: `results/part2action/evaluations/diffusion_30epoch_20261013/`.

## Next experiment — October 15, 2026

The completed30-epoch comparison favors diffusion control33/447 over MLP3/447;
diffusion contact trails at22/447. Test next-action targets before further
contact training, now using **two matched30-epoch runs from scratch**.
Same-index6719765 and next-action6719766 share freshly randomized policy
weights, frozen pretrained encoders, optimizer recipe, stride1, LR1e-4, two
frames, and training seed42. Only offset0/1 differs. Shared initialization is
marked epoch-1/zero steps and fingerprint-verified before training, rather
than loading any previously trained policy. Four initialization-protection
tests and seven action-alignment tests passed.

Evaluations6719767/6719768 follow their training on149 test1 scenarios at
seeds42,43,44; export6719769 follows both and shows expert reference left.
The superseded continuations6719759/6719760 and their dependent jobs were
cancelled; abandoned configs, outputs, links, and logs were removed.
Submission records:
`results/part2action/runs/diffusion_timing_scratch_submissions_20261015.json`.
Review: `results/part2action/evaluations/diffusion_timing_scratch_20261015/`.

An existing-result contact review separates outcome groups: control-only25,
contact-only14, both-success8, both-fail400. Contact's trained policy does
better on touch-starting cases (9 versus4 successes) and worse on grasp-starting
cases (13 versus29). These are exploratory subgroup findings, with repeated
evaluation seeds and one training seed per model. Do not interpret them as a
causal contact-input effect. Review the matched GIFs before deciding whether
to implement a frozen-diffusion contact-input intervention with matched sampling
noise. No contact training or new contact-intervention rollouts were submitted.
Review: `results/part2action/evaluations/diffusion_30epoch_20261013/contact_impact_review.html`.

### Early contact fusion implementation — October 16, 2026

Following the professor's suggestion, an optional early temporal input path is
implemented: latest-frame perception predicts contact; a contact/robot-state
token joins visual history before temporal attention and action-feature fusion.
It replaces late contact addition in this mode. Auxiliary contact supervision
remains, and action conditioning uses detached predictions during training and
deployment. The attention peak included in the token is not calibrated confidence.

Separate 30-epoch diffusion early-control and early-contact configs are ready,
with identical two-frame, stride-2, same-index settings and fresh training.
The control zeroes only the contact geometry and attention peak; both retain
the robot-state inputs. Submitted October 16: contact `6721407` remains pending
with flexible A40/A100/H100 allocation. Control `6721406` was cancelled before
starting at the user's request; the requested comparison uses existing models.
Configuration snapshots, source hashes, and job commands are saved in
`results/part2action/runs/diffusion_early_submission_20261016/submissions.json`.
The current timing experiment continues independently. The early variant also
uses latest-frame features for phase selection, so use its matched control to
isolate contact's contribution rather than attributing every difference from
the original late model to fusion placement alone.

Configuration details and submission commands: [DIFFUSION_ABLATION.md](DIFFUSION_ABLATION.md#early-contact-fusion--october-16-2026).

Early-contact evaluation submitted October 18: training `6721407` completed;
evaluation `6724354` runs all 149 test1 scenarios at seeds 42/43/44 (447 rollouts).
Dependent GIF export `6724355` combines expert reference, existing diffusion
control and late-contact recordings, and the new early-contact recordings.
Review: `results/part2action/evaluations/diffusion_early_contact_20261018/`.

### Transformer diffusion decoder — October 18, 2026

User selected a Transformer denoiser as the next non-contact architecture
experiment. Training `6724615` starts fresh for 30 epochs with two frames,
stride 2, same-index targets, existing pooled conditioning, and contact input
disabled. The part-mask gate and skill hierarchy remain enabled. Separate
modality-token cross-attention is not implemented in this first comparison.
The new denoiser uses two layers/four heads and increases decoder capacity;
its performance cannot isolate attention from parameter count alone.
Shared-module initialization is preserved across architectures under seed 42.

GPU preflight runs before full training. Dependent evaluation `6724616` uses
149 scenarios × three seeds, followed by comparison GIF export `6724617`.
The original diffusion-control recordings are reused for the comparison.
Review: `results/part2action/archive/superseded/diffusion_transformer_30epoch_20261018/review/`.

### Matched 10-epoch screening replaces queued 30-epoch Transformer — October 18

User requested two fresh 10-epoch trainings, previous diffusion baseline
`6724877` and Transformer diffusion `6724878`. Both use two frames, next-index
targets, stride 1, seed 42, same data/losses/optimizer settings, and no contact
conditioning. They load no trained policy or optimizer state. The visual part
mask gate remains enabled; the separate 3D part head remains disabled.
Config validation and the matched-pair check passed. Real-data GPU preflight
runs inside each training job before a fresh full-training process starts.

Evaluations `6724879`/`6724880` and GIF export `6724881` follow automatically.
The unused 30-epoch same-index Transformer pipeline `6724615`/`6724616`/`6724617`
was cancelled before starting. Its archived entry above is historical.
Review: `results/part2action/evaluations/diffusion_screening_10epoch_next_action_20261018/`.

### Completed diffusion results — October 19, 2026

Early fusion: 40/447 successes (8.9%), compared with original control 33/447
(7.4%) and late contact 22/447 (4.9%), under the 30-epoch, stride-2, same-index
protocol. Early fusion improves the aggregate score in this experiment, but
also changes upstream phase/contact features. The early control was cancelled
at the user's request; this is not a clean isolated contact-input effect.
Only one training seed per model was evaluated, at three rollout seeds.
All 447 comparison GIFs are ready in
`results/part2action/evaluations/diffusion_early_contact_20261018/`.

Matched scratch timing (30 epochs, stride 1, contact off): same-index 10/447,
next-index 16/447. Both training and evaluations completed; GIF export was
still running at this check. These scores use a different stride from the
original 33/447 control reference. The two new 10-epoch next-index decoder
trainings are still running; their automatic evaluations are already queued.

## Transformer stride comparison — October 21, 2026

The completed 10-epoch stride-1 comparison scored baseline 46/447 (10.3%)
and Transformer 64/447 (14.3%). Transformer won in all three evaluation seeds;
A40 steady policy-forward means were 80.5 ms and 147.4 ms respectively.
One training seed per model; these measurements exclude simulator and media work.

User requested a fresh 10-epoch Transformer with training stride 2. Submitted
training `6731746`, dependent full-test1 evaluation `6731747`, and expert-left
stride comparison GIF export `6731748`. Two consecutive observation frames,
next-index targets, contact conditioning disabled, seed 42, batch 16, learning
rate 1e-4, and source code match the completed stride-1 Transformer. Only stride,
experiment name, and output directory differ in configuration. No resume or
trained initialization is used. Equal epochs imply fewer optimizer updates for
stride 2; this tests the practical screening setting, not an equal-update budget.
Evaluation reuses all 149 test1 scenarios and seeds 42/43/44 (447 rollouts).
Completed stride-1 evaluation is reused rather than rerun.

Config: `configs/architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_transformer_stride2.yaml`.
Submission ledger: `results/part2action/runs/diffusion_transformer_stride2_submission_20261021/submissions.json`.
Review: `results/part2action/evaluations/diffusion_transformer_stride_comparison_20261021/index.html`.

## Released PartInstruct reference — October 22, 2026

For the reproduction objective, use the downloaded author-released DP-S and
DP3-S checkpoints rather than continuing our historical self-trained versions.
Both EMA policies passed strict CPU state loading. GPU evaluations `6734186`
(DP-S) and `6734187` (DP3-S) perform one native scissors preflight, followed by
176 native-sampled test1 trials per model across 11 categories. These are
privileged-input references: GT target-part observations and oracle skill
progression, without SAM/VLM calls. They are not the 447-rollout Part2Action
comparison and do not yet reproduce every paper protocol setting. Source
revision, compatibility adjustments, commands, and videos are recorded under
`results/partinstruct/evaluations/official_release_20261022/`.

## Quantitative mechanism evaluation — October 22, 2026

User requested data-driven comparisons instead of choosing directions primarily
from video inspection. Added reusable `scripts/compare_rollout_metrics.py`, which
reuses existing contact-motion CSV parsing and slide PNG rendering. It verifies
matched scenario/seed keys and protocol fields, emits long-form per-trial CSV,
paired outcome CSV, grouped JSON, dated PNG tables, and an HTML report. Success
uncertainty uses scenario-cluster bootstrap (5,000 resamples, seed 42), retaining
the three evaluation seeds together rather than treating 447 trials as independent
scenes. This does not estimate training-seed uncertainty or mechanism causality.

Completed baseline/Transformer report:
`results/part2action/evaluations/diffusion_screening_10epoch_next_action_20261018/quantitative_report/index.html`.
Overall 46/447 versus 64/447; paired outcomes both success 21, baseline only 25,
Transformer only 43, both fail 358. Success delta +4.03 percentage points,
scenario-cluster 95% bootstrap interval [-0.22, +8.72] points. One training seed.
Tasks starting with grasp: baseline 40/339, Transformer 27/339. Tasks starting
with touch: 6/108 versus 37/108. These are whole-task outcomes grouped by FIRST
skill, not isolated grasp or touch success rates. Gains are not uniform across
skills: investigate grasp-starting failures separately. Both policies have
explicit contact input disabled, so these results do not establish a benefit
from feeding predicted contact into the policy.

Existing quantitative coverage: task success/completion, sampled first-part
surface distance for predicted contact/TCP/action targets, phase transitions,
gripper command saturation, hardware-labelled policy latency. Geometry averages
are over visited rollout states with different lengths; proximity is not contact
or stable grasp. Diagnostic GT geometry is never passed to the policy.

For the next controlled contact evaluations, predefine and log physical metrics
in the simulator: designated-part finger contacts, grasp acquisition criteria,
contact-loss events and durations, relative gripper/object pose drift, and
per-skill success/transition timing. Define thresholds and denominators before
comparing policies, distinguishing attempted-stage rates from conditional rates
after successful acquisition. Track part/body identity and sensor-step timestamps;
keep unsupported and missing measurements explicit, not zero. These require
additional instrumentation and rollouts; current videos/distances cannot recover
all of them reliably. No such physics metrics or new A/B/C training were claimed
or submitted as part of this offline report. Contact perturbations must hold
initial scene and diffusion randomness fixed; report conditioning shifts and
compare retrained controls as well. Test held-out instances and multiple training
seeds before a generalization/novelty claim.

## Explicit contact input in Transformer — October 22, 2026

User requested enabling contact conditioning after clarifying that all recent
10-epoch decoder/stride runs retained contact auxiliary learning but disabled
explicit predicted-contact coordinates in the action path. That choice originally
isolated decoder and stride changes, not the final contact-enabled architecture.

The stride comparison completed: stride 1 Transformer 64/447, stride 2 48/447
(16/149 at each seed for stride 2). Therefore the new contact-on experiment uses
stride 1 and the completed 64/447 Transformer 6724878 as its contact-off control.
It starts fresh for 10 total epochs, seed 42, two consecutive frames, next-index
labels, batch 16, LR 1e-4, and the same model/training source. Configuration differs
only in `contact_conditioning_enabled: true`, name, and output directory. Contact
prediction and its auxiliary loss remain enabled in both. This is comparison C
versus B; no A model without contact auxiliary loss was submitted.

Existing late normalized-global conditioning is used: detached predicted world
contact minus measured TCP position, scaled by 0.1 m, projected into action
features with LayerNorm and a learned global weight initialized at .01. Skill
conditioning remains unchanged. Early temporal fusion, contact-relative action
outputs, and GT contact inputs are not enabled. Action gradients do not train the
contact point predictor through this detached branch; its auxiliary supervision
remains. This tests explicit input rather than changing fusion placement.

Submitted training `6734379`, dependent evaluation `6734380`, and comparison
GIF plus quantitative report `6734381`. GPU job performs its existing real-data
AMP optimization and sampling preflight before fresh training. Evaluation uses
149 fixed test1 scenarios x seeds 42/43/44 =447 trials, 120 actions maximum,
execute one action, legacy termination, videos and contact diagnostics. Completed
contact-off rollouts are reused. CPU export supports an opt-in quantitative report
in the comparison request, producing grouped CSV/JSON and dated PNG tables as
well as expert → contact off → contact on GIFs. Existing requests without that
flag retain their behavior. Physical contact/retention metrics are not newly
instrumented by this experiment.

Config: `configs/architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_transformer_contact.yaml`.
Submission: `results/part2action/runs/diffusion_transformer_contact_submission_20261022/submissions.json`.
Review: `results/part2action/evaluations/diffusion_transformer_contact_comparison_20261022/index.html`.
Validation: matched-config comparison, unchanged model/training source hashes,
file/config validation, two normalized-contact tests (enabled signal/gradient,
detached contact, control bias preservation, checkpoint round trip), and shell
syntax check. One training seed per condition; this is not a novelty proof.

User stopped the contact-ON experiment before it began. Jobs 6734379, 6734380,
and 6734381 were all CANCELLED at elapsed 00:00:00; no contact-ON model or
rollout results were produced. Preserve the config for review, but do not
resubmit without a new request. The completed stride comparison remains
64/447 for stride 1 versus 48/447 for stride 2, both contact OFF.

User decision, October 22: fix training sample stride at 1 for future Part2Action
experiments unless explicitly changed by the user. This is the working default
from the matched 10-epoch Transformer comparison (64/447 vs 48/447), not a claim
that stride 1 is universally optimal. Contact-ON remains cancelled; do not
resubmit it merely because stride 1 was selected. At the status check, only
official DP-S/DP3-S evaluations 6734186/6734187 were queued for Part2Action;
there was no active Part2Action training job.

## Contact experiment resubmitted — October 22, 2026

User explicitly requested restarting the cancelled contact-ON experiment.
New jobs: training 6734402, dependent 447-rollout evaluation 6734403, and
expert/off/on GIF plus quantitative report 6734404. Same configuration and source
hashes as the cancelled attempt were verified before resubmitting. Fresh retry1
output directories preserve the prior cancellation ledger. No resumed training.
Stride 1, 10 epochs, two consecutive frames, next-index, Transformer denoiser,
late normalized-global predicted-contact conditioning ON, contact auxiliary loss
ON. Control is completed contact-OFF Transformer 6724878 (64/447).
Review: `results/part2action/evaluations/diffusion_transformer_contact_comparison_20261022_retry1/`.
Submission: `results/part2action/runs/diffusion_transformer_contact_submission_20261022_retry1/submissions.json`.

After completion: inspect the paired B/C quantitative report and task-group
breakdowns before changing architectures. If contact helps, test whether frozen
policies respond meaningfully to controlled contact perturbations, with initial
scenes and diffusion randomness held fixed, and replicate training seeds. If
benefit is absent or negative, separate contact localization error, conditioning
usage/weight, and label quality with reference-contact and masking diagnostics;
do not infer that contact information is inherently useless. Instrument actual
part contacts, grasp acquisition/loss and pose retention before claiming physical
mechanisms. Native official DP/DP3 results are implementation/reference checks;
comparisons against Part2Action must align scenarios and rollout conditions and
separately account for GT part observations/oracle skill progression. These future
experiments are proposed only; no further jobs were submitted by this step.
