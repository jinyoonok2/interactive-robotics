# Part2Action Project Overview for a Profile Webpage

Part2Action is an ongoing robotics research project that studies how a robot can
translate instructions about object parts into manipulation actions. Given an
instruction such as “grasp the bottle at its neck,” the policy combines visual
observations, scene geometry, language, and robot state to predict how the robot
should move. The central research question is whether an explicitly predicted
contact point can improve action generation when it guides the policy throughout
the motion.

The project uses PartInstruct demonstrations for learning and the PartGym
simulator for evaluation with a Franka Panda robot. The current work includes
part grounding, contact prediction, skill selection, temporal action prediction,
and controlled comparisons of how contact geometry influences actions. It is an
implemented simulation research project; improvement from the latest contact
conditioning design remains to be established through completed evaluations.

## Motivation and research question

Object identity alone does not specify how to interact with an object. Grasping
a bottle at its neck and touching its body require different target regions and
different motions, even though both instructions refer to the same object.
Part2Action addresses this relationship between language, the relevant object
part, and the robot action needed to carry out the instruction.

A key distinction is between locating a useful interaction point and producing
a successful motion. A predicted point provides spatial guidance, but the robot
must also approach it with an appropriate wrist orientation, gripper opening,
and sequence of actions. The project therefore investigates how geometry should
enter the policy, rather than assuming that a contact location alone determines
the command.

The current controlled experiment asks whether supplying predicted contact
geometry to the action policy improves manipulation compared with an otherwise
matched policy that does not receive this explicit geometry input.

## What the robot observes and predicts

The deployed policy receives a camera image, the full scene point cloud, the task
instruction, the robot tool center point pose, and measured gripper opening. The
tool center point, or TCP, is the reference pose used to describe the robot hand.
The point cloud represents visible scene geometry in three dimensions.

Dataset part masks, target-part point clouds, and skill labels are used as
training supervision. They are not supplied as deployment inputs to Part2Action.
This makes predicting the relevant part and interaction geometry part of the
policy's task. Current training recipes also exclude joint angles from the policy
inputs.

The action head predicts eight future actions. Each contains seven values:
world-frame X, Y, and Z position; wrist roll, pitch, and yaw; and a gripper
command. This is an action sequence, not a separate trajectory plotting head.
The current evaluation executes the first predicted action, obtains a new
observation, and predicts again. Repeated observation and action prediction form
the closed-loop control process.

## How the architecture works

Frozen DINOv2 ViT-S/14 with register tokens encodes the camera image, and frozen
Flan-T5-base encodes the task language. Trainable modules combine these features
with scene geometry and robot state. The project uses these pretrained models
as components of its policy architecture.

A learned skill hierarchy predicts skill slots, selects the current skill, and
predicts termination. A skill-conditioned visual part gate emphasizes relevant
image features, with dataset part masks supervising that prediction. The visual
part gate remains active in the unified policy variants; removing the contact
distance switch does not remove visual part grounding or the skill hierarchy.

A contact attention head scores scene points and predicts one world-frame
contact location. This point is a learned interaction target. It is distinct
from a part mask and does not certify physical contact between the fingers and
the object. The policy still needs to predict a suitable hand pose and motion.

An MLP action head produces the action sequence. The near-contact and unified
variants also use a two-frame temporal transformer to incorporate recent visual
history. The original baseline uses one frame.

## How contact information influences actions

The project has four implemented variants:

| Variant | Use of predicted contact geometry |
| --- | --- |
| Original baseline | Learns contact prediction as an auxiliary task, without feeding the predicted coordinates directly into the action head or replacing its XYZ command |
| Near-contact policy | Uses the action head while farther away; inside a 3 cm distance threshold, replaces XYZ with predicted contact plus a bounded offset of at most 3 cm |
| Unified control | Uses one action head throughout the motion, with zeros supplied to the explicit contact-conditioning projection |
| Unified contact | Uses one action head throughout the motion, with predicted contact relative to the TCP supplied to action features |

In the near-contact design, contact geometry determines the position command
only after a distance condition activates. Wrist and gripper commands remain
with the action head. The training switch uses demonstration contact labels,
while deployment uses predicted contact geometry.

In the unified contact design, geometry is available from the start of the
motion. The model computes the vector from the current TCP position to the
predicted contact point, normalizes it by a 0.1 m scale, and projects it into
the action features. The action head learns how to use this information to
generate positions, wrist rotations, and gripper commands. There is no separate
distance-triggered XYZ replacement or bounded contact-offset branch.

The unified design uses predicted contact geometry during both training and
deployment. Ground-truth geometry continues to supervise the auxiliary contact
head. In this first experiment, the explicit geometry vector is detached from
action-loss gradients, allowing the comparison to test its usefulness without
adding that new gradient path into the contact predictor. Contact confidence is
not an implemented conditioning input.

## Why the matched comparison matters

The unified control and unified contact policies use the same two-frame temporal
encoder, hierarchy, action decoder, auxiliary supervision, training data, seed,
and training budget. Both contain the same conditioning projection. The control
supplies zeros to it, while the contact variant supplies predicted relative
geometry.

This comparison isolates the effect of explicit contact conditioning more
closely than comparing the new policy with the original baseline. The original
baseline also differs in observation history, and the near-contact policy
changes the way position commands are produced. Their results provide useful
references but cannot isolate contact conditioning alone.

Both unified variants still learn part and contact predictions. The experiment
tests whether explicitly using the predicted contact coordinates helps action
generation beyond the shared features and auxiliary supervision already present
in the control.

## Dataset and simulator

The full released PartInstruct dataset is available locally and contains 11
object categories: bottle, box, bucket, dispenser, display, kitchenpot, knife,
mug, pliers, scissors, and stapler. The raw demonstrations, assets, and metadata
occupy approximately 69.09 GiB.

The current controlled training experiment uses four categories: bottle, mug,
pliers, and scissors. Training and test episodes are separate. The current
recipes use train and test data without a held-out validation split. Training
losses therefore describe fitting to demonstrations, rather than held-out task
success.

Closed-loop evaluation uses PartGym, the simulator used by the PartInstruct
framework. The matched Part2Action protocol contains 39 test trials per model:
13 supported object/task combinations with three trials each. Each trial permits
up to 120 steps, uses the same termination setting, and executes one predicted
action per observation cycle. This initial protocol measures behavior within
the selected simulation tasks; it does not establish real-robot performance or
broad generalization to unseen environments.

## Implemented work and evaluation progress

As recorded on October 8, 2026, all four Part2Action variants completed 30 epochs
of training. Each checkpoint also passed a short H100 simulator smoke test that
reset the environment and executed five actions. Matched 39-trial evaluations
were launched afterward. These implementation and execution checks establish
that the policies can run; they do not establish a success-rate improvement.
The maintained environment guide records subsequent evaluation progress.

An additional experiment trained the official PartInstruct DP3-S baseline on
scissors demonstrations. A five-trial pilot using ground-truth target-part cloud
input succeeded once and reached the step limit in four trials. This small
pilot uses a different input setting and evaluation protocol from Part2Action,
so it is not a matched performance ranking or a reproduction of the paper's
full evaluation.

The completed engineering work includes configurable model variants, data
loading and coordinate handling, auxiliary supervision, training launchers,
simulator rollouts, and organized experiment outputs. GPU training and evaluation
run through Slurm on the University of Virginia CS servers, with launchers that
support A40, A100, and H100 allocations. Heavy datasets, environments, and
experiment artifacts are consolidated in dedicated project storage.

## Next research steps

The immediate priority is to compare the unified control and unified contact
policies under the matched rollout protocol, examining success by object and
skill as well as motion behavior and contact localization. Repeated training
seeds would strengthen any promising result.

If explicit geometry conditioning helps, the next experiment is a stronger
action decoder based on diffusion or flow matching. The existing MLP already
predicts an action chunk, so this comparison should preserve the action horizon
and other settings to distinguish decoder improvements from changes in sequence
length or observations. This decoder experiment is planned, not yet established
as an improvement.

Other future directions include expanding training from four categories to the
full release, retaining multiple contact candidates, predicting interaction
poses, and improving correspondence between visual features and scene points.
These directions address different questions and should be tested separately.

## Profile webpage wording

### Short description

Part2Action is my ongoing robotics research project on manipulation guided by
instructions about object parts. I investigate how visual part grounding and
predicted 3D contact geometry can guide robot action generation, using
PartInstruct demonstrations and closed-loop evaluation in PartGym.

### Longer description

I am developing Part2Action to study how robots can turn instructions such as
“grasp the bottle at its neck” into manipulation actions. The policy combines
pretrained visual and language representations with scene point clouds, robot
state, learned part grounding, and contact prediction. It learns from
PartInstruct demonstrations without requiring ground-truth part masks or
target-part clouds as deployment inputs.

My current experiment compares two otherwise matched policies: one receives
predicted contact geometry throughout the motion, and the other does not receive
that explicit geometry input. This tests whether connecting contact prediction
directly to action generation improves manipulation. I have implemented and
trained the policy variants and built a Slurm-based training and simulator
evaluation workflow on UVA CS servers. Planned extensions include broader
dataset coverage and diffusion or flow-matching action decoders.

## Repository references

- [Repository README](README.md) provides project navigation.
- [Part2Action README](part2action/README.md) describes model interfaces and variants.
- [Project environment guide](part2action/docs/PROJECT_ENVIRONMENT.md) records current experiment status and execution details.
