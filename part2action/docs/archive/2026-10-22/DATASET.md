# PartInstruct dataset and evaluation splits

Updated October 12, 2026. This is the reference for dataset contents, split
meanings, counts, and the scope of our experiments. Server locations and storage
verification remain in [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md#dataset-storage);
download and evaluation commands remain in [SETUP.md](SETUP.md).

## What the dataset contains

PartInstruct supplies expert demonstrations for part-level robot manipulation
and test scenarios executed in the PartGym simulator. Instructions identify
parts of everyday objects, such as grasping a bottle at its neck or touching a
mug handle. A scenario may require several basic skills in sequence.

There are two different kinds of entries:

| Entry | Contents | How we use it |
| --- | --- | --- |
| Training demonstration | Recorded observation/action trajectory, task and per-step skill instructions, and geometric annotations | Supervised policy and auxiliary-head training |
| Test scenario | Object instance, initial pose and scale, task/skill instructions, skill-chain parameters, and episode identifier | Initialize the simulator; let the trained policy generate a rollout and score task completion |

Test metadata does not supply an expert action trajectory for every test
scenario. Expert videos in review GIFs can therefore come from task-matched
training demonstrations, with their different source episode and scene clearly
identified. A scenario is also not an object category: one category contains
many instances, instructions, part choices, and initial states.

## Release structure and annotation types

The installed release includes one training HDF5 file per category, shared JSON
metadata, and simulation assets:

```text
PartInstruct/
  demos/<category>.hdf5
  episodes_meta_train.json
  episodes_meta_test.json
  object_meta.json
  part_semantic_lexicon.json
  assets/
```

Each HDF5 contains `data/demo_<index>/` groups. The loader reads recorded RGB,
scene point clouds, robot/TCP and gripper state, skill instructions, and
seven-dimensional actions: world XYZ, wrist Euler angles, and finger opening.
Part masks and target-part point clouds provide auxiliary supervision. Exact
available keys vary by observation configuration; the maintained schema is in
[partinstruct_loader.py](../../../data/partinstruct_loader.py).

Contact targets in our project are derived from demonstration geometry and
skill/gripper timing, rather than measured physical finger-contact annotations.
Our policy uses predicted part/contact information at deployment; ground-truth
part masks or target-part clouds do not enter its action inputs. The official
DP3-S pilot used privileged part-cloud input and must be described separately.

## How the authors define the splits

The benchmark has five evaluation levels that test different changes relative
to training. They are not interchangeable random partitions. Definitions below
follow the [official PartInstruct evaluation protocol](https://partinstruct.github.io/).

| Split | Code | What is held out | Full-release test entries |
| --- | --- | --- | ---: |
| Test1 | OS | New initial object states, including positions/orientations, for familiar task/object settings | 488 |
| Test2 | OI | New object instances within training object categories | 710 |
| Test3 | TP | New part combinations within familiar task categories | 558 |
| Test4 | TC | New manipulation task categories | 253 |
| Test5 | OC | New object categories absent from the released training split | 962 |
| **Total** | | | **2,971** |

For example, changing the pose of a familiar mug is a state-generalization
question; using a new mug instance is an instance-generalization question.
Changing which parts participate in a familiar task tests part combinations.
A new task category can require composing known basic skills in a new way.
An unseen category asks the policy to transfer to a different kind of object.
These examples illustrate the distinctions; they are not exact instructions
for every scenario in the release.

The local training metadata contains task-type IDs 1–10. Test1, test2, test3,
and test5 use those task-type IDs; test4 uses the held-out task-type IDs 11–16.
A task-type ID is a category of manipulation, not one specific instruction or
one specific part. Do not assume that the same task-type ID means the same
part combination or object instance was demonstrated during training.

## Full release counts by category

Counts below come from the installed split metadata, rather than paper headline
numbers or the number of demos sampled by a training job. There are **10,026
training metadata entries across 11 categories**, and **2,971 test metadata
entries across 14 categories**. Eyeglasses, kettle, and laptop have no released
training demonstrations and appear in test5.

| Category | Train demos | Test1 | Test2 | Test3 | Test4 | Test5 | Total test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 1,243 | 46 | 72 | 49 | 13 | 0 | 180 |
| box | 872 | 50 | 66 | 58 | 19 | 0 | 193 |
| bucket | 981 | 49 | 65 | 63 | 29 | 0 | 206 |
| dispenser | 1,064 | 50 | 65 | 67 | 28 | 0 | 210 |
| display | 1,255 | 50 | 81 | 54 | 19 | 0 | 204 |
| eyeglasses | 0 | 0 | 0 | 0 | 0 | 444 | 444 |
| kettle | 0 | 0 | 0 | 0 | 0 | 258 | 258 |
| kitchenpot | 490 | 50 | 61 | 47 | 18 | 0 | 176 |
| knife | 387 | 40 | 56 | 40 | 10 | 0 | 146 |
| laptop | 0 | 0 | 0 | 0 | 0 | 260 | 260 |
| mug | 1,973 | 50 | 97 | 50 | 30 | 0 | 227 |
| pliers | 468 | 32 | 51 | 35 | 9 | 0 | 127 |
| scissors | 369 | 21 | 29 | 40 | 10 | 0 | 100 |
| stapler | 924 | 50 | 67 | 55 | 68 | 0 | 240 |
| **Total** | **10,026** | **488** | **710** | **558** | **253** | **962** | **2,971** |


A zero means the category has no entries in that split. The three test5-only
categories account for 962 entries; the 11 training categories account for 2,009
entries across test1–test4. These counts describe the released split structure,
not all combinations that could be generated in the simulator.

## Our four-object training scope

Current experiments train on bottle, mug, pliers, and scissors. They use all
4,053 valid demonstrations for these categories, with no demo-count cap. The
four HDF5 files also contain two groups without actions, excluded by the loader
and the completed timing audit. Downloading all 11 files does not automatically
add their categories to a training configuration.

| Category | Train demos | Test1 | Test2 | Test3 | Test4 | Total test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 1,243 | 46 | 72 | 49 | 13 | 180 |
| mug | 1,973 | 50 | 97 | 50 | 30 | 227 |
| pliers | 468 | 32 | 51 | 35 | 9 | 127 |
| scissors | 369 | 21 | 29 | 40 | 10 | 100 |
| **Total** | **4,053** | **149** | **249** | **174** | **62** | **634** |


The four categories have **634 test entries across test1–test4**. Their149 test1
entries are only one split, not their entire test suite. Test5 has no entries
for these four categories because its categories are held out from the full
training release. The remaining2,337 test entries belong to other categories.

## Current evaluation coverage

The queued timing comparison uses **all 149 four-object test1 scenarios** and
fixed evaluation seeds 42,43,44: **447 rollouts per model**. Each scenario is
selected explicitly once per seed, using matching scenario/seed pairs between
models. Report per-seed success and the mean with sample standard deviation.
These are evaluation seeds; each checkpoint currently comes from one training
seed, so they do not measure independent-training variability.

Test1 was selected as the initial controlled comparison of execution under
changed initial states. It is not a requirement to exclude the other splits,
and it does not establish instance, part-combination, or task-category
performance. No test2–test4 jobs are currently included in this comparison.
A complete four-object test1–test4 evaluation would cover634 entries per seed,
or 1,902 rollouts per model with three evaluation seeds. Report each split separately so
state-generalization results are not mixed with broader transfer results.

Expanding to the full release would mean 10,026 training metadata entries and
2,971 test entries across all five splits, with explicit category changes.
Evaluating all five splits at three seeds would produce 8,913 rollouts per model.
Those totals describe potential coverage, not jobs that have been submitted.

The earlier39-rollout protocol sampled three trials from 13 object–task groups,
restricted to task types1–4. Those groups contain 65 four-object test1 entries.
The full149 list covers 32 groups, including the other available task types.
Historical39-rollout scores differ in both coverage and task composition and
must be kept separate from the new results.

## Train/test separation and metadata identity

Training configs reference only training HDF5 demonstrations; test JSON metadata
initializes and scores rollouts. `val_hdf5` is empty at the user's request.
Offline fitting metrics therefore use training data, rather than a held-out
validation trajectory set. Test data is not used to fit the action policy.

A metadata check found **no overlap of category/episode-ID pairs between train
and test1**, including the four-object test1 scenarios currently queued.
However, the provided test2 metadata shares 3 category/episode-ID pairs with
training, and test3 shares 30. These are 33 pairs in total; eight involve our four
categories. This means we should not describe every released split as strictly
disjoint based on its split name. For all 33 reused IDs, the object ID, initial pose, and skill-chain
parameters also match the corresponding training metadata entry. Verify their
provenance before claiming complete held-out coverage on those splits. This check does not compare every stored HDF5 frame or establish
what happened during upstream dataset construction. No dataset entries were
removed or relabeled as part of this documentation change.

The2,971 total is a sum of split entries. Some episode IDs recur between test
splits: the local metadata has 2,738 distinct category/episode-ID pairs across
all tests. Keep the split name as part of a scenario's identity and report
split-specific results; do not call the 2,971 entries 2,971 necessarily different
physical scenes.

## Sources and reproducibility

Split meanings: [official project page](https://partinstruct.github.io/).
Counts and task-type membership: local
[training metadata](../../../../../../../../../p/part2action/data/PartInstruct/episodes_meta_train.json) and
[test metadata](../../../../../../../../../p/part2action/data/PartInstruct/episodes_meta_test.json).
The audit below was computed from these installed files on October12,2026.

| Metadata file | SHA-256 |
| --- | --- |
| `episodes_meta_train.json` | `3f12de437b88d0cf9df23500eb5fe533c8d658b04c312f03944ee2d4d12e4c55` |
| `episodes_meta_test.json` | `5e6902643430c35903c9156629b5ff7b7e173850c5db5abe73e678520dc4d436` |
