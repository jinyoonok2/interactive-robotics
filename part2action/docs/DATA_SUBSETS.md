# PartInstruct Data Subsets

PartInstruct is gated on Hugging Face. You do not need to download the full
dataset to train on a useful subset: each object has its own HDF5 demo file, and
`download_subset.sh` downloads only the objects you request.

## 1. Get Access

Accept the dataset terms in your browser:

```text
https://huggingface.co/datasets/SCAI-JHU/PartInstruct
```

Create a Hugging Face access token from your account settings. The token only
needs permission to read gated repositories.

## 2. Save The Token Locally

If `huggingface-cli login` hangs on this machine, save the token directly:

```bash
mkdir -p ~/.cache/huggingface
printf '%s' 'YOUR_HF_TOKEN_HERE' > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

Do not commit or paste this token into tracked files. If a token is exposed in a
terminal transcript or chat, revoke it on Hugging Face and create a new one.

## 3. Download Selected Objects

Run from `part2action/`:

```bash
OBJECTS="mug bottle scissors" bash download_subset.sh
```

By default, `download_subset.sh` writes to the clone-relative path
`../datasets/PartInstruct/`. Override this with `DATA_DIR=/path/to/PartInstruct`
if your data must live elsewhere.

This downloads the shared metadata:

```text
../datasets/PartInstruct/object_meta.json
../datasets/PartInstruct/part_semantic_lexicon.json
../datasets/PartInstruct/episodes_meta_train.json
../datasets/PartInstruct/episodes_meta_test.json
```

and the selected object demos:

```text
../datasets/PartInstruct/demos/mug.hdf5
../datasets/PartInstruct/demos/bottle.hdf5
../datasets/PartInstruct/demos/scissors.hdf5
```

The script uses `curl -4` internally to force IPv4 and avoid the IPv6 download
hang observed with some Hugging Face CLI calls. Existing non-empty files are
skipped by default. To redownload, set `FORCE=1`:

```bash
FORCE=1 OBJECTS="mug bottle scissors" bash download_subset.sh
```

Available HDF5 demo objects:

```text
bottle, box, bucket, dispenser, display, kitchenpot,
knife, mug, pliers, scissors, stapler
```

## 4. Generate Matching Training Configs

Generate configs for exactly the objects you downloaded:

```bash
python scripts/make_object_configs.py \
  --objects mug bottle scissors \
  --tag mug_bottle_scissors
```

This creates configs under:

```text
configs/generated/
```

For example:

```text
configs/generated/part_action_mlp_real_mug_bottle_scissors.yaml
configs/generated/heatmap_real_mug_bottle_scissors.yaml
```

The generated configs point `train_hdf5` at the chosen object files. Re-running
the generator with the same `--tag` overwrites those generated config files, but
it does not redownload data.

## 5. Train

```bash
conda activate part2action
python scripts/train.py \
  --config configs/generated/part_action_mlp_real_mug_bottle_scissors.yaml
```

The output directory is namespaced by the tag, for example:

```text
results/part_action_mlp_real/mug_bottle_scissors/
```
