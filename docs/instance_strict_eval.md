# Instance-level Evaluation of the Strict Binary Baseline

This evaluation answers the question the binary Dice hides: **once you turn the
predicted foreground into instances, how well are individual (often touching)
cells separated?** That is the metric family SC-Track / pcnaDeep live in.

It does **not** retrain anything. The script `scripts/eval_instance_strict.py`
reuses the trained binary checkpoint and the exact data path from
`scripts/train_b0_frameunet_strict.py`, then:

1. runs the model on each strict-split patch to get a binary foreground mask,
2. converts that mask into instances with two post-processors:
   - `cc` — 8-connectivity connected components (the floor: touching cells merge),
   - `watershed` — distance-transform watershed (what a cheap split buys),
3. scores them against the **instance** ground truth that training discards via
   `mask > 0` (the raw `mask_path` PNG, kept here with its instance labels).

## Metrics

Per `split` x `method`, plus per density bucket and per sample:

| Field | Meaning |
| --- | --- |
| `seg` | CTC-style SEG: mean IoU of GT instances matched at IoU > 0.5 |
| `ap50`, `ap75` | Cellpose-style AP = TP / (TP + FP + FN) at IoU 0.50 / 0.75 |
| `map5095` | mean AP over IoU 0.50:0.05:0.95 |
| `mean_matched_iou` | mean IoU of TP@0.5 matches |
| `count_ratio` | predicted instances / GT instances (`< 1` => merged cells) |
| `precision50`, `recall50`, `f1_50` | detection P/R/F1 at IoU 0.50 |

Matching is greedy by descending IoU (no scipy dependency). If a GT PNG turns
out to be binary rather than instance-labeled, the script falls back to
connected components for the GT and prints a one-time warning — check that
warning, because it changes how `seg`/`ap` should be read.

## Outputs (under `--out-dir`)

```text
instance_strict_split_metrics.csv
instance_strict_density_bucket_metrics.csv
instance_strict_sample_metrics.csv
instance_strict_summary.json
```

## Server commands

```bash
cd /home/hjk4090d/projects/test-project
git pull

source ~/miniconda3/etc/profile.d/conda.sh
conda activate micro

python -m py_compile scripts/eval_instance_strict.py
```

Tiny debug run (val + test, 64 patches each):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_instance_strict.py \
  --checkpoint /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline/frameunet_strict_best.pt \
  --max-val-patches 64 \
  --max-test-patches 64 \
  --out-dir /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline
```

Full instance evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_instance_strict.py \
  --checkpoint /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline/frameunet_strict_best.pt \
  --splits val test \
  --out-dir /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline
```

Successful evaluation ends with:

```text
INSTANCE_STRICT_EVAL_OK
```

## How to read the result

- If `cc` `count_ratio` is well below 1 in the `high` density bucket, the binary
  model is merging touching cells — the instance gap is real and large there.
- The delta between `cc` and `watershed` tells you how much a cheap split
  recovers without any model change. If watershed already closes most of the
  gap, instance separation may not need a new architecture; if it does not, that
  motivates the interior+boundary / distance-map output head (the next step).
- These `seg` / `ap50` numbers — not the binary Dice — are what line up with
  SC-Track / pcnaDeep reporting, so this is the table to take into the meeting.
