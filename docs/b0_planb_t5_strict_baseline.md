# B0 PlanB T5 Strict Baseline

This experiment trains a formal PlanB T=5 temporal baseline for binary foreground segmentation. The model uses five MCY frames as temporal context and predicts the center-frame foreground mask.

The script is `scripts/train_b0_planb_t5_strict.py`. It reads each MCY TIFF frame lazily, crops all five 512 x 512 patches using `y0:y1, x0:x1`, reads the center instance mask PNG, converts `mask > 0` to binary foreground, and trains a small U-Net style model named `PlanBT5UNet`.

The PlanB-lite architecture reshapes input from `[B, 5, 1, 512, 512]` to `[B, 5, 512, 512]`, then runs a 2D U-Net with five input channels and one output logit channel.

## Expected CSV

Default CSV:

```text
/home/hjk4090d/microscopy_project/annotations/dataset_index/supervised_temporal_window_index_T5_strict.csv
```

Required columns:

```text
sample_id, split, center_t, t_minus2, t_minus1, t0, t_plus1, t_plus2,
mcy_path, center_mask_path, y0, x0, y1, x1, patch_h, patch_w,
frame_height, frame_width, center_foreground_ratio_patch,
center_has_foreground
```

## Strict Split

No random train/validation split is performed. The CSV `split` column is the only source of assignment:

```text
train: split == "train"
val:   split == "val"
test:  split == "test"
```

The optional `--max-train-windows`, `--max-val-windows`, and `--max-test-windows` flags only cap each already-defined split in CSV order for debugging.

## Outputs

All outputs are written under `--out-dir`, outside the Git repository by default:

```text
planb_t5_strict_best.pt
planb_t5_strict_last.pt
planb_t5_strict_epoch_log.csv
planb_t5_strict_summary.json
planb_t5_strict_split_metrics.csv
planb_t5_strict_sample_metrics.csv
planb_t5_strict_density_bucket_metrics.csv
planb_t5_strict_config.json
prediction_overlays/
```

Metrics include loss, binary Dice, binary IoU, precision, recall, foreground pixel ratio, and number of windows for train, validation, and test. Additional CSVs report per-sample metrics and foreground-density bucket metrics.

## Server Commands

```bash
cd /home/hjk4090d/projects/test-project
git pull

source ~/miniconda3/etc/profile.d/conda.sh
conda activate micro

mkdir -p "$HOME/tmp" "$HOME/pip_cache"
export TMPDIR=$HOME/tmp TEMP=$HOME/tmp TMP=$HOME/tmp PIP_CACHE_DIR=$HOME/pip_cache

python -m py_compile scripts/train_b0_planb_t5_strict.py
```

Tiny strict debug run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_b0_planb_t5_strict.py \
  --epochs 1 \
  --batch-size 2 \
  --max-train-windows 256 \
  --max-val-windows 64 \
  --max-test-windows 64 \
  --num-workers 0 \
  --out-dir /home/hjk4090d/microscopy_project/runs/b0_planb_t5_strict_debug
```

Full strict baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_b0_planb_t5_strict.py \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 2 \
  --out-dir /home/hjk4090d/microscopy_project/runs/b0_planb_t5_strict_baseline
```

Successful training ends with:

```text
PLANB_T5_STRICT_BASELINE_OK
```
