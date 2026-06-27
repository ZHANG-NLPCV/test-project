# B0 FrameUNet Strict Baseline

This experiment trains a supervised FrameUNet T=1 baseline for binary foreground segmentation on the B0 microscopy patch index.

The script is `scripts/train_b0_frameunet_strict.py`. It reads MCY TIFF frames lazily, crops each 512 x 512 patch using `y0:y1, x0:x1`, reads the instance mask PNG, converts `mask > 0` to binary foreground, and trains a small U-Net with `BCEWithLogitsLoss + Dice loss`.

## Expected CSV

Default CSV:

```text
/home/hjk4090d/microscopy_project/annotations/dataset_index/supervised_patch_index_512_stride512.csv
```

Required columns include:

```text
sample_id, split, mcy_path, mask_path, y0, x0, y1, x1, patch_h, patch_w,
frame_height, frame_width, foreground_ratio_patch, has_foreground
```

The script also requires a frame index column named one of:

```text
t, frame_index, frame_idx, frame, time_index, time_idx, timepoint
```

## Strict Split

No random train/validation split is performed. The CSV `split` column is the only source of assignment:

```text
train: split == "train"
val:   split == "val"
test:  split == "test"
```

The optional `--max-train-patches`, `--max-val-patches`, and `--max-test-patches` flags only cap each already-defined split in CSV order for debugging.

## Outputs

All outputs are written under `--out-dir`, outside the Git repository by default:

```text
frameunet_strict_best.pt
frameunet_strict_last.pt
frameunet_strict_epoch_log.csv
frameunet_strict_summary.json
frameunet_strict_split_metrics.csv
frameunet_strict_sample_metrics.csv
frameunet_strict_density_bucket_metrics.csv
frameunet_strict_config.json
prediction_overlays/
```

Metrics include loss, binary Dice, binary IoU, precision, recall, foreground pixel ratio, and number of patches for train, validation, and test. Additional CSVs report per-sample metrics and foreground-density bucket metrics.

## Server Commands

```bash
cd /home/hjk4090d/projects/test-project
git pull

source ~/miniconda3/etc/profile.d/conda.sh
conda activate micro

mkdir -p "$HOME/tmp" "$HOME/pip_cache"
export TMPDIR=$HOME/tmp TEMP=$HOME/tmp TMP=$HOME/tmp PIP_CACHE_DIR=$HOME/pip_cache

python -m py_compile scripts/train_b0_frameunet_strict.py
```

Tiny strict debug run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_b0_frameunet_strict.py \
  --epochs 1 \
  --batch-size 4 \
  --max-train-patches 512 \
  --max-val-patches 128 \
  --max-test-patches 128 \
  --num-workers 0 \
  --out-dir /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_debug
```

Full strict baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_b0_frameunet_strict.py \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 2 \
  --out-dir /home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline
```

Successful training ends with:

```text
FRAMEUNET_STRICT_BASELINE_OK
```
