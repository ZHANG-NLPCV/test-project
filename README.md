# Live-cell microscopy TIFF preprocessing

Small Python utilities for inspecting raw live-cell microscopy TIFF stacks, pairing DIC/MCY channels, and generating lightweight QC previews without loading complete stacks into memory.

The raw server data is private and intentionally excluded from git. Keep TIFF, ZIP, array, model, manifest, and preview outputs outside the repository or in ignored folders.

## Server assumptions

- Host OS: CentOS 7
- User: `hjk4090d`
- Conda environment: `micro`
- Python: 3.10
- Raw root: `/home/hjk4090d/microscopy_project/raw`
- Output root: `/home/hjk4090d/microscopy_project`

Expected raw sample layout:

| Sample folder | Files | Expected stack shape | Expected dtype |
| --- | --- | --- | --- |
| `MCF10A_copy02` | `dic-sub1.tif`, `mcy-sub1.tif` | `(101, 1024, 1024)` | `uint16` |
| `MCF10A_copy11` | `dic-sub1.tif`, `mcy-sub1.tif` | `(101, 1024, 1024)` | `uint16` |
| `copy_of_1_xy01` | `dic.tif`, `mcy.tif` | `(369, 2048, 2048)` | `uint16` |
| `copy_of_xy_19` | `dic.tif`, `mcy.tif` | `(369, 2048, 2048)` | `uint16` |

The first axis is treated as time regardless of TIFF axis metadata. The paired abstraction is `T x C x H x W` with `C=2` channels ordered as `[DIC, MCY]`. The code reads individual pages with `tif.pages[t].asarray()` and avoids full-stack loads.

## Setup

Run these commands from the repository root on the server:

```bash
conda create -n micro python=3.10 -y
conda activate micro
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch needs a CUDA-specific wheel, install the appropriate `torch` build for the server before running GPU-dependent work.

## Usage

Check the Python environment and optional GPU visibility:

```bash
python scripts/check_env.py
```

Inspect all TIFF files under the raw root without loading complete stacks:

```bash
python scripts/check_tifs.py \
  --data_root /home/hjk4090d/microscopy_project/raw \
  --max_depth 4 \
  --out_csv /home/hjk4090d/microscopy_project/manifests/tif_manifest.csv
```

Build the paired DIC/MCY manifest:

```bash
python scripts/build_manifest.py \
  --data_root /home/hjk4090d/microscopy_project/raw \
  --out_csv /home/hjk4090d/microscopy_project/manifests/paired_dataset_index.csv
```

Generate QC previews for rows with `status == OK`:

```bash
python scripts/make_qc_preview.py \
  --index_csv /home/hjk4090d/microscopy_project/manifests/paired_dataset_index.csv \
  --preview_dir /home/hjk4090d/microscopy_project/qc_preview \
  --frames auto
```

Inspect a ZIP/archive without extracting it:

```bash
python scripts/inspect_zip.py --zip_path /path/to/archive.zip
```

The ZIP inspector uses `7z` or `7za` for integrity checks when available and otherwise falls back to Python's ZIP reader.

## Development checks

Run unit tests with synthetic TIFF stacks:

```bash
python -m pytest tests
```

These tests create tiny temporary TIFF files only under pytest-managed temp directories.

## Repository layout

```text
configs/server.yaml
scripts/check_env.py
scripts/check_tifs.py
scripts/build_manifest.py
scripts/make_qc_preview.py
scripts/inspect_zip.py
src/microscopy_pipeline/
tests/
```

Generated manifests and previews should live under `/home/hjk4090d/microscopy_project`, not in git-tracked source folders.
