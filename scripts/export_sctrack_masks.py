#!/usr/bin/env python3
"""Export SC-Track-ingestible instance masks from the strict binary baseline.

SC-Track (chan-labsite/SC-Track) consumes a grayscale multi-page TIFF per movie
where every frame holds segmented instances (each cell a distinct integer
label). It then produces ``track.csv``, which the SC-Track-evaluation scripts
score against the published GT (Zenodo record 10441055, ``tracking results.zip``
-- same samples as ours: MCF10A_copy02, MCF10A_copy11, copy_of_1_xy01,
copy_of_1_xy19, src06).

This script is the missing adapter between this repo's *per-patch binary*
prediction and that pipeline. For each sample it:

1. groups the strict CSV rows by ``frame_index``,
2. runs the trained FrameUNet on every 512x512 patch and ORs the binary
   foreground back onto a full-frame canvas at ``y0:y1, x0:x1`` (the
   stride-512 index tiles the frame, so this reassembles it),
3. instance-labels the *full frame* (connected components or watershed) so
   cells split across patch borders stay whole, with an optional min-area
   filter to drop speckle,
4. writes one ``<sample_id>.tif`` (T, H, W) uint16 stack per sample.

Downstream (run separately, see docs/sctrack_integration.md):

    SC-Track <sample>.tif  ->  track.csv  ->  evaluate-MOT.py (IDF1/MOTA)
                                          ->  prepare_TRA_compute_data.py (CTC SEG/TRA)

Cell-cycle phase metrics (CDF1 / phase classification) need per-cell phase
labels this binary model does not yet predict; that is a later step, not here.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_b0_frameunet_strict import (  # noqa: E402
    DEFAULT_CSV,
    PATCH_SIZE,
    crop_patch,
    load_csv_rows,
    parse_int,
    read_tiff_page,
    resolve_path,
    split_rows_by_csv,
    validate_row,
)
from eval_instance_strict import (  # noqa: E402
    DEFAULT_OUT_DIR,
    connected_components,
    load_model,
    predict_binary,
    watershed_instances,
)


def filter_min_area(labels: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return labels
    out = labels.copy()
    ids, counts = np.unique(labels, return_counts=True)
    for label_id, count in zip(ids, counts):
        if label_id != 0 and count < min_area:
            out[out == label_id] = 0
    return out


def label_frame(binary: np.ndarray, method: str, min_area: int) -> np.ndarray:
    instances = connected_components(binary) if method == "cc" else watershed_instances(binary)
    instances = filter_min_area(instances, min_area)
    if int(instances.max()) > np.iinfo(np.uint16).max:
        raise ValueError("More instances in one frame than uint16 can hold.")
    return instances.astype(np.uint16)


def group_rows_by_sample(
    rows: list[dict[str, str]], columns: Any
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[columns.sample_id]).strip()].append(row)
    return grouped


def export_sample(
    sample_id: str,
    sample_rows: list[dict[str, str]],
    columns: Any,
    csv_path: Path,
    model: Any,
    device: torch.device,
    args: argparse.Namespace,
    out_path: Path,
) -> dict[str, Any]:
    # Bucket patch rows per frame and capture the full-frame size.
    frames: dict[int, list[tuple[dict[str, str], dict[str, Any]]]] = defaultdict(list)
    frame_h = frame_w = 0
    for row in sample_rows:
        values = validate_row(row, columns)
        frames[int(values["frame_index"])].append((row, values))
        frame_h = parse_int(row[columns.frame_height], columns.frame_height, 0)
        frame_w = parse_int(row[columns.frame_width], columns.frame_width, 0)

    if frame_h <= 0 or frame_w <= 0:
        raise ValueError(f"{sample_id}: could not determine frame size")

    ordered_frames = sorted(frames)
    total_instances = 0
    with tifffile.TiffWriter(out_path, bigtiff=True) as writer:
        for position, frame_index in enumerate(ordered_frames):
            canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
            for row, values in frames[frame_index]:
                mcy_path = resolve_path(
                    row[columns.mcy_path], csv_path, int(values["row_number"]), columns.mcy_path
                )
                image = read_tiff_page(mcy_path, frame_index)
                patch = crop_patch(
                    image,
                    int(values["y0"]),
                    int(values["x0"]),
                    int(values["y1"]),
                    int(values["x1"]),
                    PATCH_SIZE,
                    PATCH_SIZE,
                    f"{sample_id} t={frame_index}",
                )
                binary = predict_binary(model, patch, device, args.threshold)
                region = canvas[
                    int(values["y0"]) : int(values["y1"]),
                    int(values["x0"]) : int(values["x1"]),
                ]
                np.maximum(region, binary, out=region)

            instances = label_frame(canvas, args.method, args.min_area)
            total_instances += int(instances.max())
            writer.write(
                instances,
                contiguous=True,
                metadata={"axes": "YX", "frame_index": int(frame_index)},
            )
            if args.progress_every and (position + 1) % args.progress_every == 0:
                print(f"  {sample_id}: {position + 1}/{len(ordered_frames)} frames", flush=True)

    return {
        "sample_id": sample_id,
        "frames": len(ordered_frames),
        "frame_height": frame_h,
        "frame_width": frame_w,
        "total_instances": total_instances,
        "output": str(out_path),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir).expanduser() / "sctrack_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, columns, csv_path = load_csv_rows(args.csv)
    split_rows = split_rows_by_csv(rows, columns, None, None, None)
    selected: list[dict[str, str]] = []
    for split in args.splits:
        selected.extend(split_rows.get(split, []))
    grouped = group_rows_by_sample(selected, columns)
    if args.samples:
        grouped = {s: r for s, r in grouped.items() if s in set(args.samples)}
    if not grouped:
        raise ValueError("No samples to export after split/sample filtering.")

    model = load_model(Path(args.checkpoint).expanduser(), device)
    print(f"[export] samples={list(grouped)} method={args.method} -> {out_dir}", flush=True)

    reports: list[dict[str, Any]] = []
    for sample_id, sample_rows in grouped.items():
        out_path = out_dir / f"{sample_id}.tif"
        report = export_sample(
            sample_id, sample_rows, columns, csv_path, model, device, args, out_path
        )
        reports.append(report)
        print(
            f"[done] {sample_id}: {report['frames']} frames, "
            f"{report['frame_height']}x{report['frame_width']} -> {out_path}",
            flush=True,
        )
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUT_DIR / "frameunet_strict_best.pt",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--method", choices=("cc", "watershed"), default="watershed")
    parser.add_argument("--min-area", type=int, default=20, help="Drop instances smaller than this many pixels.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--samples", nargs="+", default=None, help="Optional sample_id filter.")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    print("EXPORT_SCTRACK_MASKS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
