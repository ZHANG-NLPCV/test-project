#!/usr/bin/env python3
"""Orchestrate the SC-Track benchmark: instance masks -> track.csv -> IDF1/MOTA.

This does NOT reimplement SC-Track or invent its tracking. It chains the
external Chan-lab tools and the public GT into one command and emits a single
summary table:

    <sample>.tif --(SC-Track)--> track.csv --(motmetrics)--> IDF1 / MOTA  vs  GT

The MOT computation mirrors `chan-labsite/SC-Track-evaluation/evaluate-MOT.py`
exactly (per-frame centroid matching via `motmetrics`, same metric list), so
results are comparable to their published numbers without depending on that
script's hard-coded local paths.

Two alignment hazards this guards against (see docs/sctrack_integration.md):
  1. Column naming. GT and tracker tables both need
     `track_id, cell_id, frame_index, center_x, center_y`; we alias common
     variants and fail loudly if a required field is truly missing.
  2. Frame indexing. We optionally rebase each table's `frame_index` to start
     at 0 so GT and prediction align even if one is 1-based (`--rebase-frames`).

Run modes:
  * Full:  --tif-dir + --sctrack-cmd + --gt-dir   (run SC-Track then evaluate)
  * Eval-only: --track-dir + --gt-dir             (already have track.csv)

Because the exact GT schema could not be verified offline, run one sample first
and check the printed column-resolution report before trusting the table.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Canonical -> accepted aliases (normalized: lowercased, separators stripped).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "track_id": ("track_id", "trackid", "track", "lineage_id", "lineageid"),
    "cell_id": ("cell_id", "cellid", "id", "label", "instance_id", "instanceid"),
    "frame_index": ("frame_index", "frameindex", "frame", "t", "time", "timepoint", "slice"),
    "center_x": ("center_x", "centerx", "cx", "x", "centroid_x", "centroidx", "pos_x"),
    "center_y": ("center_y", "centery", "cy", "y", "centroid_y", "centroidy", "pos_y"),
}
REQUIRED = ("cell_id", "frame_index", "center_x", "center_y")  # track_id optional for MOT
METRIC_NAMES = [
    "num_frames", "idf1", "idp", "idr", "recall", "precision", "num_objects",
    "mostly_tracked", "partially_tracked", "mostly_lost", "num_false_positives",
    "num_misses", "num_switches", "num_fragmentations", "mota",
]


def _norm(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def resolve_and_rename(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Map a table's columns onto the canonical schema; raise on missing required."""
    norm_map = {_norm(c): c for c in df.columns}
    rename: dict[str, str] = {}
    resolved: dict[str, str | None] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = next((norm_map[_norm(a)] for a in aliases if _norm(a) in norm_map), None)
        resolved[canonical] = found
        if found is not None:
            rename[found] = canonical
    print(f"  [{label}] column resolution: " +
          ", ".join(f"{k}<-{v}" for k, v in resolved.items()))
    missing = [c for c in REQUIRED if resolved[c] is None]
    if missing:
        raise ValueError(
            f"[{label}] missing required column(s) {missing}. "
            f"Available: {list(df.columns)}. Add an alias in COLUMN_ALIASES or "
            f"pre-rename the CSV."
        )
    out = df.rename(columns=rename)
    if resolved["track_id"] is None:
        out["track_id"] = out["cell_id"]
    return out


def load_track_table(path: Path, label: str, rebase_frames: bool) -> pd.DataFrame:
    df = resolve_and_rename(pd.read_csv(path), label)
    df = df.copy()
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce")
    df = df.dropna(subset=["frame_index"])
    df["frame_index"] = df["frame_index"].astype(int)
    if rebase_frames and len(df):
        df["frame_index"] = df["frame_index"] - int(df["frame_index"].min())
    return df.sort_values(by=["cell_id", "frame_index"]).reset_index(drop=True)


def evaluate_mot(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> "pd.Series":
    """Per-frame centroid matching MOT metrics — mirrors evaluate-MOT.py.

    Iterates over GT frames (so frames the tracker missed entirely still count
    as misses), matching detections to GT by squared Euclidean centroid
    distance, exactly as the reference script does.
    """
    import numpy as np
    import motmetrics as mm

    acc = mm.MOTAccumulator()
    gt_by_frame = dict(tuple(gt_df.groupby("frame_index")))
    pred_by_frame = dict(tuple(pred_df.groupby("frame_index")))
    for i, frame in enumerate(sorted(gt_by_frame)):
        gt_group = gt_by_frame[frame]
        oids = list(gt_group["cell_id"].values)
        dt_group = pred_by_frame.get(frame)
        if dt_group is None or len(dt_group) == 0:
            hids: list[Any] = []
            dists = np.zeros((len(oids), 0))
        else:
            hids = list(dt_group["cell_id"].values)
            dists = mm.distances.norm2squared_matrix(
                gt_group[["center_x", "center_y"]].values,
                dt_group[["center_x", "center_y"]].values,
            )
        try:
            acc.update(oids, hids, dists, frameid=i)
        except (KeyError, ValueError):
            continue
    metrics = mm.metrics.create()
    summary = metrics.compute(acc, metrics=METRIC_NAMES)
    return summary.iloc[0]


def run_sctrack(tif_path: Path, cmd_template: str, out_track: Path) -> None:
    """Invoke an external SC-Track command. {tif} and {out} are substituted."""
    cmd = cmd_template.format(tif=str(tif_path), out=str(out_track))
    print(f"  [sctrack] {cmd}", flush=True)
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"SC-Track command failed ({result.returncode}) for {tif_path}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not out_track.is_file():
        raise FileNotFoundError(
            f"SC-Track did not produce expected track.csv at {out_track}. "
            f"Check --sctrack-cmd output template. stdout:\n{result.stdout}"
        )


def discover_samples(args: argparse.Namespace) -> list[str]:
    if args.samples:
        return list(args.samples)
    source = Path(args.track_dir) if args.track_dir else Path(args.tif_dir)
    pattern = "*.csv" if args.track_dir else "*.tif"
    names = sorted({p.stem for p in source.glob(pattern)})
    if not names:
        raise ValueError(f"No samples found under {source} ({pattern}).")
    return names


def find_gt(gt_dir: Path, sample: str) -> Path:
    """Locate the GT CSV for a sample, tolerating the GT naming suffixes."""
    candidates = [gt_dir / f"{sample}.csv"] + sorted(gt_dir.glob(f"{sample}*.csv"))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No GT csv for sample '{sample}' under {gt_dir}")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    track_dir = Path(args.track_dir).expanduser() if args.track_dir else out_dir / "track_csv"
    track_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir).expanduser()

    samples = discover_samples(args)
    print(f"[eval] samples={samples}", flush=True)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        print(f"\n=== {sample} ===", flush=True)
        track_csv = track_dir / f"{sample}.csv"

        if args.tif_dir and args.sctrack_cmd:
            tif_path = Path(args.tif_dir).expanduser() / f"{sample}.tif"
            if not tif_path.is_file():
                print(f"  [skip] no tif: {tif_path}")
                continue
            run_sctrack(tif_path, args.sctrack_cmd, track_csv)
        if not track_csv.is_file():
            print(f"  [skip] no track.csv for {sample} (expected {track_csv})")
            continue

        try:
            gt_path = find_gt(gt_dir, sample)
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue

        pred_df = load_track_table(track_csv, "pred", args.rebase_frames)
        gt_df = load_track_table(gt_path, "gt", args.rebase_frames)
        summary = evaluate_mot(gt_df, pred_df)
        row = {"sample_id": sample, **{m: summary.get(m) for m in METRIC_NAMES}}
        rows.append(row)
        print(f"  IDF1={row.get('idf1'):.4f}  MOTA={row.get('mota'):.4f}", flush=True)

    if not rows:
        raise RuntimeError("No samples evaluated. Check --tif-dir/--track-dir/--gt-dir.")

    table = pd.DataFrame(rows, columns=["sample_id", *METRIC_NAMES])
    # Object-weighted mean for headline metrics; plain mean is misleading across
    # movies of different length/cell count.
    total_obj = table["num_objects"].sum()
    mean_row: dict[str, Any] = {"sample_id": "MEAN(obj-weighted)"}
    for m in METRIC_NAMES:
        if m in ("idf1", "idp", "idr", "recall", "precision", "mota") and total_obj > 0:
            mean_row[m] = float((table[m] * table["num_objects"]).sum() / total_obj)
        else:
            mean_row[m] = float(table[m].sum())
    table = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)

    out_csv = out_dir / "sctrack_mot_summary.csv"
    table.to_csv(out_csv, index=False)
    print(f"\n[written] {out_csv}")
    print(table[["sample_id", "idf1", "mota", "num_objects", "num_switches"]].to_string(index=False))
    return out_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tif-dir", type=Path, default=None,
                   help="Dir of <sample>.tif from export_sctrack_masks.py (to run SC-Track).")
    p.add_argument("--sctrack-cmd", type=str, default=None,
                   help="SC-Track command template, e.g. 'sctrack -p {tif} -o {out}'. "
                        "{tif} and {out} are substituted.")
    p.add_argument("--track-dir", type=Path, default=None,
                   help="Dir of existing <sample>.csv track tables (skip running SC-Track).")
    p.add_argument("--gt-dir", type=Path, required=True,
                   help="Dir of GT <sample>.csv (Zenodo 10441055 tracking results).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--samples", nargs="+", default=None)
    p.add_argument("--rebase-frames", action="store_true",
                   help="Rebase frame_index to start at 0 for both GT and pred.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.track_dir and not (args.tif_dir and args.sctrack_cmd):
        print("ERROR: provide either --track-dir, or both --tif-dir and --sctrack-cmd.",
              file=sys.stderr)
        return 2
    run(args)
    print("RUN_SCTRACK_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
