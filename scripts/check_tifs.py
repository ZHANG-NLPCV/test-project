#!/usr/bin/env python3
"""Inspect TIFF files without loading full stacks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_pipeline.io import discover_tiffs, inspect_tiff, shape_to_string


DEFAULT_OUT_CSV = Path("/home/hjk4090d/microscopy_project/manifests/tif_manifest.csv")

FIELDS = [
    "path",
    "size_mb",
    "pages",
    "series_shape",
    "series_axes",
    "series_dtype",
    "first_page_shape",
    "first_page_dtype",
    "first_page_min",
    "first_page_max",
    "status",
    "error",
]


def _row_for_tiff(path: Path) -> dict[str, str]:
    try:
        info = inspect_tiff(path, read_first_page=True)
    except Exception as exc:
        return {
            "path": str(path),
            "status": "ERROR",
            "error": str(exc),
        }

    return {
        "path": str(info.path),
        "size_mb": f"{info.size_mb:.3f}",
        "pages": str(info.pages),
        "series_shape": shape_to_string(info.series_shape),
        "series_axes": info.axes,
        "series_dtype": info.dtype,
        "first_page_shape": shape_to_string(info.first_page_shape),
        "first_page_dtype": info.first_page_dtype or "",
        "first_page_min": "" if info.first_page_min is None else str(info.first_page_min),
        "first_page_max": "" if info.first_page_max is None else str(info.first_page_max),
        "status": "OK",
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="Root folder containing raw TIFF data")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum recursion depth")
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [_row_for_tiff(path) for path in discover_tiffs(args.data_root, args.max_depth)]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    ok_count = sum(row.get("status") == "OK" for row in rows)
    error_count = len(rows) - ok_count
    print(f"Wrote {len(rows)} TIFF rows to {args.out_csv}")
    print(f"OK: {ok_count}; ERROR: {error_count}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
