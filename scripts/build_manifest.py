#!/usr/bin/env python3
"""Build a paired DIC/MCY dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_pipeline.manifest import build_manifest_rows, manifest_summary, write_manifest_csv


DEFAULT_OUT_CSV = Path("/home/hjk4090d/microscopy_project/manifests/paired_dataset_index.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="Root folder containing sample folders")
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV, help="Output manifest CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_manifest_rows(args.data_root)
    write_manifest_csv(rows, args.out_csv)
    print(f"Wrote {len(rows)} sample rows to {args.out_csv}")
    print(manifest_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
