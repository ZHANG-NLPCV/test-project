#!/usr/bin/env python3
"""Create QC previews for OK DIC/MCY pairs in a manifest CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microscopy_pipeline.preview import create_sample_previews


DEFAULT_PREVIEW_DIR = Path("/home/hjk4090d/microscopy_project/qc_preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_csv", required=True, type=Path, help="Paired manifest CSV")
    parser.add_argument("--preview_dir", type=Path, default=DEFAULT_PREVIEW_DIR, help="Preview output folder")
    parser.add_argument("--frames", default="auto", help="'auto' or comma-separated frame indices")
    return parser.parse_args()


def _ok_rows(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "OK"]


def main() -> int:
    args = parse_args()
    rows = _ok_rows(args.index_csv)
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    output_count = 0
    for row in rows:
        sample_id = row["sample_id"]
        try:
            outputs = create_sample_previews(
                sample_id=sample_id,
                dic_path=row["dic_path"],
                mcy_path=row["mcy_path"],
                preview_root=args.preview_dir,
                frames=args.frames,
            )
        except Exception as exc:
            failures += 1
            print(f"ERROR {sample_id}: {exc}")
            continue
        output_count += len(outputs)
        print(f"OK {sample_id}: wrote {len(outputs)} files")

    print(f"Processed {len(rows)} OK rows; wrote {output_count} files; failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
