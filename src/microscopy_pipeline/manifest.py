"""Pair DIC/MCY TIFF stacks and build dataset manifest rows."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from microscopy_pipeline.io import (
    EXCLUDED_DIR_NAMES,
    TIFF_SUFFIXES,
    TiffInfo,
    inspect_tiff,
    is_excluded_path,
    shape_to_string,
)


MANIFEST_FIELDS = [
    "sample_id",
    "folder",
    "dic_path",
    "mcy_path",
    "dic_shape",
    "mcy_shape",
    "dic_axes",
    "mcy_axes",
    "dic_dtype",
    "mcy_dtype",
    "dic_pages",
    "mcy_pages",
    "dic_size_mb",
    "mcy_size_mb",
    "status",
]


def _is_tiff(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TIFF_SUFFIXES


def _contains_keyword(path: Path, keyword: str) -> bool:
    return keyword in path.name.lower()


def _empty_row(folder: Path, dic_files: list[Path], mcy_files: list[Path], status: str) -> dict[str, str]:
    return {
        "sample_id": folder.name,
        "folder": str(folder),
        "dic_path": str(dic_files[0]) if len(dic_files) == 1 else "",
        "mcy_path": str(mcy_files[0]) if len(mcy_files) == 1 else "",
        "dic_shape": "",
        "mcy_shape": "",
        "dic_axes": "",
        "mcy_axes": "",
        "dic_dtype": "",
        "mcy_dtype": "",
        "dic_pages": "",
        "mcy_pages": "",
        "dic_size_mb": "",
        "mcy_size_mb": "",
        "status": status,
    }


def _skip_status(dic_files: list[Path], mcy_files: list[Path]) -> str | None:
    reasons: list[str] = []
    if not dic_files:
        reasons.append("MISSING_DIC")
    if not mcy_files:
        reasons.append("MISSING_MCY")
    if len(dic_files) > 1:
        reasons.append("MULTIPLE_DIC")
    if len(mcy_files) > 1:
        reasons.append("MULTIPLE_MCY")
    return f"SKIP_{'_'.join(reasons)}" if reasons else None


def _row_from_infos(folder: Path, dic_info: TiffInfo, mcy_info: TiffInfo, status: str) -> dict[str, str]:
    return {
        "sample_id": folder.name,
        "folder": str(folder),
        "dic_path": str(dic_info.path),
        "mcy_path": str(mcy_info.path),
        "dic_shape": shape_to_string(dic_info.series_shape),
        "mcy_shape": shape_to_string(mcy_info.series_shape),
        "dic_axes": dic_info.axes,
        "mcy_axes": mcy_info.axes,
        "dic_dtype": dic_info.dtype,
        "mcy_dtype": mcy_info.dtype,
        "dic_pages": str(dic_info.pages),
        "mcy_pages": str(mcy_info.pages),
        "dic_size_mb": f"{dic_info.size_mb:.3f}",
        "mcy_size_mb": f"{mcy_info.size_mb:.3f}",
        "status": status,
    }


def build_manifest_row(folder: Path | str) -> dict[str, str] | None:
    """Build one manifest row for a folder containing candidate TIFF files."""

    sample_folder = Path(folder).expanduser().resolve()
    if is_excluded_path(sample_folder):
        return None

    tiffs = sorted(path for path in sample_folder.iterdir() if _is_tiff(path))
    dic_files = [path for path in tiffs if _contains_keyword(path, "dic")]
    mcy_files = [path for path in tiffs if _contains_keyword(path, "mcy")]
    if not dic_files and not mcy_files:
        return None

    skip_status = _skip_status(dic_files, mcy_files)
    if skip_status is not None:
        return _empty_row(sample_folder, dic_files, mcy_files, skip_status)

    try:
        dic_info = inspect_tiff(dic_files[0], read_first_page=False)
        mcy_info = inspect_tiff(mcy_files[0], read_first_page=False)
    except Exception as exc:
        return _empty_row(sample_folder, dic_files, mcy_files, f"SKIP_INSPECT_ERROR: {exc}")

    if dic_info.series_shape != mcy_info.series_shape:
        status = "MISMATCH_SHAPE"
    elif dic_info.dtype != mcy_info.dtype:
        status = "MISMATCH_DTYPE"
    else:
        status = "OK"
    return _row_from_infos(sample_folder, dic_info, mcy_info, status)


def build_manifest_rows(data_root: Path | str) -> list[dict[str, str]]:
    """Recursively build manifest rows for folders with candidate DIC/MCY TIFFs."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {root}")

    rows: list[dict[str, str]] = []
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname.lower() not in EXCLUDED_DIR_NAMES
            and not dirname.lower().startswith("inspect")
        ]
        if is_excluded_path(current.relative_to(root)):
            dirnames[:] = []
            continue
        if any(Path(filename).suffix.lower() in TIFF_SUFFIXES for filename in filenames):
            row = build_manifest_row(current)
            if row is not None:
                rows.append(row)
    return sorted(rows, key=lambda row: row["folder"])


def write_manifest_csv(rows: list[dict[str, str]], out_csv: Path | str) -> None:
    """Write manifest rows to CSV, creating the parent directory as needed."""

    output_path = Path(out_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def manifest_summary(rows: list[dict[str, str]]) -> str:
    """Create a compact terminal summary for manifest generation."""

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    lines = ["Manifest summary", "status,count"]
    for status, count in sorted(counts.items()):
        lines.append(f"{status},{count}")
    lines.append("")
    lines.append("sample_id,status,dic_shape,mcy_shape")
    for row in rows:
        lines.append(
            f"{row['sample_id']},{row['status']},{row.get('dic_shape', '')},{row.get('mcy_shape', '')}"
        )
    return "\n".join(lines)
