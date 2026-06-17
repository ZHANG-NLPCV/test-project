"""TIFF discovery and lazy page I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


EXCLUDED_DIR_NAMES = {
    "__macosx",
    "qc_preview",
    "processed",
    "outputs",
    "manifests",
    "broken_zip",
    "zip_rescue",
    "salvaged_from_broken_zip",
}

TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class TiffInfo:
    """Metadata collected without loading a complete TIFF stack."""

    path: Path
    size_mb: float
    pages: int
    series_shape: tuple[int, ...]
    axes: str
    dtype: str
    first_page_shape: tuple[int, ...] | None = None
    first_page_dtype: str | None = None
    first_page_min: float | None = None
    first_page_max: float | None = None


def is_excluded_path(path: Path) -> bool:
    """Return True when any path component is a generated or temporary folder."""

    for part in path.parts:
        lowered = part.lower()
        if lowered in EXCLUDED_DIR_NAMES or lowered.startswith("inspect"):
            return True
    return False


def _prune_dirs(dirnames: list[str]) -> None:
    dirnames[:] = [
        dirname
        for dirname in dirnames
        if dirname.lower() not in EXCLUDED_DIR_NAMES
        and not dirname.lower().startswith("inspect")
    ]


def discover_tiffs(data_root: Path | str, max_depth: int | None = 4) -> list[Path]:
    """Recursively discover TIFF files while pruning known output folders."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {root}")

    tiffs: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        _prune_dirs(dirnames)
        if is_excluded_path(current.relative_to(root)):
            dirnames[:] = []
            continue

        depth = len(current.relative_to(root).parts)
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []

        for filename in filenames:
            path = current / filename
            if path.suffix.lower() in TIFF_SUFFIXES:
                tiffs.append(path)

    return sorted(tiffs)


def inspect_tiff(path: Path | str, read_first_page: bool = True) -> TiffInfo:
    """Inspect one TIFF without materializing the full stack."""

    tiff_path = Path(path).expanduser().resolve()
    size_mb = tiff_path.stat().st_size / (1024 * 1024)

    with tifffile.TiffFile(tiff_path) as tif:
        series = tif.series[0]
        pages = len(tif.pages)
        info = {
            "path": tiff_path,
            "size_mb": size_mb,
            "pages": pages,
            "series_shape": tuple(int(dim) for dim in series.shape),
            "axes": str(series.axes),
            "dtype": str(series.dtype),
        }
        if read_first_page and pages > 0:
            first_page = tif.pages[0].asarray()
            info.update(
                {
                    "first_page_shape": tuple(int(dim) for dim in first_page.shape),
                    "first_page_dtype": str(first_page.dtype),
                    "first_page_min": float(np.min(first_page)),
                    "first_page_max": float(np.max(first_page)),
                }
            )

    return TiffInfo(**info)


def read_tiff_page(path: Path | str, frame_index: int) -> np.ndarray:
    """Read a single TIFF page by index using tifffile's lazy page API."""

    if frame_index < 0:
        raise IndexError(f"Frame index must be non-negative, got {frame_index}")

    tiff_path = Path(path).expanduser().resolve()
    with tifffile.TiffFile(tiff_path) as tif:
        page_count = len(tif.pages)
        if frame_index >= page_count:
            raise IndexError(
                f"Frame index {frame_index} out of range for {tiff_path} "
                f"with {page_count} pages"
            )
        return tif.pages[frame_index].asarray()


def shape_to_string(shape: Iterable[int] | None) -> str:
    """Format shapes consistently for CSV output."""

    if shape is None:
        return ""
    return str(tuple(int(dim) for dim in shape))
