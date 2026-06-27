#!/usr/bin/env python3
"""Check B0 supervised patch and temporal-window dataloader shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_b0_frameunet_smoke import (
    FRAME_COLUMNS,
    FramePatchDataset,
    MASK_PATH_COLUMNS,
    MCY_PATH_COLUMNS,
    PATCH_SIZE,
    X_COLUMNS,
    Y_COLUMNS,
    crop_or_validate_patch,
    find_column,
    find_optional_column,
    image_patch_tensor,
    load_csv_rows,
    mask_patch_tensor,
    parse_int,
    read_mask_png,
    read_tiff_frame,
    row_path,
)


DEFAULT_PATCH_CSV = Path(
    "/home/hjk4090d/microscopy_project/annotations/dataset_index/"
    "supervised_patch_index_512_stride512.csv"
)
DEFAULT_TEMPORAL_CSV = Path(
    "/home/hjk4090d/microscopy_project/annotations/dataset_index/"
    "supervised_temporal_window_index_T5_strict.csv"
)
WINDOW_SIZE = 5


TEMPORAL_FRAME_COLUMN_GROUPS = (
    ("t0", "t1", "t2", "t3", "t4"),
    ("frame0", "frame1", "frame2", "frame3", "frame4"),
    ("frame_0", "frame_1", "frame_2", "frame_3", "frame_4"),
    ("frame_t0", "frame_t1", "frame_t2", "frame_t3", "frame_t4"),
    ("window_t0", "window_t1", "window_t2", "window_t3", "window_t4"),
    (
        "window_frame_0",
        "window_frame_1",
        "window_frame_2",
        "window_frame_3",
        "window_frame_4",
    ),
    (
        "frame_index_0",
        "frame_index_1",
        "frame_index_2",
        "frame_index_3",
        "frame_index_4",
    ),
)
START_FRAME_COLUMNS = (
    "start_t",
    "t_start",
    "window_start_t",
    "window_t_start",
    "start_frame",
    "frame_start",
    "window_start_frame",
)


class TemporalWindowDataset(Dataset):
    def __init__(
        self,
        csv_path: Path | str,
        nmax: int | None = None,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        rows, fieldnames, resolved_csv = load_csv_rows(csv_path)
        self.rows = rows[:nmax] if nmax is not None else rows
        self.csv_path = resolved_csv
        self.patch_size = patch_size
        self.mcy_col = find_column(fieldnames, MCY_PATH_COLUMNS, "MCY TIFF path")
        self.mask_col = find_column(fieldnames, MASK_PATH_COLUMNS, "instance mask PNG path")
        self.x_col = find_optional_column(fieldnames, X_COLUMNS)
        self.y_col = find_optional_column(fieldnames, Y_COLUMNS)
        if (self.x_col is None) != (self.y_col is None):
            raise ValueError(
                "CSV must provide both x and y patch coordinate columns, or neither. "
                f"Found x={self.x_col!r}, y={self.y_col!r}"
            )
        self.frame_cols = self._find_frame_column_group(fieldnames)
        self.start_col = find_optional_column(fieldnames, START_FRAME_COLUMNS)
        self.center_col = find_optional_column(fieldnames, FRAME_COLUMNS)
        if self.frame_cols is None and self.start_col is None and self.center_col is None:
            available = ", ".join(fieldnames)
            raise ValueError(
                "Temporal CSV must provide five frame columns, a start frame column, "
                f"or a center frame column. Available columns: [{available}]"
            )

    def _find_frame_column_group(self, fieldnames: list[str]) -> list[str] | None:
        for group in TEMPORAL_FRAME_COLUMN_GROUPS:
            columns = [find_optional_column(fieldnames, (name,)) for name in group]
            if all(column is not None for column in columns):
                return [column for column in columns if column is not None]
        return None

    def __len__(self) -> int:
        return len(self.rows)

    def frame_indices(self, row: dict[str, str], row_number: int) -> list[int]:
        if self.frame_cols is not None:
            return [parse_int(row[column], column, row_number) for column in self.frame_cols]
        if self.start_col is not None:
            start = parse_int(row[self.start_col], self.start_col, row_number)
            return list(range(start, start + WINDOW_SIZE))
        center = parse_int(row[self.center_col], self.center_col, row_number)
        half = WINDOW_SIZE // 2
        return list(range(center - half, center + half + 1))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        row_number = index + 2
        mcy_path = row_path(row, self.mcy_col, self.csv_path, row_number)
        mask_path = row_path(row, self.mask_col, self.csv_path, row_number)
        x0 = parse_int(row[self.x_col], self.x_col, row_number) if self.x_col else None
        y0 = parse_int(row[self.y_col], self.y_col, row_number) if self.y_col else None
        frame_indices = self.frame_indices(row, row_number)
        if len(frame_indices) != WINDOW_SIZE:
            raise ValueError(f"Row {row_number}: expected {WINDOW_SIZE} frame indices")

        images = []
        for frame_index in frame_indices:
            image = read_tiff_frame(mcy_path, frame_index)
            image = crop_or_validate_patch(
                image,
                x0,
                y0,
                self.patch_size,
                f"temporal image row {row_number} t={frame_index}",
            )
            images.append(image_patch_tensor(image).numpy())

        mask = read_mask_png(mask_path)
        mask = crop_or_validate_patch(mask, x0, y0, self.patch_size, f"mask row {row_number}")
        image_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(images, axis=0))).float()
        mask_tensor = mask_patch_tensor(mask)
        if tuple(image_tensor.shape) != (WINDOW_SIZE, 1, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Temporal image tensor has wrong shape {tuple(image_tensor.shape)}")
        return image_tensor, mask_tensor


def require_shape(tensor: torch.Tensor, expected: tuple[int | None, ...], label: str) -> None:
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise ValueError(f"{label} has wrong rank: expected {expected}, got {actual}")
    for actual_dim, expected_dim in zip(actual, expected):
        if expected_dim is not None and actual_dim != expected_dim:
            raise ValueError(f"{label} has wrong shape: expected {expected}, got {actual}")


def check_loader_shapes(args: argparse.Namespace) -> None:
    patch_ds = FramePatchDataset(
        args.patch_csv,
        rows=load_csv_rows(args.patch_csv)[0][: args.nmax],
    )
    temporal_ds = TemporalWindowDataset(args.temporal_csv, nmax=args.nmax)
    patch_loader = DataLoader(
        patch_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    temporal_loader = DataLoader(
        temporal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    patch_images, patch_masks = next(iter(patch_loader))
    temporal_images, temporal_masks = next(iter(temporal_loader))
    require_shape(patch_images, (None, 1, PATCH_SIZE, PATCH_SIZE), "Frame patch image batch")
    require_shape(patch_masks, (None, 1, PATCH_SIZE, PATCH_SIZE), "Frame patch mask batch")
    require_shape(
        temporal_images,
        (None, WINDOW_SIZE, 1, PATCH_SIZE, PATCH_SIZE),
        "Temporal window image batch",
    )
    require_shape(temporal_masks, (None, 1, PATCH_SIZE, PATCH_SIZE), "Temporal mask batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-csv", type=Path, default=DEFAULT_PATCH_CSV)
    parser.add_argument("--temporal-csv", type=Path, default=DEFAULT_TEMPORAL_CSV)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nmax", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_loader_shapes(args)
    print("B0_DATALOADER_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
