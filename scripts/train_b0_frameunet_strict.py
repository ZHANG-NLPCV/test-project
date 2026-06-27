#!/usr/bin/env python3
"""Formal strict-split FrameUNet baseline for the B0 microscopy experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


DEFAULT_CSV = Path(
    "/home/hjk4090d/microscopy_project/annotations/dataset_index/"
    "supervised_patch_index_512_stride512.csv"
)
DEFAULT_OUT_DIR = Path("/home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline")
PATCH_SIZE = 512
OVERLAY_COUNT = 20
VALID_SPLITS = {"train", "val", "test"}

REQUIRED_COLUMNS = (
    "sample_id",
    "split",
    "mcy_path",
    "mask_path",
    "y0",
    "x0",
    "y1",
    "x1",
    "patch_h",
    "patch_w",
    "frame_height",
    "frame_width",
    "foreground_ratio_patch",
    "has_foreground",
)
FRAME_COLUMNS = (
    "t",
    "frame_index",
    "frame_idx",
    "frame",
    "time_index",
    "time_idx",
    "timepoint",
)


@dataclass(frozen=True)
class CsvColumns:
    sample_id: str
    split: str
    mcy_path: str
    mask_path: str
    y0: str
    x0: str
    y1: str
    x1: str
    patch_h: str
    patch_w: str
    frame_height: str
    frame_width: str
    foreground_ratio_patch: str
    has_foreground: str
    frame: str


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    num_patches: int = 0
    intersection: float = 0.0
    pred_sum: float = 0.0
    target_sum: float = 0.0
    union: float = 0.0
    total_pixels: float = 0.0

    def add(self, loss: float, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_bin = pred.float()
        target_bin = target.float()
        self.loss_sum += float(loss)
        self.num_patches += 1
        self.intersection += float(torch.sum(pred_bin * target_bin).item())
        pred_count = float(torch.sum(pred_bin).item())
        target_count = float(torch.sum(target_bin).item())
        self.pred_sum += pred_count
        self.target_sum += target_count
        self.union += float(torch.sum((pred_bin + target_bin) > 0).item())
        self.total_pixels += float(target_bin.numel())

    def metrics(self) -> dict[str, float | int]:
        eps = 1e-6
        if self.num_patches == 0:
            return {
                "loss": float("nan"),
                "binary_dice": float("nan"),
                "binary_iou": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "foreground_pixel_ratio": float("nan"),
                "num_patches": 0,
            }
        return {
            "loss": self.loss_sum / self.num_patches,
            "binary_dice": (2.0 * self.intersection + eps)
            / (self.pred_sum + self.target_sum + eps),
            "binary_iou": (self.intersection + eps) / (self.union + eps),
            "precision": (self.intersection + eps) / (self.pred_sum + eps),
            "recall": (self.intersection + eps) / (self.target_sum + eps),
            "foreground_pixel_ratio": self.target_sum / max(self.total_pixels, eps),
            "num_patches": self.num_patches,
        }


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def find_column(fieldnames: Iterable[str], aliases: Iterable[str], label: str) -> str:
    normalized = {normalize_column_name(name): name for name in fieldnames}
    for alias in aliases:
        column = normalized.get(normalize_column_name(alias))
        if column is not None:
            return column
    tried = ", ".join(aliases)
    available = ", ".join(fieldnames)
    raise ValueError(
        f"CSV is missing {label} column. Tried [{tried}]. Available columns: [{available}]"
    )


def resolve_columns(fieldnames: list[str]) -> CsvColumns:
    values: dict[str, str] = {}
    for column in REQUIRED_COLUMNS:
        values[column] = find_column(fieldnames, (column,), column)
    values["frame"] = find_column(fieldnames, FRAME_COLUMNS, "frame index / t")
    return CsvColumns(**values)


def load_csv_rows(csv_path: Path | str) -> tuple[list[dict[str, str]], CsvColumns, Path]:
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        columns = resolve_columns(list(reader.fieldnames))
        rows = []
        for row_number, row in enumerate(reader, start=2):
            row["__row_number"] = str(row_number)
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows, columns, path


def parse_int(value: str, label: str, row_number: int) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: missing integer value for {label}")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: cannot parse {label}={text!r} as int") from exc


def parse_float(value: str, label: str, row_number: int) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: missing float value for {label}")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: cannot parse {label}={text!r} as float") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Row {row_number}: {label} is not finite: {text!r}")
    return parsed


def resolve_path(value: str, csv_path: Path, row_number: int, label: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: empty path in {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = csv_path.parent / path
    return path


def squeeze_grayscale(array: np.ndarray, label: str) -> np.ndarray:
    squeezed = np.asarray(array)
    while squeezed.ndim > 2 and 1 in squeezed.shape:
        squeezed = np.squeeze(squeezed)
    if squeezed.ndim != 2:
        raise ValueError(f"{label} must be a single 2D image, got shape {array.shape}")
    return squeezed


def read_tiff_page(path: Path, frame_index: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"TIFF path does not exist: {path}")
    if frame_index < 0:
        raise IndexError(f"Frame index must be non-negative, got {frame_index}")
    try:
        with tifffile.TiffFile(path) as tif:
            page_count = len(tif.pages)
            if frame_index >= page_count:
                raise IndexError(
                    f"Frame index {frame_index} out of range for {path} "
                    f"with {page_count} pages"
                )
            return squeeze_grayscale(tif.pages[frame_index].asarray(), f"TIFF page {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF page t={frame_index} from {path}: {exc}") from exc


def read_mask_png(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask path does not exist: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask PNG with cv2.imread: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[2] >= 3 and np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(
            mask[:, :, 0], mask[:, :, 2]
        ):
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Mask PNG must be single-channel, got shape {mask.shape}: {path}")
    return squeeze_grayscale(mask, f"mask PNG {path}")


def crop_patch(
    array: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    expected_h: int,
    expected_w: int,
    label: str,
    allow_precropped: bool = False,
) -> np.ndarray:
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"{label}: invalid crop y0={y0}, y1={y1}, x0={x0}, x1={x1}")
    height, width = array.shape
    if 0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width:
        patch = array[y0:y1, x0:x1]
    elif allow_precropped and array.shape == (expected_h, expected_w):
        patch = array
    else:
        raise ValueError(
            f"{label}: crop y={y0}:{y1}, x={x0}:{x1} is outside image shape {array.shape}"
        )
    if patch.shape != (expected_h, expected_w):
        raise ValueError(f"{label}: expected patch shape {(expected_h, expected_w)}, got {patch.shape}")
    return patch


def percentile_normalize(image: np.ndarray) -> np.ndarray:
    p_low, p_high = np.percentile(image, [1.0, 99.5])
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        return np.zeros(image.shape, dtype=np.float32)
    normalized = (image.astype(np.float32, copy=False) - np.float32(p_low)) / np.float32(
        p_high - p_low
    )
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def density_bucket(foreground_ratio: float) -> str:
    if foreground_ratio == 0:
        return "empty"
    if 0 < foreground_ratio <= 0.01:
        return "low"
    if foreground_ratio <= 0.10:
        return "medium"
    return "high"


def validate_row(row: dict[str, str], columns: CsvColumns) -> dict[str, int | float | str]:
    row_number = parse_int(row["__row_number"], "__row_number", 0)
    split = str(row[columns.split]).strip().lower()
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Row {row_number}: split must be one of {sorted(VALID_SPLITS)}, got {split!r}"
        )

    y0 = parse_int(row[columns.y0], columns.y0, row_number)
    x0 = parse_int(row[columns.x0], columns.x0, row_number)
    y1 = parse_int(row[columns.y1], columns.y1, row_number)
    x1 = parse_int(row[columns.x1], columns.x1, row_number)
    patch_h = parse_int(row[columns.patch_h], columns.patch_h, row_number)
    patch_w = parse_int(row[columns.patch_w], columns.patch_w, row_number)
    frame_height = parse_int(row[columns.frame_height], columns.frame_height, row_number)
    frame_width = parse_int(row[columns.frame_width], columns.frame_width, row_number)
    frame_index = parse_int(row[columns.frame], columns.frame, row_number)
    foreground_ratio = parse_float(
        row[columns.foreground_ratio_patch],
        columns.foreground_ratio_patch,
        row_number,
    )

    if patch_h != PATCH_SIZE or patch_w != PATCH_SIZE:
        raise ValueError(f"Row {row_number}: expected 512x512 patch, got {patch_h}x{patch_w}")
    if y1 - y0 != patch_h or x1 - x0 != patch_w:
        raise ValueError(
            f"Row {row_number}: crop size {(y1 - y0, x1 - x0)} does not match "
            f"patch_h/patch_w {(patch_h, patch_w)}"
        )
    if y0 < 0 or x0 < 0 or y1 > frame_height or x1 > frame_width:
        raise ValueError(
            f"Row {row_number}: crop y={y0}:{y1}, x={x0}:{x1} outside "
            f"frame shape {(frame_height, frame_width)}"
        )
    return {
        "row_number": row_number,
        "split": split,
        "y0": y0,
        "x0": x0,
        "y1": y1,
        "x1": x1,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "frame_index": frame_index,
        "foreground_ratio_patch": foreground_ratio,
        "density_bucket": density_bucket(foreground_ratio),
    }


class StrictFramePatchDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        columns: CsvColumns,
        csv_path: Path,
        split_name: str,
    ) -> None:
        if not rows:
            raise ValueError(f"No rows provided for split {split_name!r}")
        self.rows = rows
        self.columns = columns
        self.csv_path = csv_path
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.rows)

    def read_item_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        row = self.rows[index]
        values = validate_row(row, self.columns)
        row_number = int(values["row_number"])
        mcy_path = resolve_path(row[self.columns.mcy_path], self.csv_path, row_number, self.columns.mcy_path)
        mask_path = resolve_path(
            row[self.columns.mask_path],
            self.csv_path,
            row_number,
            self.columns.mask_path,
        )

        image = read_tiff_page(mcy_path, int(values["frame_index"]))
        image_patch = crop_patch(
            image,
            int(values["y0"]),
            int(values["x0"]),
            int(values["y1"]),
            int(values["x1"]),
            PATCH_SIZE,
            PATCH_SIZE,
            f"image row {row_number}",
        )
        mask = read_mask_png(mask_path)
        mask_patch = crop_patch(
            mask,
            int(values["y0"]),
            int(values["x0"]),
            int(values["y1"]),
            int(values["x1"]),
            PATCH_SIZE,
            PATCH_SIZE,
            f"mask row {row_number}",
            allow_precropped=True,
        )

        meta = {
            "row_number": row_number,
            "sample_id": str(row[self.columns.sample_id]).strip(),
            "split": str(values["split"]),
            "frame_index": int(values["frame_index"]),
            "foreground_ratio_patch": float(values["foreground_ratio_patch"]),
            "density_bucket": str(values["density_bucket"]),
            "mcy_path": str(mcy_path),
            "mask_path": str(mask_path),
        }
        if not meta["sample_id"]:
            raise ValueError(f"Row {row_number}: empty sample_id")
        return image_patch, mask_patch, meta

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        image_patch, mask_patch, meta = self.read_item_arrays(index)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(percentile_normalize(image_patch)[None, :, :])
        ).float()
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray((mask_patch > 0).astype(np.float32, copy=False)[None, :, :])
        ).float()
        if tuple(image_tensor.shape) != (1, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Image tensor has wrong shape {tuple(image_tensor.shape)}")
        if tuple(mask_tensor.shape) != (1, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Mask tensor has wrong shape {tuple(mask_tensor.shape)}")
        return image_tensor, mask_tensor, meta


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FrameUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.bottleneck = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def loss_per_sample(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=(1, 2, 3))
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    return bce + (1.0 - dice)


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return loss_per_sample(logits, targets).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False")
    return device


def split_rows_by_csv(
    rows: list[dict[str, str]],
    columns: CsvColumns,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
) -> dict[str, list[dict[str, str]]]:
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        row_number = parse_int(row["__row_number"], "__row_number", 0)
        split = str(row[columns.split]).strip().lower()
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Row {row_number}: split must be one of {sorted(VALID_SPLITS)}, got {split!r}"
            )
        split_rows[split].append(row)

    caps = {"train": max_train, "val": max_val, "test": max_test}
    for split, cap in caps.items():
        if cap is not None:
            if cap <= 0:
                raise ValueError(f"--max-{split}-patches must be positive when provided")
            split_rows[split] = split_rows[split][:cap]
        if not split_rows[split]:
            raise ValueError(f"No rows found for required split {split!r}")
    return split_rows


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def require_batch_shapes(images: torch.Tensor, masks: torch.Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Image batch has wrong shape {tuple(images.shape)}")
    if masks.ndim != 4 or tuple(masks.shape[1:]) != (1, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Mask batch has wrong shape {tuple(masks.shape)}")


def meta_values(meta: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = meta[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value for _ in range(batch_size)]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    use_amp: bool,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    patch_count = 0

    for images, masks, _meta in loader:
        require_batch_shapes(images, masks)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = combined_loss(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Training loss became NaN or Inf at step {global_step + 1}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(images.shape[0])
        global_step += 1
        loss_sum += float(loss.item()) * batch_size
        patch_count += batch_size
        if global_step % 100 == 0:
            print(
                f"epoch={epoch} step={global_step} train_loss={float(loss.item()):.6f}",
                flush=True,
            )

    if patch_count == 0:
        raise RuntimeError(f"Training loader produced no batches in epoch {epoch}")
    return loss_sum / patch_count, global_step


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    use_amp: bool,
    collect_groups: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    overall = MetricAccumulator()
    sample_accs: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    bucket_accs: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)

    with torch.no_grad():
        for images, masks, meta in loader:
            require_batch_shapes(images, masks)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                sample_losses = loss_per_sample(logits, masks)
            if not torch.isfinite(sample_losses).all():
                raise FloatingPointError(f"{split_name} evaluation loss became NaN or Inf")

            preds = (torch.sigmoid(logits) >= 0.5).float()
            batch_size = int(images.shape[0])
            sample_ids = meta_values(meta, "sample_id", batch_size)
            buckets = meta_values(meta, "density_bucket", batch_size)

            for i in range(batch_size):
                loss_value = float(sample_losses[i].item())
                overall.add(loss_value, preds[i], masks[i])
                if collect_groups:
                    sample_accs[str(sample_ids[i])].add(loss_value, preds[i], masks[i])
                    bucket_accs[str(buckets[i])].add(loss_value, preds[i], masks[i])

    split_metrics = overall.metrics()
    sample_rows = [
        {"split": split_name, "sample_id": sample_id, **acc.metrics()}
        for sample_id, acc in sorted(sample_accs.items())
    ]
    bucket_order = {"empty": 0, "low": 1, "medium": 2, "high": 3}
    bucket_rows = [
        {"split": split_name, "density_bucket": bucket, **acc.metrics()}
        for bucket, acc in sorted(bucket_accs.items(), key=lambda item: bucket_order.get(item[0], 99))
    ]
    return split_metrics, sample_rows, bucket_rows


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": json_safe(config),
            "metrics": json_safe(metrics),
        },
        path,
    )


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:80] if cleaned else "sample"


def draw_boundary(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, color, 1)


def generate_prediction_overlays(
    model: nn.Module,
    datasets: list[tuple[str, StrictFramePatchDataset]],
    out_dir: Path,
    device: torch.device,
    use_amp: bool,
    total_count: int = OVERLAY_COUNT,
) -> list[str]:
    overlay_dir = out_dir / "prediction_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    model.eval()

    with torch.no_grad():
        for split_name, dataset in datasets:
            if len(saved) >= total_count:
                break
            per_split_limit = max(1, total_count // max(len(datasets), 1))
            for index in range(min(len(dataset), per_split_limit)):
                if len(saved) >= total_count:
                    break
                image_patch, mask_patch, meta = dataset.read_item_arrays(index)
                image_norm = percentile_normalize(image_patch)
                image_tensor = torch.from_numpy(
                    np.ascontiguousarray(image_norm[None, None, :, :])
                ).float()
                with autocast(enabled=use_amp):
                    logits = model(image_tensor.to(device))
                pred = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
                gt = (mask_patch > 0).astype(np.uint8)

                base = (np.clip(image_norm, 0.0, 1.0) * 255).astype(np.uint8)
                rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
                draw_boundary(rgb, gt, (0, 255, 0))
                draw_boundary(rgb, pred, (255, 0, 0))

                sample = sanitize_filename(str(meta["sample_id"]))
                filename = f"{split_name}_{len(saved):03d}_{sample}_t{meta['frame_index']}.png"
                out_path = overlay_dir / filename
                ok = cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"Failed to write prediction overlay: {out_path}")
                saved.append(str(out_path))
    return saved


def build_datasets(args: argparse.Namespace) -> tuple[dict[str, StrictFramePatchDataset], CsvColumns, Path]:
    rows, columns, csv_path = load_csv_rows(args.csv)
    split_rows = split_rows_by_csv(
        rows,
        columns,
        args.max_train_patches,
        args.max_val_patches,
        args.max_test_patches,
    )
    datasets = {
        split: StrictFramePatchDataset(split_rows[split], columns, csv_path, split)
        for split in ("train", "val", "test")
    }
    return datasets, columns, csv_path


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, _columns, csv_path = build_datasets(args)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        datasets["train"],
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    eval_loaders = {
        split: make_loader(
            dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        )
        for split, dataset in datasets.items()
    }

    model = FrameUNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)

    config = {
        **vars(args),
        "csv": csv_path,
        "out_dir": out_dir,
        "use_amp": use_amp,
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
    }
    write_json(out_dir / "frameunet_strict_config.json", config)

    best_path = out_dir / "frameunet_strict_best.pt"
    last_path = out_dir / "frameunet_strict_last.pt"
    best_val_dice = -1.0
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            global_step,
            use_amp,
        )
        val_metrics, _sample_rows, _bucket_rows = evaluate_model(
            model,
            eval_loaders["val"],
            device,
            "val",
            use_amp,
            collect_groups=False,
        )
        val_dice = float(val_metrics["binary_dice"])
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_binary_dice": val_metrics["binary_dice"],
            "val_binary_iou": val_metrics["binary_iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_foreground_pixel_ratio": val_metrics["foreground_pixel_ratio"],
            "val_num_patches": val_metrics["num_patches"],
        }
        epoch_rows.append(epoch_row)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={float(val_metrics['loss']):.6f} "
            f"val_dice={val_dice:.6f} val_iou={float(val_metrics['binary_iou']):.6f}",
            flush=True,
        )

        save_checkpoint(last_path, model, optimizer, epoch, config, {"val": val_metrics})
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            save_checkpoint(best_path, model, optimizer, epoch, config, {"val": val_metrics})

    write_rows_csv(
        out_dir / "frameunet_strict_epoch_log.csv",
        epoch_rows,
        [
            "epoch",
            "train_loss",
            "val_loss",
            "val_binary_dice",
            "val_binary_iou",
            "val_precision",
            "val_recall",
            "val_foreground_pixel_ratio",
            "val_num_patches",
        ],
    )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    split_metric_rows: list[dict[str, Any]] = []
    sample_metric_rows: list[dict[str, Any]] = []
    bucket_metric_rows: list[dict[str, Any]] = []
    final_metrics: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "test"):
        split_metrics, sample_rows, bucket_rows = evaluate_model(
            model,
            eval_loaders[split],
            device,
            split,
            use_amp,
            collect_groups=True,
        )
        final_metrics[split] = split_metrics
        split_metric_rows.append({"split": split, **split_metrics})
        sample_metric_rows.extend(sample_rows)
        bucket_metric_rows.extend(bucket_rows)

    metric_fields = [
        "loss",
        "binary_dice",
        "binary_iou",
        "precision",
        "recall",
        "foreground_pixel_ratio",
        "num_patches",
    ]
    write_rows_csv(
        out_dir / "frameunet_strict_split_metrics.csv",
        split_metric_rows,
        ["split", *metric_fields],
    )
    write_rows_csv(
        out_dir / "frameunet_strict_sample_metrics.csv",
        sample_metric_rows,
        ["split", "sample_id", *metric_fields],
    )
    write_rows_csv(
        out_dir / "frameunet_strict_density_bucket_metrics.csv",
        bucket_metric_rows,
        ["split", "density_bucket", *metric_fields],
    )

    overlays = generate_prediction_overlays(
        model,
        [("val", datasets["val"]), ("test", datasets["test"])],
        out_dir,
        device,
        use_amp,
        total_count=OVERLAY_COUNT,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "final_metrics": final_metrics,
        "outputs": {
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
            "epoch_log": out_dir / "frameunet_strict_epoch_log.csv",
            "split_metrics": out_dir / "frameunet_strict_split_metrics.csv",
            "sample_metrics": out_dir / "frameunet_strict_sample_metrics.csv",
            "density_bucket_metrics": out_dir / "frameunet_strict_density_bucket_metrics.csv",
            "config": out_dir / "frameunet_strict_config.json",
            "prediction_overlays": overlays,
        },
        "config": config,
    }
    write_json(out_dir / "frameunet_strict_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-patches", type=int, default=None)
    parser.add_argument("--max-val-patches", type=int, default=None)
    parser.add_argument("--max-test-patches", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_training(args)
    print("FRAMEUNET_STRICT_BASELINE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
