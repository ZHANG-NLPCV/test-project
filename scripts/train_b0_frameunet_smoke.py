#!/usr/bin/env python3
"""B0 smoke training for frame-level foreground segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_CSV = Path(
    "/home/hjk4090d/microscopy_project/annotations/dataset_index/"
    "supervised_patch_index_512_stride512.csv"
)
DEFAULT_OUT_DIR = Path("/home/hjk4090d/microscopy_project/runs/b0_frameunet_smoke")
PATCH_SIZE = 512

MCY_PATH_COLUMNS = (
    "mcy_path",
    "mcy_tiff_path",
    "mcy_tif_path",
    "mcy_file",
    "mcy_stack_path",
    "image_path",
    "image_tiff_path",
    "tiff_path",
    "source_tiff_path",
)
MASK_PATH_COLUMNS = (
    "instance_mask_path",
    "instance_mask_png_path",
    "mask_path",
    "mask_png_path",
    "label_path",
    "label_png_path",
    "target_path",
)
FRAME_COLUMNS = (
    "frame_index",
    "frame_idx",
    "frame",
    "t",
    "time_index",
    "time_idx",
    "timepoint",
    "center_frame",
    "center_frame_index",
    "center_t",
    "target_t",
)
X_COLUMNS = (
    "x0",
    "x",
    "xmin",
    "x_start",
    "patch_x",
    "patch_left",
    "left",
    "col",
    "col0",
    "col_start",
    "tile_x",
    "x_offset",
    "x0_px",
)
Y_COLUMNS = (
    "y0",
    "y",
    "ymin",
    "y_start",
    "patch_y",
    "patch_top",
    "top",
    "row",
    "row0",
    "row_start",
    "tile_y",
    "y_offset",
    "y0_px",
)


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def find_column(fieldnames: Iterable[str], aliases: Iterable[str], label: str) -> str:
    column = find_optional_column(fieldnames, aliases)
    if column is None:
        alias_text = ", ".join(aliases)
        available = ", ".join(fieldnames)
        raise ValueError(
            f"CSV is missing {label} column. Tried [{alias_text}]. "
            f"Available columns: [{available}]"
        )
    return column


def find_optional_column(fieldnames: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {normalize_column_name(name): name for name in fieldnames}
    for alias in aliases:
        column = normalized.get(normalize_column_name(alias))
        if column is not None:
            return column
    return None


def load_csv_rows(csv_path: Path | str) -> tuple[list[dict[str, str]], list[str], Path]:
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        rows = [row for row in reader]

    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows, list(reader.fieldnames), path


def parse_int(value: str, label: str, row_number: int) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: missing integer value for {label}")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: cannot parse {label}={text!r} as int") from exc


def row_path(row: dict[str, str], column: str, csv_path: Path, row_number: int) -> Path:
    value = str(row.get(column, "")).strip()
    if not value:
        raise ValueError(f"Row {row_number}: empty path in column {column}")
    path = Path(value).expanduser()
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


def read_tiff_frame(path: Path, frame_index: int) -> np.ndarray:
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
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask PNG: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(
            mask[:, :, 0], mask[:, :, 2]
        ):
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Mask PNG must be single-channel, got shape {mask.shape}: {path}")
    return squeeze_grayscale(mask, f"mask PNG {path}")


def crop_or_validate_patch(
    array: np.ndarray,
    x0: int | None,
    y0: int | None,
    patch_size: int,
    label: str,
) -> np.ndarray:
    if array.shape == (patch_size, patch_size):
        return array
    if x0 is None or y0 is None:
        raise ValueError(
            f"{label} has shape {array.shape}, not ({patch_size}, {patch_size}), "
            "and CSV has no patch coordinate columns"
        )

    height, width = array.shape
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"{label} crop [{x0}:{x1}, {y0}:{y1}] is outside image shape {array.shape}"
        )
    return array[y0:y1, x0:x1]


def percentile_normalize(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    p_low, p_high = np.percentile(array, [1.0, 99.5])
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        return np.zeros(array.shape, dtype=np.float32)
    normalized = (array.astype(np.float32, copy=False) - np.float32(p_low)) / np.float32(
        p_high - p_low
    )
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def image_patch_tensor(image: np.ndarray) -> torch.Tensor:
    normalized = percentile_normalize(image)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized[None, :, :]))
    if tuple(tensor.shape) != (1, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Image tensor has wrong shape {tuple(tensor.shape)}")
    return tensor.float()


def mask_patch_tensor(mask: np.ndarray) -> torch.Tensor:
    binary = (mask > 0).astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(binary[None, :, :]))
    if tuple(tensor.shape) != (1, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Mask tensor has wrong shape {tuple(tensor.shape)}")
    return tensor.float()


class FramePatchDataset(Dataset):
    def __init__(
        self,
        csv_path: Path | str,
        rows: list[dict[str, str]] | None = None,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        all_rows, fieldnames, resolved_csv = load_csv_rows(csv_path)
        self.rows = rows if rows is not None else all_rows
        self.csv_path = resolved_csv
        self.patch_size = patch_size
        self.mcy_col = find_column(fieldnames, MCY_PATH_COLUMNS, "MCY TIFF path")
        self.mask_col = find_column(fieldnames, MASK_PATH_COLUMNS, "instance mask PNG path")
        self.frame_col = find_column(fieldnames, FRAME_COLUMNS, "frame index")
        self.x_col = find_optional_column(fieldnames, X_COLUMNS)
        self.y_col = find_optional_column(fieldnames, Y_COLUMNS)
        if (self.x_col is None) != (self.y_col is None):
            raise ValueError(
                "CSV must provide both x and y patch coordinate columns, or neither. "
                f"Found x={self.x_col!r}, y={self.y_col!r}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        row_number = index + 2
        mcy_path = row_path(row, self.mcy_col, self.csv_path, row_number)
        mask_path = row_path(row, self.mask_col, self.csv_path, row_number)
        frame_index = parse_int(row[self.frame_col], self.frame_col, row_number)
        x0 = parse_int(row[self.x_col], self.x_col, row_number) if self.x_col else None
        y0 = parse_int(row[self.y_col], self.y_col, row_number) if self.y_col else None

        image = read_tiff_frame(mcy_path, frame_index)
        image = crop_or_validate_patch(image, x0, y0, self.patch_size, f"image row {row_number}")
        mask = read_mask_png(mask_path)
        mask = crop_or_validate_patch(mask, x0, y0, self.patch_size, f"mask row {row_number}")
        return image_patch_tensor(image), mask_patch_tensor(mask)


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


class TinyUNet(nn.Module):
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


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_loss: nn.Module,
) -> torch.Tensor:
    return bce_loss(logits, targets) + dice_loss(logits, targets)


def split_rows(
    rows: list[dict[str, str]],
    train_nmax: int,
    val_nmax: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(rows) < 2:
        raise ValueError(f"Need at least 2 rows for train/val split, got {len(rows)}")

    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_count = min(train_nmax, len(rows) - 1)
    val_count = min(val_nmax, len(rows) - train_count)
    if train_count <= 0 or val_count <= 0:
        raise ValueError(
            f"Invalid split train_n={train_count}, val_n={val_count} from {len(rows)} rows"
        )

    train_rows = [rows[index] for index in indices[:train_count]]
    val_rows = [rows[index] for index in indices[train_count : train_count + val_count]]
    return train_rows, val_rows


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    bce_loss: nn.Module,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    intersection = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    union = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = combined_loss(logits, masks, bce_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss became NaN or Inf")

            preds = (torch.sigmoid(logits) >= 0.5).float()
            intersection += float(torch.sum(preds * masks).item())
            pred_sum += float(torch.sum(preds).item())
            target_sum += float(torch.sum(masks).item())
            union += float(torch.sum((preds + masks) > 0).item())
            loss_sum += float(loss.item())
            batch_count += 1

    if batch_count == 0:
        raise RuntimeError("Validation loader produced no batches")

    eps = 1e-6
    return {
        "mean_val_loss": loss_sum / batch_count,
        "val_dice": (2.0 * intersection + eps) / (pred_sum + target_sum + eps),
        "val_iou": (intersection + eps) / (union + eps),
    }


def write_summary_csv(path: Path, metrics: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = resolve_device(args.device)

    all_rows, _, csv_path = load_csv_rows(args.csv)
    train_rows, val_rows = split_rows(all_rows, args.train_nmax, args.val_nmax, args.seed)
    train_ds = FramePatchDataset(csv_path, rows=train_rows)
    val_ds = FramePatchDataset(csv_path, rows=val_rows)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = TinyUNet(in_channels=1, out_channels=1, base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "frameunet_smoke.pt"
    summary_path = out_dir / "frameunet_smoke_summary.csv"
    log_path = out_dir / "frameunet_smoke_log.json"

    global_step = 0
    train_losses: list[float] = []
    epoch_logs: list[dict[str, float | int]] = []
    final_train_loss = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for images, masks in train_loader:
            global_step += 1
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = combined_loss(logits, masks, bce_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Training loss became NaN or Inf at step {global_step}")
            loss.backward()
            optimizer.step()

            final_train_loss = float(loss.item())
            train_losses.append(final_train_loss)
            epoch_losses.append(final_train_loss)
            if global_step % 20 == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={final_train_loss:.6f}",
                    flush=True,
                )

        if not epoch_losses:
            raise RuntimeError(f"Training loader produced no batches in epoch {epoch}")
        val_metrics = validate(model, val_loader, device, bce_loss)
        epoch_log = {
            "epoch": epoch,
            "mean_train_loss": float(np.mean(epoch_losses)),
            **val_metrics,
        }
        epoch_logs.append(epoch_log)
        print(
            f"epoch={epoch} mean_train_loss={epoch_log['mean_train_loss']:.6f} "
            f"mean_val_loss={epoch_log['mean_val_loss']:.6f} "
            f"val_dice={epoch_log['val_dice']:.6f} val_iou={epoch_log['val_iou']:.6f}",
            flush=True,
        )

    if not train_losses or not np.isfinite(final_train_loss):
        raise FloatingPointError("Final train loss is not finite")

    final_metrics: dict[str, object] = {
        "csv": str(csv_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "mean_train_loss": float(np.mean(train_losses)),
        "final_train_loss": final_train_loss,
        "mean_val_loss": epoch_logs[-1]["mean_val_loss"],
        "val_dice": epoch_logs[-1]["val_dice"],
        "val_iou": epoch_logs[-1]["val_iou"],
        "checkpoint": str(checkpoint_path),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": final_metrics,
        },
        checkpoint_path,
    )
    write_summary_csv(summary_path, final_metrics)
    log_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "epochs": epoch_logs,
                "final": final_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-nmax", type=int, default=512)
    parser.add_argument("--val-nmax", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train(args)
    print("FRAMEUNET_SMOKE_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
