#!/usr/bin/env python3
"""Instance-level evaluation for the strict-split FrameUNet binary baseline.

The T=1 / T=5 baselines only predict a *binary foreground* mask. That number
(Dice ~0.94) hides the hard part: separating touching cells into individual
instances, which is exactly what SC-Track / pcnaDeep care about. This script
does NOT retrain anything. It reuses the trained binary checkpoint, runs the
model on each strict-split patch, turns the binary foreground prediction into
instances by post-processing (connected components, and optionally a
distance-transform watershed), and scores them against the *instance* ground
truth that the training pipeline currently throws away with ``mask > 0``.

Reported instance metrics (per split, per density bucket, per sample):

- ``seg``        CTC-style SEG: mean IoU of GT instances matched at IoU > 0.5.
- ``ap50/ap75``  Cellpose-style AP = TP / (TP + FP + FN) at IoU 0.50 / 0.75.
- ``map5095``    mean AP over IoU thresholds 0.50:0.05:0.95.
- ``mean_matched_iou`` mean IoU over matched (TP@0.5) pairs.
- ``count_ratio``      predicted instance count / GT instance count
                       (< 1 => under-segmentation / merged cells).

Run it once per post-processing method to see the floor (``cc``) versus what a
cheap watershed buys (``watershed``).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse the training script's CSV loading, dataset, and model so the data path
# (lazy TIFF read, 512x512 crop, percentile normalize) stays byte-for-byte
# identical to how the baseline was trained.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_b0_frameunet_strict import (  # noqa: E402
    DEFAULT_CSV,
    FrameUNet,
    StrictFramePatchDataset,
    density_bucket,
    json_safe,
    load_csv_rows,
    percentile_normalize,
    resolve_device,
    split_rows_by_csv,
    write_rows_csv,
    write_json,
)

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50 .. 0.95
DEFAULT_OUT_DIR = Path(
    "/home/hjk4090d/microscopy_project/runs/b0_frameunet_strict_baseline"
)
METHODS = ("cc", "watershed")


# --------------------------------------------------------------------------- #
# Instance extraction
# --------------------------------------------------------------------------- #
def gt_instances(mask_patch: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return an int32 instance-label image from a raw mask PNG crop.

    A true instance mask encodes each cell as a distinct positive integer. If
    the mask is effectively binary (one positive value spanning multiple blobs)
    we fall back to connected components so the script still produces a sane
    instance count. The bool flag reports whether that fallback was used.
    """
    labels = mask_patch.astype(np.int64)
    positive = np.unique(labels[labels > 0])
    if positive.size > 1:
        # Relabel to a compact 1..N int32 image.
        remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for new_id, old_id in enumerate(positive, start=1):
            remap[int(old_id)] = new_id
        out = remap[np.clip(labels, 0, None).astype(np.int64)]
        return out.astype(np.int32), False
    # Binary-looking GT: separate by connectivity instead.
    return connected_components((labels > 0).astype(np.uint8)), True


def connected_components(binary: np.ndarray) -> np.ndarray:
    import cv2

    num, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    _ = num
    return lab.astype(np.int32)


def watershed_instances(binary: np.ndarray) -> np.ndarray:
    """Distance-transform watershed to split touching blobs in a binary mask."""
    import cv2

    binary = (binary > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.int32)
    try:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        coords = peak_local_max(
            distance, min_distance=7, labels=binary, exclude_border=False
        )
        markers = np.zeros(distance.shape, dtype=np.int32)
        for idx, (row, col) in enumerate(coords, start=1):
            markers[row, col] = idx
        if markers.max() == 0:  # no peak found -> single instance
            return connected_components(binary)
        markers, _ = ndi.label(markers > 0)
        labels = watershed(-distance, markers, mask=binary.astype(bool))
        return labels.astype(np.int32)
    except Exception:
        # Any missing dependency / failure: degrade to connected components.
        return connected_components(binary)


# --------------------------------------------------------------------------- #
# Instance matching + per-patch counts
# --------------------------------------------------------------------------- #
def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, int, int]:
    """IoU matrix of shape (n_gt, n_pred); background label 0 is ignored."""
    n_gt = int(gt.max())
    n_pred = int(pred.max())
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64), n_gt, n_pred

    gt_flat = gt.reshape(-1).astype(np.int64)
    pred_flat = pred.reshape(-1).astype(np.int64)
    both = (gt_flat > 0) & (pred_flat > 0)
    pair = (gt_flat[both] - 1) * n_pred + (pred_flat[both] - 1)
    inter = np.bincount(pair, minlength=n_gt * n_pred).reshape(n_gt, n_pred)
    gt_area = np.bincount(gt_flat[gt_flat > 0] - 1, minlength=n_gt).reshape(n_gt, 1)
    pred_area = np.bincount(pred_flat[pred_flat > 0] - 1, minlength=n_pred).reshape(1, n_pred)
    union = gt_area + pred_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float64), n_gt, n_pred


def greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching of GT/pred by descending IoU above threshold."""
    if iou.size == 0:
        return []
    pairs = np.argwhere(iou >= threshold)
    if pairs.size == 0:
        return []
    order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]])
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gi, pi in pairs[order]:
        gi, pi = int(gi), int(pi)
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, float(iou[gi, pi])))
    return matches


@dataclass
class InstanceAccumulator:
    n_patches: int = 0
    n_gt: int = 0
    n_pred: int = 0
    seg_iou_sum: float = 0.0          # sum over GT instances of matched IoU (SEG)
    matched_iou_sum: float = 0.0      # sum of IoU for TP@0.5 pairs
    matched_count: int = 0
    tp: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    fp: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    fn: dict[float, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, gt: np.ndarray, pred: np.ndarray) -> None:
        iou, n_gt, n_pred = iou_matrix(gt, pred)
        self.n_patches += 1
        self.n_gt += n_gt
        self.n_pred += n_pred

        # SEG: best matched IoU per GT instance, counted only if > 0.5.
        if n_gt > 0 and n_pred > 0:
            best = iou.max(axis=1)
            self.seg_iou_sum += float(best[best > 0.5].sum())
        # (GT with no overlap contributes 0 to SEG, which is already the case.)

        for thr in IOU_THRESHOLDS:
            matches = greedy_match(iou, thr)
            tp = len(matches)
            self.tp[thr] += tp
            self.fp[thr] += n_pred - tp
            self.fn[thr] += n_gt - tp
            if abs(thr - 0.5) < 1e-9:
                self.matched_count += tp
                self.matched_iou_sum += float(sum(m[2] for m in matches))

    def metrics(self) -> dict[str, float | int]:
        def ap(thr: float) -> float:
            denom = self.tp[thr] + self.fp[thr] + self.fn[thr]
            return self.tp[thr] / denom if denom else float("nan")

        ap_values = [ap(thr) for thr in IOU_THRESHOLDS]
        finite = [v for v in ap_values if v == v]  # drop NaN
        tp50 = self.tp[0.5]
        precision50 = tp50 / (tp50 + self.fp[0.5]) if (tp50 + self.fp[0.5]) else float("nan")
        recall50 = tp50 / (tp50 + self.fn[0.5]) if (tp50 + self.fn[0.5]) else float("nan")
        f1_50 = (
            2 * precision50 * recall50 / (precision50 + recall50)
            if precision50 == precision50 and recall50 == recall50 and (precision50 + recall50) > 0
            else float("nan")
        )
        return {
            "n_patches": self.n_patches,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "count_ratio": (self.n_pred / self.n_gt) if self.n_gt else float("nan"),
            "seg": (self.seg_iou_sum / self.n_gt) if self.n_gt else float("nan"),
            "mean_matched_iou": (
                self.matched_iou_sum / self.matched_count if self.matched_count else float("nan")
            ),
            "ap50": ap(0.5),
            "ap75": ap(0.75),
            "map5095": float(np.mean(finite)) if finite else float("nan"),
            "precision50": precision50,
            "recall50": recall50,
            "f1_50": f1_50,
        }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_strict_datasets(
    csv_path: Path,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
) -> dict[str, StrictFramePatchDataset]:
    rows, columns, resolved_csv = load_csv_rows(csv_path)
    split_rows = split_rows_by_csv(rows, columns, max_train, max_val, max_test)
    return {
        split: StrictFramePatchDataset(split_rows[split], columns, resolved_csv, split)
        for split in ("train", "val", "test")
    }


def load_model(checkpoint_path: Path, device: torch.device) -> FrameUNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    base_channels = int(config.get("base_channels", 16))
    model = FrameUNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_binary(
    model: FrameUNet, image_patch: np.ndarray, device: torch.device, threshold: float
) -> np.ndarray:
    normalized = percentile_normalize(image_patch)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized[None, None, :, :])).float()
    with torch.no_grad():
        logits = model(tensor.to(device))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return (prob >= threshold).astype(np.uint8)


def evaluate_split(
    model: FrameUNet,
    dataset: StrictFramePatchDataset,
    split_name: str,
    device: torch.device,
    threshold: float,
    methods: tuple[str, ...],
    progress_every: int,
) -> dict[str, dict[str, InstanceAccumulator]]:
    """Returns accumulators keyed as scope -> group_label -> accumulator.

    Scopes: ``split`` (overall), ``bucket``, ``sample``. Group labels embed the
    post-processing method, e.g. ``"high|cc"``.
    """
    accs: dict[str, dict[str, InstanceAccumulator]] = {
        "split": defaultdict(InstanceAccumulator),
        "bucket": defaultdict(InstanceAccumulator),
        "sample": defaultdict(InstanceAccumulator),
    }
    gt_fallback_warned = False

    for index in range(len(dataset)):
        image_patch, mask_patch, meta = dataset.read_item_arrays(index)
        gt, used_fallback = gt_instances(mask_patch)
        if used_fallback and not gt_fallback_warned:
            print(
                f"[warn] {split_name}: GT mask looks binary (single label); "
                "using connected components for GT instances.",
                flush=True,
            )
            gt_fallback_warned = True

        binary = predict_binary(model, image_patch, device, threshold)
        bucket = str(meta.get("density_bucket") or density_bucket(float(meta["foreground_ratio_patch"])))
        sample_id = str(meta["sample_id"])

        for method in methods:
            pred = connected_components(binary) if method == "cc" else watershed_instances(binary)
            accs["split"][method].add(gt, pred)
            accs["bucket"][f"{bucket}|{method}"].add(gt, pred)
            accs["sample"][f"{sample_id}|{method}"].add(gt, pred)

        if progress_every and (index + 1) % progress_every == 0:
            print(f"  {split_name}: {index + 1}/{len(dataset)} patches", flush=True)

    return accs


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
METRIC_FIELDS = [
    "n_patches",
    "n_gt",
    "n_pred",
    "count_ratio",
    "seg",
    "mean_matched_iou",
    "ap50",
    "ap75",
    "map5095",
    "precision50",
    "recall50",
    "f1_50",
]
BUCKET_ORDER = {"empty": 0, "low": 1, "medium": 2, "high": 3}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(m for m in METHODS if m in set(args.methods))
    if not methods:
        raise ValueError(f"No valid methods selected from {METHODS}, got {args.methods}")

    datasets = build_strict_datasets(
        Path(args.csv), args.max_train_patches, args.max_val_patches, args.max_test_patches
    )
    model = load_model(Path(args.checkpoint).expanduser(), device)

    splits = [s for s in ("train", "val", "test") if s in args.splits]
    split_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    summary_final: dict[str, Any] = {}

    for split in splits:
        print(f"[eval] {split}: {len(datasets[split])} patches, methods={methods}", flush=True)
        accs = evaluate_split(
            model, datasets[split], split, device, args.threshold, methods, args.progress_every
        )
        summary_final[split] = {}
        for method in methods:
            metrics = accs["split"][method].metrics()
            summary_final[split][method] = metrics
            split_rows.append({"split": split, "method": method, **metrics})

        for key, acc in accs["bucket"].items():
            bucket, method = key.split("|", 1)
            bucket_rows.append(
                {"split": split, "density_bucket": bucket, "method": method, **acc.metrics()}
            )
        for key, acc in accs["sample"].items():
            sample_id, method = key.split("|", 1)
            sample_rows.append(
                {"split": split, "sample_id": sample_id, "method": method, **acc.metrics()}
            )

    bucket_rows.sort(key=lambda r: (r["split"], BUCKET_ORDER.get(r["density_bucket"], 99), r["method"]))
    sample_rows.sort(key=lambda r: (r["split"], r["sample_id"], r["method"]))

    write_rows_csv(
        out_dir / "instance_strict_split_metrics.csv",
        split_rows,
        ["split", "method", *METRIC_FIELDS],
    )
    write_rows_csv(
        out_dir / "instance_strict_density_bucket_metrics.csv",
        bucket_rows,
        ["split", "density_bucket", "method", *METRIC_FIELDS],
    )
    write_rows_csv(
        out_dir / "instance_strict_sample_metrics.csv",
        sample_rows,
        ["split", "sample_id", "method", *METRIC_FIELDS],
    )

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "csv": str(Path(args.csv)),
        "threshold": args.threshold,
        "methods": list(methods),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "final_metrics": summary_final,
        "outputs": {
            "split_metrics": str(out_dir / "instance_strict_split_metrics.csv"),
            "density_bucket_metrics": str(out_dir / "instance_strict_density_bucket_metrics.csv"),
            "sample_metrics": str(out_dir / "instance_strict_sample_metrics.csv"),
        },
    }
    write_json(out_dir / "instance_strict_summary.json", json_safe(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUT_DIR / "frameunet_strict_best.pt",
        help="Trained binary FrameUNet checkpoint (frameunet_strict_best.pt).",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="Post-processing to turn binary foreground into instances.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Which strict splits to evaluate.",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-train-patches", type=int, default=None)
    parser.add_argument("--max-val-patches", type=int, default=None)
    parser.add_argument("--max-test-patches", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    print("INSTANCE_STRICT_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
