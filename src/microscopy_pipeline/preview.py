"""QC preview generation for paired DIC/MCY TIFF stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from microscopy_pipeline.io import inspect_tiff, read_tiff_page
from microscopy_pipeline.normalize import percentile_normalize


def auto_frames(frame_count: int) -> list[int]:
    """Return first, middle, and last frame indices without duplicates."""

    if frame_count <= 0:
        return []
    return sorted({0, frame_count // 2, frame_count - 1})


def parse_frame_argument(value: str | None, frame_count: int) -> list[int]:
    """Parse 'auto' or a comma-separated frame list."""

    if value is None or value.lower() == "auto":
        return auto_frames(frame_count)
    frames = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    for frame in frames:
        if frame < 0 or frame >= frame_count:
            raise ValueError(f"Frame {frame} out of range for {frame_count} frames")
    return frames


def save_paired_preview(
    sample_id: str,
    dic_path: Path | str,
    mcy_path: Path | str,
    frame_index: int,
    output_dir: Path | str,
) -> Path:
    """Save one side-by-side DIC/MCY PNG for a single frame."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dic_image = percentile_normalize(read_tiff_page(dic_path, frame_index))
    mcy_image = percentile_normalize(read_tiff_page(mcy_path, frame_index))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(dic_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"{sample_id} DIC t={frame_index}")
    axes[1].imshow(mcy_image, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"{sample_id} MCY t={frame_index}")
    for axis in axes:
        axis.axis("off")

    out_path = out_dir / f"{sample_id}_paired_frame_{frame_index}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_contact_sheet(sample_id: str, image_paths: Sequence[Path], output_dir: Path | str) -> Path | None:
    """Create a simple horizontal contact sheet from generated preview PNGs."""

    if len(image_paths) < 2:
        return None
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        sheet = Image.new("RGB", (width * len(images), height), color="white")
        for index, image in enumerate(images):
            sheet.paste(image, (index * width, 0))
        out_path = Path(output_dir) / f"{sample_id}_contact_sheet.png"
        sheet.save(out_path)
        return out_path
    finally:
        for image in images:
            image.close()


def create_sample_previews(
    sample_id: str,
    dic_path: Path | str,
    mcy_path: Path | str,
    preview_root: Path | str,
    frames: str | None = "auto",
) -> list[Path]:
    """Generate paired previews and a contact sheet for one OK sample."""

    dic_info = inspect_tiff(dic_path, read_first_page=False)
    mcy_info = inspect_tiff(mcy_path, read_first_page=False)
    frame_count = min(dic_info.pages, mcy_info.pages)
    selected_frames = parse_frame_argument(frames, frame_count)
    sample_dir = Path(preview_root) / sample_id
    outputs = [
        save_paired_preview(sample_id, dic_path, mcy_path, frame_index, sample_dir)
        for frame_index in selected_frames
    ]
    contact_sheet = save_contact_sheet(sample_id, outputs, sample_dir)
    if contact_sheet is not None:
        outputs.append(contact_sheet)
    return outputs
