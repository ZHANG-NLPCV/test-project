"""Image normalization helpers."""

from __future__ import annotations

import numpy as np


def percentile_normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    sample_step: int = 8,
) -> np.ndarray:
    """Normalize an image to float32 [0, 1] using sampled percentiles."""

    array = np.asarray(image)
    sampled = array[::sample_step, ::sample_step] if array.ndim >= 2 else array
    if sampled.size == 0:
        sampled = array

    p_low, p_high = np.percentile(sampled, [lower_percentile, upper_percentile])
    output = array.astype(np.float32, copy=False)
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        return np.zeros(array.shape, dtype=np.float32)

    output = (output - np.float32(p_low)) / np.float32(p_high - p_low)
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
