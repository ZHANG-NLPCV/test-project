from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microscopy_pipeline.normalize import percentile_normalize


def test_percentile_normalization_returns_finite_float32_unit_range() -> None:
    image = np.array(
        [
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 1000],
        ],
        dtype=np.uint16,
    )

    normalized = percentile_normalize(image)

    assert normalized.dtype == np.float32
    assert normalized.shape == image.shape
    assert np.all(np.isfinite(normalized))
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_percentile_normalization_handles_constant_images() -> None:
    image = np.full((4, 4), 42, dtype=np.uint16)

    normalized = percentile_normalize(image)

    assert normalized.dtype == np.float32
    assert np.all(normalized == 0.0)
