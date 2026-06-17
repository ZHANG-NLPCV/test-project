from pathlib import Path
import sys

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microscopy_pipeline.manifest import build_manifest_rows


def _write_stack(path: Path, shape: tuple[int, int, int]) -> None:
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    tifffile.imwrite(path, data)


def test_dic_and_mcy_files_are_paired_by_keyword(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_a"
    sample_dir.mkdir()
    _write_stack(sample_dir / "dic-sub1.tif", (3, 5, 7))
    _write_stack(sample_dir / "mcy-sub1.tif", (3, 5, 7))

    rows = build_manifest_rows(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "sample_a"
    assert row["status"] == "OK"
    assert row["dic_shape"] == "(3, 5, 7)"
    assert row["mcy_shape"] == "(3, 5, 7)"
    assert row["dic_dtype"] == "uint16"
    assert row["mcy_dtype"] == "uint16"


def test_nonmatching_shape_is_reported_as_mismatch(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_mismatch"
    sample_dir.mkdir()
    _write_stack(sample_dir / "dic.tif", (2, 4, 4))
    _write_stack(sample_dir / "mcy.tif", (3, 4, 4))

    rows = build_manifest_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["status"] == "MISMATCH_SHAPE"


def test_manifest_creation_includes_skipped_sample_folders(tmp_path: Path) -> None:
    ok_dir = tmp_path / "ok_sample"
    ok_dir.mkdir()
    _write_stack(ok_dir / "dic.tif", (2, 3, 4))
    _write_stack(ok_dir / "mcy.tif", (2, 3, 4))

    missing_dir = tmp_path / "missing_mcy"
    missing_dir.mkdir()
    _write_stack(missing_dir / "dic.tif", (2, 3, 4))

    rows = build_manifest_rows(tmp_path)

    by_sample = {row["sample_id"]: row for row in rows}

    assert by_sample["ok_sample"]["status"] == "OK"
    assert by_sample["missing_mcy"]["status"].startswith("SKIP")
