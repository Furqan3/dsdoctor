"""Segmentation polygons and pose keypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from dsdoctor.dataset import (Dataset, SEGMENT, POSE, DETECT, polygon_area,
                              self_intersections, looks_like_polygon)
from dsdoctor.detectors import run
from dsdoctor.sweep import sweep


def _image(root: Path, split: str, stem: str, seed: int = 0):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(root / "images" / split / f"{stem}.jpg", quality=95)


def seg_dataset(tmp_path: Path, rows_by_stem: dict[str, list[str]],
                task: str = "segment") -> Path:
    root = tmp_path / "seg"
    for i, (stem, rows) in enumerate(rows_by_stem.items()):
        _image(root, "train", stem, seed=i)
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            "\n".join(rows) + "\n")
    cfg = {"names": ["thing"], "nc": 1}
    if task:
        cfg["task"] = task
    (root / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root


SQUARE = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"


# --------------------------------------------------------------- parsing

def test_looks_like_polygon_rejects_a_stray_column():
    assert looks_like_polygon(7) is True        # class + 3 points
    assert looks_like_polygon(9) is True        # class + 4 points
    assert looks_like_polygon(6) is False       # detection row + confidence
    assert looks_like_polygon(5) is False       # a plain detection row
    assert looks_like_polygon(8) is False       # odd coordinate count


def test_polygon_rows_derive_a_bounding_box(tmp_path):
    ds = Dataset(seg_dataset(tmp_path, {"a": [SQUARE]}))
    assert ds.task == SEGMENT
    box = ds.get("train/a").label.boxes[0]
    assert box.polygon == ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
    assert box.xc == pytest.approx(0.5) and box.yc == pytest.approx(0.5)
    assert box.w == pytest.approx(0.6) and box.h == pytest.approx(0.6)


def test_derived_boxes_make_existing_checks_work_on_segmentation(tmp_path):
    """A polygon leaving the frame is found by the ordinary geometry check."""
    root = seg_dataset(tmp_path, {"a": ["0 0.5 0.5 1.4 0.5 1.4 1.4 0.5 1.4"]})
    types = {f.type for f in sweep(Dataset(root)).findings}
    assert "out_of_bounds" in types


def test_task_comes_from_data_yaml_when_declared(tmp_path):
    root = seg_dataset(tmp_path, {"a": [SQUARE]}, task="segment")
    ds = Dataset(root)
    assert ds.task == SEGMENT and ds.task_source == "data.yaml task"


def test_task_is_inferred_when_not_declared(tmp_path):
    root = seg_dataset(tmp_path, {f"s{i}": [SQUARE] for i in range(6)}, task="")
    ds = Dataset(root)
    assert ds.task == SEGMENT and "inferred" in ds.task_source


def test_a_few_odd_rows_do_not_turn_a_detection_set_into_segmentation(tmp_path):
    """The majority rule protects the malformed-row defect."""
    rows = {f"d{i}": ["0 0.5 0.5 0.2 0.2"] for i in range(10)}
    rows["d0"] = ["0 0.5 0.5 0.2 0.2 0.87"]        # stray confidence column
    rows["d1"] = [SQUARE]                          # one genuine polygon-width row
    root = seg_dataset(tmp_path, rows, task="")
    ds = Dataset(root)
    assert ds.task == DETECT
    assert any("expected 5 fields" in e.reason
               for e in ds.get("train/d0").label.parse_errors)


# ------------------------------------------------------------- polygons

def test_too_few_points(tmp_path):
    root = seg_dataset(tmp_path, {"a": ["0 0.2 0.2 0.8 0.8"]})   # 2 points
    ds = Dataset(root)
    # a 5-field row in a segment dataset is not polygon-shaped at all
    assert ds.get("train/a").label.parse_errors

    root = seg_dataset(tmp_path, {"b": ["0 0.2 0.2 0.4 0.4 0.2 0.2"]})
    found = run("polygon_scan", Dataset(root))
    assert {f.type for f in found} == {"polygon_zero_area"}


def test_zero_area_from_collinear_points(tmp_path):
    root = seg_dataset(tmp_path, {"a": ["0 0.1 0.1 0.3 0.3 0.5 0.5 0.7 0.7"]})
    found = run("polygon_scan", Dataset(root))
    assert [f.type for f in found] == ["polygon_zero_area"]
    assert found[0].severity == "critical"


def test_self_intersecting_polygon(tmp_path):
    """A symmetric bow tie: signed area is exactly zero because the lobes cancel.

    This is the case that pins the check order. The honest diagnosis is the
    crossing, not the arithmetic consequence of it.
    """
    bowtie = "0 0.2 0.2 0.8 0.8 0.8 0.2 0.2 0.8"
    assert polygon_area([(0.2, 0.2), (0.8, 0.8), (0.8, 0.2),
                         (0.2, 0.8)]) == pytest.approx(0.0, abs=1e-12)
    found = run("polygon_scan", Dataset(seg_dataset(tmp_path, {"a": [bowtie]})))
    assert [f.type for f in found] == ["polygon_self_intersecting"]
    assert "even-odd" in found[0].detail


def test_self_intersecting_polygon_with_nonzero_area(tmp_path):
    """Crossing detection must not depend on the lobes cancelling."""
    skew = "0 0.1 0.1 0.9 0.9 0.9 0.1 0.3 0.85 0.15 0.5"
    found = run("polygon_scan", Dataset(seg_dataset(tmp_path, {"a": [skew]})))
    assert [f.type for f in found] == ["polygon_self_intersecting"]


def test_valid_polygon_is_silent(tmp_path):
    assert run("polygon_scan", Dataset(seg_dataset(tmp_path, {"a": [SQUARE]}))) == []


def test_polygon_scan_is_inert_on_detection_data(clean_root):
    assert run("polygon_scan", Dataset(clean_root)) == []


def test_polygon_helpers():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert polygon_area(square) == pytest.approx(1.0)
    assert polygon_area(square[::-1]) == pytest.approx(-1.0)   # winding sign
    assert self_intersections(square) == []
    assert self_intersections([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert polygon_area([(0, 0), (1, 1)]) == 0.0               # too few points


# ------------------------------------------------------------ keypoints

def pose_dataset(tmp_path: Path, rows: list[str], n_kpt: int = 2) -> Path:
    root = tmp_path / "pose"
    _image(root, "train", "a")
    (root / "labels" / "train" / "a.txt").write_text("\n".join(rows) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"names": ["person"], "nc": 1, "kpt_shape": [n_kpt, 3]}, sort_keys=False))
    return root


def test_pose_task_and_keypoint_parsing(tmp_path):
    ds = Dataset(pose_dataset(tmp_path, ["0 0.5 0.5 0.4 0.4  0.45 0.45 2  0.55 0.55 2"]))
    assert ds.task == POSE and ds.n_kpt == 2
    kp = ds.get("train/a").label.boxes[0].keypoints
    assert kp == ((0.45, 0.45, 2.0), (0.55, 0.55, 2.0))


def test_invalid_visibility_flag(tmp_path):
    root = pose_dataset(tmp_path, ["0 0.5 0.5 0.4 0.4  0.45 0.45 0.87  0.55 0.55 2"])
    found = run("keypoint_scan", Dataset(root))
    assert [f.type for f in found] == ["keypoint_visibility_invalid"]
    assert "0.87" in " ".join(found[0].evidence)


def test_keypoint_outside_its_box(tmp_path):
    root = pose_dataset(tmp_path, ["0 0.5 0.5 0.2 0.2  0.95 0.95 2  0.5 0.5 2"])
    found = run("keypoint_scan", Dataset(root))
    assert [f.type for f in found] == ["keypoint_outside_box"]


def test_unlabelled_keypoint_position_is_not_a_claim(tmp_path):
    """visibility 0 means "not labelled"; its coordinates mean nothing."""
    root = pose_dataset(tmp_path, ["0 0.5 0.5 0.2 0.2  0.99 0.99 0  0.5 0.5 2"])
    assert run("keypoint_scan", Dataset(root)) == []


def test_keypoint_scan_is_inert_on_detection_data(clean_root):
    assert run("keypoint_scan", Dataset(clean_root)) == []


def test_a_symmetric_bowtie_is_reported_as_crossing_not_as_zero_area(tmp_path):
    """The shoelace area of a bow-tie is zero because its lobes cancel.

    Checking area first would call this "zero area", which is true of the
    formula and wrong about the defect - and would send the reader looking for
    collinear points that are not there.
    """
    bowtie = "0 0.2 0.2 0.8 0.8 0.8 0.2 0.2 0.8"
    from dsdoctor.dataset import polygon_area
    pts = [(0.2, 0.2), (0.8, 0.8), (0.8, 0.2), (0.2, 0.8)]
    assert abs(polygon_area(pts)) < 1e-12          # the trap this guards

    found = run("polygon_scan", Dataset(seg_dataset(tmp_path, {"a": [bowtie]})))
    assert [f.type for f in found] == ["polygon_self_intersecting"]
