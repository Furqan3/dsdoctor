"""The split, metadata and privacy detectors.

Each test builds the defect it is testing rather than asserting against a
fixture, so a detector that stops working fails here rather than quietly
returning nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from PIL import Image

from conftest import write_sample, NAMES
from dsdoctor.dataset import Dataset
from dsdoctor.detectors import run


# ------------------------------------------------------------------ split

def test_class_absent_from_val_is_reported(clean_root):
    """Strip one class out of val entirely; its AP becomes undefined."""
    for p in (clean_root / "labels" / "val").iterdir():
        kept = [ln for ln in p.read_text().splitlines() if not ln.startswith("2 ")]
        p.write_text("\n".join(kept) + "\n")

    found = run("split_scan", Dataset(clean_root))
    absent = [f for f in found if f.type == "class_absent_from_val"]
    assert len(absent) == 1
    assert "cup" in " ".join(absent[0].evidence)


def test_split_scan_is_silent_on_a_healthy_split(clean_root):
    assert run("split_scan", Dataset(clean_root)) == []


def test_split_ratio_extreme_flags_a_tiny_val(tmp_path):
    root = tmp_path / "ds"
    for i in range(40):
        write_sample(root, "train", f"t{i:03d}", ["0 0.5 0.5 0.2 0.2"], seed=i)
    write_sample(root, "val", "v000", ["0 0.5 0.5 0.2 0.2"], seed=999)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["a"], "nc": 1}))

    types = {f.type for f in run("split_scan", Dataset(root))}
    assert "split_ratio_extreme" in types


def test_split_ratio_ignores_datasets_too_small_to_judge(tmp_path):
    root = tmp_path / "ds"
    for i in range(4):
        write_sample(root, "train", f"t{i}", ["0 0.5 0.5 0.2 0.2"], seed=i)
    write_sample(root, "val", "v0", ["0 0.5 0.5 0.2 0.2"], seed=9)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["a"], "nc": 1}))
    assert not [f for f in run("split_scan", Dataset(root))
                if f.type == "split_ratio_extreme"]


# --------------------------------------------------------------- metadata

def _write_with_exif(path, exif, size=(64, 48), seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "JPEG", exif=exif, quality=95)


def test_exif_orientation_is_detected(clean_root):
    exif = Image.Exif()
    exif[274] = 6                                   # rotate 90 CW
    _write_with_exif(clean_root / "images" / "train" / "t000.jpg", exif)

    found = run("exif_orientation_scan", Dataset(clean_root))
    assert len(found) == 1
    assert found[0].type == "exif_orientation"
    assert found[0].items == ["train/t000"]
    assert "swaps width and height" in found[0].detail


def test_orientation_one_is_not_a_finding(clean_root):
    exif = Image.Exif()
    exif[274] = 1                                   # "as stored" - not a defect
    _write_with_exif(clean_root / "images" / "train" / "t000.jpg", exif)
    assert run("exif_orientation_scan", Dataset(clean_root)) == []


def test_no_exif_means_no_finding(clean_root):
    assert run("exif_orientation_scan", Dataset(clean_root)) == []


# ---------------------------------------------------------------- privacy

def test_gps_metadata_is_detected(clean_root):
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (51.0, 30.0, 0.0), 3: "W", 4: (0.0, 7.0, 0.0)})
    _write_with_exif(clean_root / "images" / "train" / "t000.jpg", exif)

    found = [f for f in run("privacy_scan", Dataset(clean_root))
             if f.type == "gps_metadata"]
    assert len(found) == 1
    assert found[0].items == ["train/t000"]
    assert found[0].category == "governance"


def test_gps_is_not_claimed_without_coordinates(clean_root):
    exif = Image.Exif()
    exif[274] = 1
    _write_with_exif(clean_root / "images" / "train" / "t000.jpg", exif)
    assert not [f for f in run("privacy_scan", Dataset(clean_root))
                if f.type == "gps_metadata"]


def test_missing_license_is_reported_then_satisfied(clean_root):
    ds = Dataset(clean_root)
    assert [f for f in run("privacy_scan", ds) if f.type == "missing_license"]

    (clean_root / "LICENSE").write_text("CC BY 4.0")
    assert not [f for f in run("privacy_scan", Dataset(clean_root))
                if f.type == "missing_license"]


def test_representation_skew_abstains_on_uniform_capture(clean_root):
    """One resolution everywhere: there is nothing to compare against."""
    assert run("representation_scan", Dataset(clean_root)) == []


def test_representation_skew_flags_a_single_source_class(tmp_path):
    root = tmp_path / "ds"
    # class 0 photographed only at 128x96; the rest of the set is 64x48.
    for i in range(30):
        write_sample(root, "train", f"a{i:03d}", ["1 0.5 0.5 0.2 0.2"],
                     size=(64, 48), seed=i)
    for i in range(10):
        write_sample(root, "train", f"b{i:03d}",
                     [f"0 0.5 0.5 0.2 0.2"] * 3, size=(128, 96), seed=500 + i)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": ["rare", "common"], "nc": 2}))

    found = run("representation_scan", Dataset(root))
    assert len(found) == 1
    assert found[0].type == "representation_skew"
    assert "rare" in " ".join(found[0].evidence)
    assert found[0].category == "governance"
