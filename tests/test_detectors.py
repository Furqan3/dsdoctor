"""Each detector fires on the defect it owns, and stays quiet otherwise.

The second half of that sentence is the part worth testing. A detector that
reports something on a healthy dataset is how a reviewer learns to ignore the
tool.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from dsdoctor.dataset import Dataset
from dsdoctor.detectors import REGISTRY
from conftest import write_sample


def sweep(root: Path):
    ds = Dataset(root)
    out = []
    for det in REGISTRY.values():
        if det.experimental or not det.covers:
            continue
        out.extend(det.fn(ds))
    return out


def types_found(root: Path) -> set[str]:
    return {f.type for f in sweep(root)}


def test_clean_dataset_produces_no_findings(clean_root):
    found = sweep(clean_root)
    assert found == [], f"detectors fired on a clean dataset: {[f.type for f in found]}"


def test_out_of_bounds(clean_root):
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.9 0.5 0.40 0.20\n")
    assert "out_of_bounds" in types_found(clean_root)


def test_degenerate_box(clean_root):
    (clean_root / "labels" / "train" / "t001.txt").write_text("1 0.5 0.5 0.0 0.2\n")
    assert "degenerate_box" in types_found(clean_root)


def test_tiny_box(clean_root):
    (clean_root / "labels" / "train" / "t002.txt").write_text(
        "1 0.5 0.5 0.0005 0.0005\n")
    assert "tiny_box" in types_found(clean_root)


def test_duplicate_annotation(clean_root):
    (clean_root / "labels" / "train" / "t003.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    assert "duplicate_annotation" in types_found(clean_root)


def test_malformed_row(clean_root):
    (clean_root / "labels" / "train" / "t004.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.97\n")
    assert "malformed_label_row" in types_found(clean_root)


def test_class_id_out_of_range(clean_root):
    (clean_root / "labels" / "train" / "t005.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    assert "class_id_out_of_range" in types_found(clean_root)


def test_denormalised_coords(clean_root):
    # image is 64x48; pixel coordinates should be recognised as such
    (clean_root / "labels" / "train" / "t006.txt").write_text(
        "0 32.0 24.0 16.0 12.0\n0 20.0 20.0 8.0 8.0\n")
    assert "denormalised_coords" in types_found(clean_root)


def test_missing_and_orphan_and_empty(clean_root):
    (clean_root / "labels" / "train" / "t007.txt").unlink()
    (clean_root / "labels" / "train" / "orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n")
    (clean_root / "labels" / "train" / "t008.txt").write_text("")
    found = types_found(clean_root)
    assert {"missing_label_file", "orphan_label_file", "empty_label_file"} <= found


def test_corrupt_image_needs_a_full_decode(clean_root):
    """Regression: Image.verify() passes a JPEG truncated mid-scan."""
    p = clean_root / "images" / "train" / "t009.jpg"
    data = p.read_bytes()
    p.write_bytes(data[:len(data) // 3])
    assert "corrupt_image" in types_found(clean_root)


def test_train_val_leakage(clean_root):
    src = clean_root / "images" / "train" / "t000.jpg"
    shutil.copy2(src, clean_root / "images" / "val" / "leaked.jpg")
    shutil.copy2(clean_root / "labels" / "train" / "t000.txt",
                 clean_root / "labels" / "val" / "leaked.txt")
    assert "train_val_leakage" in types_found(clean_root)


def test_near_duplicate_inside_a_split(clean_root):
    src = clean_root / "images" / "train" / "t000.jpg"
    shutil.copy2(src, clean_root / "images" / "train" / "t000_copy.jpg")
    shutil.copy2(clean_root / "labels" / "train" / "t000.txt",
                 clean_root / "labels" / "train" / "t000_copy.txt")
    assert "near_duplicate_image" in types_found(clean_root)


def test_yaml_inconsistency(clean_root):
    (clean_root / "data.yaml").write_text(
        yaml.safe_dump({"names": ["person", "car", "cup"], "nc": 7}))
    assert "yaml_inconsistency" in types_found(clean_root)


def test_no_detector_imports_an_llm():
    """The central claim of the design, enforced."""
    src = Path(__file__).resolve().parents[1] / "src" / "dsdoctor" / "detectors"
    for py in src.rglob("*.py"):
        text = py.read_text()
        assert "from ..llm" not in text and "import openai" not in text, \
            f"{py.name} reaches for a language model"


def test_finding_severity_defaults_to_the_shared_table():
    """A Finding built without an explicit severity must not silently be major."""
    from dsdoctor.findings import Finding, DEFECT_TYPES, CRITICAL

    f = Finding(type="corrupt_image", title="t", detail="d", detector="x")
    assert f.severity == CRITICAL, "critical defect defaulted to the wrong severity"

    for dtype, (sev, _) in DEFECT_TYPES.items():
        g = Finding(type=dtype, title="t", detail="d", detector="x")
        assert g.severity == sev, f"{dtype} defaulted to {g.severity}, expected {sev}"

    h = Finding(type="out_of_bounds", title="t", detail="d", detector="x",
                severity="minor")
    assert h.severity == "minor", "an explicit severity must win"
