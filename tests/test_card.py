"""The health card, its fingerprint, and the recheck loop.

The card's whole value is that a receiver can tell whether it describes the
data in front of them. So the tests are mostly about the fingerprint being
sensitive to the things that make a dataset different, and insensitive to the
act of writing the card itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsdoctor import card
from dsdoctor.dataset import Dataset
from dsdoctor.findings import Finding, CRITICAL, MAJOR
from dsdoctor.sweep import sweep


def _write(root: Path, **kw):
    ds = Dataset(root)
    res = sweep(ds, **kw)
    return ds, card.write(ds, res.findings, root, groups=kw.get("groups"),
                          detectors_run=res.detectors_run)


def test_card_writes_all_three_artefacts(clean_root):
    ds, paths = _write(clean_root)
    assert set(paths) == {"health", "card", "manifest"}
    for p in paths.values():
        assert p.is_file() and p.stat().st_size > 0
    health = json.loads(paths["health"].read_text())
    assert health["schema"] == card.CARD_SCHEMA
    assert health["verdict"] == "usable_with_caveats"
    assert health["fingerprint"]["files"] > 0


def test_writing_the_card_does_not_change_the_fingerprint(clean_root):
    """The card lives in the directory it describes; that must be stable."""
    ds, paths = _write(clean_root)
    first = json.loads(paths["health"].read_text())["fingerprint"]["digest"]
    ds2, paths2 = _write(clean_root)
    second = json.loads(paths2["health"].read_text())["fingerprint"]["digest"]
    assert first == second


def test_verify_matches_an_untouched_dataset(clean_root):
    ds, paths = _write(clean_root)
    health = json.loads(paths["health"].read_text())
    result = card.verify(Dataset(clean_root), health, paths["manifest"])
    assert result["match"] is True


def test_verify_detects_a_modified_label(clean_root):
    ds, paths = _write(clean_root)
    health = json.loads(paths["health"].read_text())
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text(p.read_text() + "0 0.5 0.5 0.1 0.1\n")

    result = card.verify(Dataset(clean_root), health, paths["manifest"])
    assert result["match"] is False
    assert result["modified"] == ["labels/train/t000.txt"]
    assert result["added"] == [] and result["removed"] == []


def test_verify_detects_a_removed_file(clean_root):
    ds, paths = _write(clean_root)
    health = json.loads(paths["health"].read_text())
    (clean_root / "images" / "val" / "v000.jpg").unlink()

    result = card.verify(Dataset(clean_root), health, paths["manifest"])
    assert result["match"] is False
    assert result["removed"] == ["images/val/v000.jpg"]


def test_fingerprint_detects_a_reshuffled_split(clean_root, tmp_path):
    """Same bytes, different split assignment, must be a different dataset."""
    rows = card.manifest(clean_root)
    before = card.digest(rows)

    src = clean_root / "images" / "train" / "t000.jpg"
    src.rename(clean_root / "images" / "val" / "t000.jpg")
    after = card.digest(card.manifest(clean_root))
    assert before != after


def test_deterministic_verdict_rule():
    crit = Finding(type="out_of_bounds", title="t", detail="d", detector="x")
    major = Finding(type="tiny_box", title="t", detail="d", detector="x")
    minor = Finding(type="empty_label_file", title="t", detail="d", detector="x")
    assert card.deterministic_verdict([]) == "usable_with_caveats"
    assert card.deterministic_verdict([minor]) == "usable_with_caveats"
    assert card.deterministic_verdict([major]) == "fix_before_training"
    assert card.deterministic_verdict([major, crit]) == "blocked"


def test_card_markdown_separates_governance(clean_root):
    ds, paths = _write(clean_root, groups=["privacy"])
    text = paths["card"].read_text()
    assert "## Trainability findings" in text
    assert "## Governance and privacy" in text
    assert "missing_license" in text
    # the governance finding must not be counted as a trainability defect
    health = json.loads(paths["health"].read_text())
    assert health["summary"]["governance"] == 1
    assert health["verdict"] == "usable_with_caveats"


def test_card_records_which_checks_were_run(clean_root):
    ds, paths = _write(clean_root, groups=["privacy"])
    health = json.loads(paths["health"].read_text())
    assert health["checks"]["groups"] == ["core", "privacy"]
    assert "privacy_scan" in health["checks"]["detectors_run"]
