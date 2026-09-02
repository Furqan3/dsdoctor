"""The safety gate. These are the tests that matter for the ground rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from dsdoctor.apply import apply_plan, AUTOMATABLE
from dsdoctor.dataset import Dataset


def _plan(root: Path, action: str, targets: list[str], review: bool = False) -> Path:
    p = root / "plan.json"
    p.write_text(json.dumps({
        "dataset": str(root), "verdict": "blocked", "generated": "now",
        "approved": False,
        "steps": [{"finding_id": "x", "type": "degenerate_box",
                   "severity": "critical", "action": action,
                   "targets": targets, "detail": None, "why": "test",
                   "requires_human_review": review}]}))
    return p


def test_declining_the_prompt_changes_nothing(clean_root, monkeypatch):
    lbl = clean_root / "labels" / "train" / "t000.txt"
    lbl.write_text("0 0.5 0.5 0.0 0.2\n")
    before = lbl.read_text()
    plan = _plan(clean_root, "drop_degenerate_boxes", ["train/t000"])

    monkeypatch.setattr("builtins.input", lambda *_: "no")
    res = apply_plan(plan)

    assert res.get("aborted") is True
    assert res["files_changed"] == 0
    assert lbl.read_text() == before


def test_approval_applies_and_backs_up(clean_root, monkeypatch):
    lbl = clean_root / "labels" / "train" / "t000.txt"
    lbl.write_text("0 0.5 0.5 0.0 0.2\n1 0.5 0.5 0.2 0.2\n")
    plan = _plan(clean_root, "drop_degenerate_boxes", ["train/t000"])

    monkeypatch.setattr("builtins.input", lambda *_: "apply")
    res = apply_plan(plan)

    assert res["files_changed"] == 1
    assert "0.0 0.2" not in lbl.read_text()
    backup = Path(res["backup"]) / "labels" / "train" / "t000.txt"
    assert backup.is_file(), "no backup was written"
    assert "0.0 0.2" in backup.read_text()


def test_human_review_steps_are_never_applied(clean_root, monkeypatch):
    lbl = clean_root / "labels" / "train" / "t000.txt"
    lbl.write_text("0 0.5 0.5 0.0 0.2\n")
    before = lbl.read_text()
    plan = _plan(clean_root, "drop_degenerate_boxes", ["train/t000"], review=True)

    monkeypatch.setattr("builtins.input", lambda *_: "apply")
    res = apply_plan(plan)

    assert res["files_changed"] == 0
    assert lbl.read_text() == before


def test_class_remap_is_not_automatable():
    """A remap is a claim about what the images show. We have not looked."""
    assert "review_class_remap" not in AUTOMATABLE
    assert "rebalance_or_merge_classes" not in AUTOMATABLE


def test_audit_and_scan_do_not_write(clean_root):
    from dsdoctor.detectors import available, EXTRA_GROUPS
    before = {p: p.stat().st_mtime_ns for p in clean_root.rglob("*") if p.is_file()}
    ds = Dataset(clean_root)
    # Every group, not just core: an optional check is exactly the place a
    # stray write would go unnoticed.
    for det in available(groups=list(EXTRA_GROUPS)):
        det.fn(ds)
    after = {p: p.stat().st_mtime_ns for p in clean_root.rglob("*") if p.is_file()}
    assert before == after, "a detector modified the dataset"
