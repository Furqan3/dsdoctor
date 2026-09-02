"""The CLI surface, its exit codes, and the new apply action.

Exit codes are the part of a CLI that other programs depend on, so they are
asserted rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from dsdoctor.cli import (main, EXIT_OK, EXIT_GATE, EXIT_ERROR,
                          EXIT_MISMATCH, EXIT_DEGRADED)
from dsdoctor.dataset import Dataset


def test_scan_exits_zero_on_a_clean_dataset(clean_root, capsys):
    assert main(["scan", str(clean_root)]) == EXIT_OK


def test_scan_gate_trips_on_a_critical(clean_root):
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.9 0.5 0.6 0.2\n")
    assert main(["scan", str(clean_root), "--fail-on", "critical"]) == EXIT_GATE


def test_scan_gate_ignores_governance(clean_root):
    """A licensing question must never turn a build red on its own."""
    assert main(["scan", str(clean_root), "--checks", "privacy",
                 "--fail-on", "any"]) == EXIT_OK


def test_scan_json_is_machine_readable(clean_root, capsys):
    assert main(["scan", str(clean_root), "--format", "json"]) == EXIT_OK
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "dsdoctor/scan/1"


def test_scan_sarif_is_machine_readable(clean_root, capsys):
    assert main(["scan", str(clean_root), "--format", "sarif"]) == EXIT_OK
    doc = json.loads(capsys.readouterr().out)
    assert doc["version"] == "2.1.0"


def test_bad_check_group_is_a_usage_error(clean_root, capsys):
    assert main(["scan", str(clean_root), "--checks", "nope"]) == EXIT_ERROR
    assert "unknown check group" in capsys.readouterr().err


def test_scan_writes_html(clean_root, tmp_path):
    out = tmp_path / "r.html"
    assert main(["scan", str(clean_root), "--html", str(out)]) == EXIT_OK
    text = out.read_text()
    assert text.startswith("<!doctype html>")
    assert "Trainability audit" in text


def test_card_then_verify_then_mismatch(clean_root, tmp_path):
    assert main(["card", str(clean_root)]) == EXIT_OK
    assert main(["verify-card", str(clean_root)]) == EXIT_OK

    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text(p.read_text() + "0 0.5 0.5 0.1 0.1\n")
    assert main(["verify-card", str(clean_root)]) == EXIT_MISMATCH


def test_verify_card_without_a_card_is_an_error(clean_root):
    assert main(["verify-card", str(clean_root)]) == EXIT_ERROR


def test_recheck_reports_a_regression(clean_root, capsys):
    assert main(["card", str(clean_root)]) == EXIT_OK
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.9 0.5 0.6 0.2\n")            # introduce a critical
    code = main(["recheck", str(clean_root)])
    out = capsys.readouterr().out
    assert code == EXIT_GATE
    assert "introduced: 1" in out
    assert "out_of_bounds" in out


def test_recheck_reports_a_fix(clean_root, capsys):
    p = clean_root / "labels" / "train" / "t000.txt"
    original = p.read_text()
    p.write_text("0 0.9 0.5 0.6 0.2\n")
    assert main(["card", str(clean_root)]) == EXIT_OK
    p.write_text(original)                          # repair it
    assert main(["recheck", str(clean_root)]) == EXIT_OK
    assert "resolved:   1" in capsys.readouterr().out


def test_audit_without_an_endpoint_degrades_to_a_scan(clean_root, capsys):
    code = main(["audit", str(clean_root), "--base-url", "http://127.0.0.1:9/v1"])
    captured = capsys.readouterr()
    assert code == EXIT_DEGRADED
    assert "could not reach" in captured.err
    assert "structure_scan" in captured.out          # the scan actually ran


def test_diff_reports_a_difference(clean_root, tmp_path, capsys):
    import shutil
    other = tmp_path / "other"
    shutil.copytree(clean_root, other)
    (other / "labels" / "train" / "t000.txt").write_text("0 0.9 0.5 0.6 0.2\n")

    assert main(["diff", str(clean_root), str(other)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "only in B: out_of_bounds" in out


def test_resplit_writes_a_verified_split(clean_root, tmp_path, capsys):
    out = tmp_path / "split"
    assert main(["resplit", str(clean_root), "--out", str(out)]) == EXIT_OK
    assert "leak_free=True" in capsys.readouterr().out
    assert (out / "data.yaml").is_file()


def test_detectors_command_lists_groups(capsys):
    assert main(["detectors"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "privacy_scan" in out and "--checks privacy" in out


# ------------------------------------------------------- lossless exif strip

def test_strip_exif_is_lossless_and_idempotent(clean_root):
    from dsdoctor.apply import strip_exif_jpeg
    from dsdoctor.detectors.metadata import _orientation
    from dsdoctor.detectors.privacy import _has_gps

    path = clean_root / "images" / "train" / "t000.jpg"
    im = Image.open(path)
    exif = Image.Exif()
    exif[274] = 6
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (51.0, 30.0, 0.0)})
    im.save(path, "JPEG", exif=exif, quality=95)

    before = Image.open(path).convert("RGB").tobytes()
    assert _orientation(path) == 6 and _has_gps(path) is True

    assert strip_exif_jpeg(path) is True
    assert Image.open(path).convert("RGB").tobytes() == before
    assert _orientation(path) is None
    assert _has_gps(path) is False
    assert strip_exif_jpeg(path) is False           # nothing left to remove


def test_strip_exif_leaves_non_jpeg_alone(tmp_path):
    from dsdoctor.apply import strip_exif_jpeg
    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(p)
    before = p.read_bytes()
    assert strip_exif_jpeg(p) is False
    assert p.read_bytes() == before


def test_delegated_actions_are_explained_not_applied(tmp_path):
    from dsdoctor.apply import summarise, AUTOMATABLE, DELEGATED
    plan = {"dataset": str(tmp_path), "verdict": "blocked", "generated": "now",
            "steps": [{"action": "resplit_removing_leaks", "type":
                       "train_val_leakage", "targets": ["val/x"],
                       "why": "leak", "severity": "critical"}]}
    text = summarise(plan)
    assert "MANUAL" in text
    assert "dsdoctor resplit" in text
    assert "resplit_removing_leaks" not in AUTOMATABLE
    assert "bake_exif_orientation" in DELEGATED
