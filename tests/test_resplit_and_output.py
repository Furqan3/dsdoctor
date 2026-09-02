"""Leak-free re-splitting, CI output formats, and the scaled duplicate scan."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest
import yaml

from conftest import write_sample
from dsdoctor import output, resplit
from dsdoctor.dataset import Dataset
from dsdoctor.detectors.duplicates import (_candidate_pairs, _hamming,
                                           brute_force_pairs, duplicate_scan,
                                           NEAR_DUP_DISTANCE, LSH_BANDS)
from dsdoctor.sweep import sweep, should_fail


# ------------------------------------------------------- duplicate scaling

def test_lsh_banding_is_exact_not_approximate():
    """More bands than the distance threshold means no pair can be missed."""
    assert LSH_BANDS > NEAR_DUP_DISTANCE
    rng = random.Random(11)
    for _ in range(15):
        hs = [rng.getrandbits(64) for _ in range(200)]
        for _ in range(25):                       # plant near-duplicates
            h = hs[rng.randrange(len(hs))]
            for _ in range(rng.randint(0, NEAR_DUP_DISTANCE)):
                h ^= 1 << rng.randrange(64)
            hs.append(h)
        cand = _candidate_pairs(list(enumerate(hs)))
        got = {(i, j) for i, j in cand
               if _hamming(hs[i], hs[j]) <= NEAR_DUP_DISTANCE}
        assert got == brute_force_pairs(hs)


def test_lsh_reduces_the_comparison_count():
    rng = random.Random(3)
    hs = [rng.getrandbits(64) for _ in range(400)]
    exhaustive = len(hs) * (len(hs) - 1) // 2
    assert len(_candidate_pairs(list(enumerate(hs)))) < exhaustive / 10


@pytest.fixture
def leaky_root(clean_root: Path) -> Path:
    """Copy two training images into val: textbook leakage."""
    for stem in ("t000", "t001"):
        shutil.copy2(clean_root / "images" / "train" / f"{stem}.jpg",
                     clean_root / "images" / "val" / f"{stem}.jpg")
        shutil.copy2(clean_root / "labels" / "train" / f"{stem}.txt",
                     clean_root / "labels" / "val" / f"{stem}.txt")
    return clean_root


def test_leakage_is_detected(leaky_root):
    found = [f for f in duplicate_scan(Dataset(leaky_root))
             if f.type == "train_val_leakage"]
    assert len(found) == 1
    assert found[0].n_items == 4          # both sides of both pairs


# ------------------------------------------------------------- re-splitting

def test_resplit_removes_the_leak(leaky_root, tmp_path):
    ds = Dataset(leaky_root)
    proposal = resplit.propose(ds, val_fraction=0.25)
    out = tmp_path / "resplit"
    resplit.materialise(ds, proposal, out)

    check = resplit.verify(out)
    assert check["leak_free"] is True
    assert check["leaked_pairs"] == 0


def test_resplit_keeps_duplicate_groups_together(leaky_root):
    ds = Dataset(leaky_root)
    proposal = resplit.propose(ds)
    side = {}
    for split, keys in proposal["assignment"].items():
        for k in keys:
            side[k.split("/", 1)[1]] = split
    # t000 exists in both train/ and val/; both copies must land on one side
    assert side.get("t000") is not None


def test_resplit_loses_no_images(leaky_root):
    ds = Dataset(leaky_root)
    proposal = resplit.propose(ds)
    placed = sum(len(v) for v in proposal["assignment"].values())
    assert placed == len(ds.samples)


def test_resplit_is_deterministic(leaky_root):
    ds = Dataset(leaky_root)
    a = resplit.propose(ds)["assignment"]
    b = resplit.propose(Dataset(leaky_root))["assignment"]
    assert a == b


def test_resplit_never_touches_the_source(leaky_root, tmp_path):
    before = {p: p.stat().st_mtime_ns
              for p in leaky_root.rglob("*") if p.is_file()}
    ds = Dataset(leaky_root)
    resplit.materialise(ds, resplit.propose(ds), tmp_path / "out")
    after = {p: p.stat().st_mtime_ns
             for p in leaky_root.rglob("*") if p.is_file()}
    assert before == after


def test_resplit_covers_every_class_in_val(clean_root):
    proposal = resplit.propose(Dataset(clean_root), val_fraction=0.2)
    assert proposal["classes_missing_from_val"] == []


# ------------------------------------------------------------ CI output

def test_json_output_is_valid_and_complete(clean_root):
    ds = Dataset(clean_root)
    res = sweep(ds, groups=["privacy"])
    doc = json.loads(output.to_json(ds, res, elapsed=1.25))
    assert doc["schema"] == "dsdoctor/scan/1"
    assert doc["checks"]["groups"] == ["core", "privacy"]
    assert doc["summary"]["governance"] == 1
    assert {f["type"] for f in doc["findings"]} == {"missing_license"}


def test_sarif_output_is_wellformed(clean_root):
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.9 0.5 0.6 0.2\n")           # out of bounds
    ds = Dataset(clean_root)
    doc = json.loads(output.to_sarif(ds, sweep(ds)))

    assert doc["version"] == "2.1.0"
    run = doc["runs"][0]
    assert run["tool"]["driver"]["name"] == "dsdoctor"
    rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert "out_of_bounds" in rule_ids
    for result in run["results"]:
        assert result["ruleId"] in rule_ids
        assert result["level"] in ("error", "warning", "note")
        assert result["locations"], "every result needs a location"
        for loc in result["locations"]:
            assert loc["physicalLocation"]["artifactLocation"]["uri"]


def test_sarif_severity_mapping_keeps_critical_distinct(clean_root):
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.9 0.5 0.6 0.2\n0 0.5 0.5 0.002 0.002\n")
    ds = Dataset(clean_root)
    doc = json.loads(output.to_sarif(ds, sweep(ds)))
    by_rule = {r["ruleId"]: r["level"] for r in doc["runs"][0]["results"]}
    assert by_rule["out_of_bounds"] == "error"      # critical
    assert by_rule["tiny_box"] == "warning"         # major


def test_fail_on_thresholds(clean_root):
    ds = Dataset(clean_root)
    assert should_fail(sweep(ds), "critical") is False

    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.5 0.5 0.002 0.002\n")        # a major, not a critical
    res = sweep(Dataset(clean_root))
    assert should_fail(res, "critical") is False
    assert should_fail(res, "major") is True
    assert should_fail(res, None) is False
    with pytest.raises(ValueError):
        should_fail(res, "nonsense")


def test_sweep_survives_a_detector_that_raises(clean_root, monkeypatch):
    from dsdoctor import detectors

    def boom(ds):
        raise RuntimeError("plugin exploded")

    monkeypatch.setitem(detectors.REGISTRY, "exploding_scan",
                        detectors.Detector(name="exploding_scan",
                                           description="", fn=boom,
                                           covers=("out_of_bounds",)))
    res = sweep(Dataset(clean_root))
    assert "exploding_scan" in res.failed
    assert "plugin exploded" in res.failed["exploding_scan"]
    assert "structure_scan" in res.detectors_run     # the rest still ran
