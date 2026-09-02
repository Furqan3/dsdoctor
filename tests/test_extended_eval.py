"""Round-trip the new injectors against the detectors that should catch them.

`eval/run_extended.py` measures this properly, on the 600-image corpus. These
tests reproduce the same property offline in milliseconds, using the synthetic
fixture as the base corpus, so an injector that silently stops injecting - or
a detector that silently stops detecting - fails here rather than in a run
nobody has done for a month.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from cases import CASES, EXTENDED_CASES, ALL_CASES, by_name   # noqa: E402
from injector import INJECTORS, build_case, DATASET_LEVEL     # noqa: E402

from dsdoctor.dataset import Dataset                          # noqa: E402
from dsdoctor.sweep import sweep                              # noqa: E402


def _facts(root: Path, groups, imgsz=640):
    ds = Dataset(root)
    ds.imgsz = imgsz
    res = sweep(ds, groups=groups)
    return {(f.type, k) for f in res.findings for k in (f.items or [DATASET_LEVEL])}


@pytest.mark.parametrize("case", EXTENDED_CASES, ids=lambda c: c["name"])
def test_extended_case_defects_are_all_found(case, clean_root, tmp_path):
    out = tmp_path / case["name"]
    gt = build_case(clean_root, out, case["recipe"], seed=case["seed"])
    truth = {(t, k) for t, k in gt["ground_truth"] if t in case["recipe"]}
    assert truth, f"{case['name']} injected nothing"

    got = _facts(out, case["groups"])
    missed = truth - got
    assert not missed, f"{case['name']} missed {sorted(missed)[:5]}"


@pytest.mark.parametrize("name", ["exif_orientation", "gps_metadata",
                                  "class_absent_from_val",
                                  "undetectable_at_imgsz",
                                  "template_annotation"])
def test_new_injectors_are_registered(name):
    assert name in INJECTORS


def test_extended_cases_are_kept_out_of_the_headline_twelve():
    """The published table describes CASES. It must keep describing CASES."""
    assert len(CASES) == 12
    names = {c["name"] for c in CASES}
    assert not names & {c["name"] for c in EXTENDED_CASES}
    assert len(ALL_CASES) == len(CASES) + len(EXTENDED_CASES)


def test_every_extended_case_names_the_groups_it_needs():
    from dsdoctor.detectors import EXTRA_GROUPS
    for case in EXTENDED_CASES:
        assert case["groups"], f"{case['name']} declares no check group"
        for g in case["groups"]:
            assert g in EXTRA_GROUPS, f"{case['name']} wants unknown group {g}"


def test_by_name_reaches_both_suites():
    assert by_name("everything")["name"] == "everything"
    assert by_name("camera_metadata")["groups"] == ["metadata", "privacy"]
    with pytest.raises(KeyError):
        by_name("no_such_case")


def test_undetectable_injection_is_above_the_tiny_box_threshold():
    """The two checks must stay distinguishable, or the case scores the wrong one."""
    from dsdoctor.detectors.geometry import TINY_SIDE
    from dsdoctor.detectors.training import FINEST_STRIDE, DEFAULT_IMGSZ
    injected_side = 0.008
    assert injected_side > TINY_SIDE, "would be reported as tiny_box instead"
    assert injected_side < FINEST_STRIDE / DEFAULT_IMGSZ, "would not be reported"
