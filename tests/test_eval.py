"""Tests for the evaluation machinery itself.

If the injector, the ground truth or the scorer are wrong then every number in
the write-up is wrong, so these matter more than the detector tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from injector import build_case, INJECTORS, DATASET_LEVEL   # noqa: E402
from score import score, expected_verdict                   # noqa: E402

from dsdoctor.dataset import Dataset                        # noqa: E402
from dsdoctor.detectors import available                    # noqa: E402


def detect_facts(root: Path) -> set:
    ds = Dataset(root)
    out = set()
    for det in available():
        if not det.covers:
            continue
        for f in det.fn(ds):
            for k in (f.items or [DATASET_LEVEL]):
                out.add((f.type, k))
    return out


# Every injector that the eval cases actually use.
INJECTED = ["out_of_bounds", "degenerate_box", "tiny_box", "duplicate_annotation",
            "denormalised_coords", "class_id_out_of_range", "missing_label_file",
            "empty_label_file", "orphan_label_file", "malformed_label_row",
            "corrupt_image", "yaml_inconsistency", "train_val_leakage",
            "near_duplicate_image", "extreme_class_imbalance"]


@pytest.mark.parametrize("dtype", INJECTED)
def test_every_injected_defect_is_detected(clean_root, tmp_path, dtype):
    """Round trip: inject a defect, and the detector suite must find it.

    This is the assumption the headline recall number rests on - that a miss
    is the arm's fault and not a defect nothing could have caught.
    """
    out = tmp_path / f"case_{dtype}"
    gt = build_case(clean_root, out, {dtype: 2}, seed=3)
    truth = {tuple(x) for x in gt["ground_truth"]}
    assert truth, f"{dtype} injected nothing"

    found = detect_facts(out)
    missed = truth - found
    assert not missed, f"{dtype}: detectors missed {sorted(missed)}"


def test_clean_corpus_copy_stays_clean(clean_root, tmp_path):
    out = tmp_path / "noop"
    gt = build_case(clean_root, out, {}, seed=1)
    assert gt["ground_truth"] == []
    assert detect_facts(out) == set()


def test_injectors_do_not_overlap(clean_root, tmp_path):
    """Two injections must not land on the same file, or ground truth lies."""
    out = tmp_path / "multi"
    gt = build_case(clean_root, out,
                    {"out_of_bounds": 2, "degenerate_box": 2, "tiny_box": 2},
                    seed=5)
    keys = [k for _, k in gt["ground_truth"]]
    assert len(keys) == len(set(keys)), f"overlapping injections: {keys}"


def test_score_counts_hits_misses_and_false_positives():
    gt = {("out_of_bounds", "train/a"), ("tiny_box", "train/b")}
    col = {("out_of_bounds", "train/c")}          # legitimate second-order truth
    rep = {("out_of_bounds", "train/a"),           # hit
           ("out_of_bounds", "train/c"),           # collateral: neither way
           ("degenerate_box", "train/z")}          # false positive
    s = score("t", "arm", rep, gt, col)
    assert s.hits == 1 and s.misses == 1
    assert s.false_positives == 1
    assert s.recall == 0.5
    assert s.precision == 0.5                      # 1 hit / (1 hit + 1 fp)


def test_collateral_is_never_a_false_positive():
    gt = {("denormalised_coords", "train/a")}
    col = {("out_of_bounds", "train/a")}
    s = score("t", "arm", gt | col, gt, col)
    assert s.false_positives == 0
    assert s.recall == 1.0


def test_expected_verdict_follows_worst_severity():
    assert expected_verdict({("out_of_bounds", "x")}) == "blocked"
    assert expected_verdict({("tiny_box", "x")}) == "fix_before_training"
    assert expected_verdict({("empty_label_file", "x")}) == "usable_with_caveats"
    assert expected_verdict(set()) == "usable_with_caveats"


def test_all_case_recipes_reference_real_injectors():
    from cases import CASES
    for c in CASES:
        for dtype in c["recipe"]:
            assert dtype in INJECTORS, f"{c['name']} uses unknown defect {dtype}"


def test_in_scope_recall_separates_coverage_from_capability():
    """An arm that never saw a file should not be judged the same as one that
    saw it and still missed the defect."""
    gt = {("out_of_bounds", "train/seen"),
          ("tiny_box", "train/unseen"),
          ("yaml_inconsistency", "<dataset>")}
    rep = {("out_of_bounds", "train/seen")}
    s = score("c", "baseline", rep, gt, set(), scope={"train/seen"})

    assert s.recall == 1 / 3                 # of everything injected
    assert s.n_in_scope == 2                 # the seen file + the dataset-level fact
    assert s.hits_in_scope == 1
    assert s.recall_in_scope == 0.5          # it saw data.yaml's problem and missed it


def test_scope_is_optional():
    gt = {("tiny_box", "train/a")}
    s = score("c", "agent", gt, gt, set())
    assert s.recall == 1.0
    assert s.n_in_scope is None and s.recall_in_scope is None


@pytest.mark.parametrize("dtype", ["train_val_leakage", "near_duplicate_image",
                                   "orphan_label_file"])
def test_injected_filenames_do_not_reveal_the_defect(clean_root, tmp_path, dtype):
    """Regression: the injector must not encode the answer in the filename.

    An earlier version named leaked copies `<stem>_v0` and duplicates
    `<stem>_dup0`. An arm that only reads the file listing could then "detect"
    leakage by spotting a shared prefix, without ever comparing two images -
    and one did, scoring 100% on the leakage case for entirely the wrong
    reason.
    """
    out = tmp_path / f"named_{dtype}"
    gt = build_case(clean_root, out, {dtype: 2}, seed=11)
    injected = [k for _, k in gt["ground_truth"]]
    assert injected

    for key in injected:
        stem = key.split("/", 1)[1]
        for tell in ("_v", "_dup", "orphan", "copy", "leak"):
            assert tell not in stem, f"{stem!r} advertises the defect"

    # New files must be indistinguishable in form from the originals.
    originals = {p.stem for p in (clean_root / "images" / "train").iterdir()}
    new = [k.split("/", 1)[1] for k in injected
           if k.split("/", 1)[1] not in originals]
    for stem in new:
        assert stem.isdigit() and len(stem) == 12, \
            f"{stem!r} does not look like an ordinary dataset id"


def test_leakage_is_not_detectable_from_the_file_listing_alone(clean_root, tmp_path):
    """The leaked val file must share no name with the train file it copies."""
    out = tmp_path / "leak_names"
    gt = build_case(clean_root, out, {"train_val_leakage": 3}, seed=12)
    keys = [k for t, k in gt["ground_truth"] if t == "train_val_leakage"]
    train_stems = {k.split("/", 1)[1] for k in keys if k.startswith("train/")}
    val_stems = {k.split("/", 1)[1] for k in keys if k.startswith("val/")}
    assert train_stems and val_stems
    assert not (train_stems & val_stems), "leaked pair shares a filename"
    for v in val_stems:
        for t in train_stems:
            assert t not in v and v not in t, f"{v!r} embeds {t!r}"
