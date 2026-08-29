"""The baseline's answer must be scored on what it said, not on whether the
model happened to close its brackets.

Every one of these cases was observed in a real run. Getting them wrong meant
reporting a baseline score of zero that the baseline had not earned.
"""

from __future__ import annotations

from dsdoctor.baseline import _parse, _salvage, baseline_facts


def test_well_formed_answer_parses():
    txt = ('{"verdict":"blocked","headline":"h","findings":['
           '{"type":"tiny_box","severity":"major","files":["train/a"],'
           '"rationale":"r"}]}')
    out = _parse(txt, None, quiet=True)
    assert out and out["verdict"] == "blocked"
    assert baseline_facts(out) == {("tiny_box", "train/a")}


def test_fenced_answer_parses():
    txt = ('Here you go:\n```json\n{"verdict":"blocked","headline":"h",'
           '"findings":[{"type":"tiny_box","files":["train/a"],'
           '"rationale":"r"}]}\n```')
    assert _parse(txt, None, quiet=True) is not None


def test_truncated_between_findings_keeps_the_complete_ones():
    txt = ('{"verdict":"blocked","headline":"h","findings":['
           '{"type":"tiny_box","severity":"major","files":["train/a","train/b"],'
           '"rationale":"r"},'
           '{"type":"out_of_bounds","severity":"critical","files":["train/c"')
    out = _salvage(txt)
    assert out is not None
    types = {f["type"] for f in out["findings"]}
    assert "tiny_box" in types and "out_of_bounds" in types
    facts = baseline_facts(out)
    assert ("tiny_box", "train/a") in facts
    assert ("out_of_bounds", "train/c") in facts


def test_truncated_inside_the_first_finding_still_recovers_it():
    """The observed failure: one defect with a very long file list, cut off
    mid-filename. Recovering only complete objects would discard everything."""
    txt = ('{"verdict":"fix_before_training","headline":"h","findings":['
           '{"type":"out_of_bounds","severity":"critical","files":['
           '"train/a","train/b","val/c","val/par')
    out = _salvage(txt)
    assert out is not None
    assert len(out["findings"]) == 1
    assert out["findings"][0]["type"] == "out_of_bounds"
    facts = baseline_facts(out)
    assert {("out_of_bounds", "train/a"), ("out_of_bounds", "train/b"),
            ("out_of_bounds", "val/c")} <= facts
    assert out["verdict"] == "fix_before_training"


def test_nothing_to_salvage_returns_none():
    assert _salvage("") is None
    assert _salvage("I could not analyse this dataset.") is None
    assert _salvage('{"verdict":"blocked"}') is None


def test_salvage_is_not_confused_by_braces_inside_strings():
    txt = ('{"findings":[{"type":"tiny_box","files":["train/a"],'
           '"rationale":"a } brace and a \\" quote"}]}')
    out = _salvage(txt)
    assert out is not None and len(out["findings"]) == 1
    assert out["findings"][0]["files"] == ["train/a"]
