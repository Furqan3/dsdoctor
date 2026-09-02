"""The check-group mechanism, and the guarantee it exists to provide.

The published results table describes one specific set of detectors. Anything
added afterwards has to be opt-in until it has been through the same twelve
cases, or the numbers stop describing the tool. These tests hold that line.
"""

from __future__ import annotations

import pytest

from dsdoctor import detectors
from dsdoctor.dataset import Dataset
from dsdoctor.findings import Finding, GOVERNANCE, TRAINABILITY

# The exact set the README's measurements were produced with.
MEASURED_CORE = {
    "structure_scan", "image_integrity_scan", "geometry_scan",
    "normalisation_scan", "class_scan", "class_distribution", "duplicate_scan",
}

# Core detectors added after those measurements. A detector may only be listed
# here if it is *structurally* incapable of firing on a detection dataset -
# `polygon_scan` and `keypoint_scan` return immediately unless the dataset's
# task is segment or pose, and the evaluation corpus is detection throughout.
# That is what makes them safe in `core` rather than an opt-in group, and
# `test_inert_core_detectors_really_are_inert` below is the evidence rather
# than the claim. Anything that can produce a finding on a detection dataset
# belongs in an opt-in group until it has been through the twelve cases.
INERT_ON_DETECTION = {"polygon_scan", "keypoint_scan"}


def test_default_detector_set_is_unchanged():
    """The scored set may not grow by anything that can score."""
    assert {d.name for d in detectors.available()} == (
        MEASURED_CORE | INERT_ON_DETECTION)


def test_inert_core_detectors_really_are_inert(clean_root):
    """Every core detector outside the measured set must find nothing here.

    `clean_root` is a detection dataset. If one of these ever returns a
    finding on one, it can move the published numbers and it does not belong
    in `core` - move it to an opt-in group and evaluate it.
    """
    ds = Dataset(clean_root)
    assert ds.task == "detect"
    for name in INERT_ON_DETECTION:
        assert detectors.run(name, ds) == [], (
            f"{name} is in core but fires on a detection dataset")


def test_inert_core_detectors_are_inert_on_defective_detection_data(clean_root):
    """Not just on clean data - on data full of the defects they might mimic."""
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.9 0.5 0.6 0.2\n"          # out of bounds
                 "1 0.5 0.5 0.0 0.0\n"          # degenerate
                 "2 0.5 0.5 0.2 0.2 0.87\n"     # a stray confidence column
                 "0 0.1 0.1 0.05 0.05\n")
    ds = Dataset(clean_root)
    assert ds.task == "detect", "a malformed detection row must not read as a polygon"
    for name in INERT_ON_DETECTION:
        assert detectors.run(name, ds) == []


def test_optional_groups_are_excluded_by_default():
    default = {d.name for d in detectors.available()}
    for name, det in detectors.REGISTRY.items():
        if det.group != detectors.CORE:
            assert name not in default, f"{name} leaked into the default set"


def test_groups_can_be_enabled():
    with_privacy = {d.name for d in detectors.available(groups=["privacy"])}
    assert "privacy_scan" in with_privacy
    assert "split_scan" not in with_privacy      # a different group
    assert MEASURED_CORE <= with_privacy         # core always runs


def test_resolve_groups_rejects_unknown_names():
    assert detectors.resolve_groups(None) == []
    assert detectors.resolve_groups("privacy") == ["privacy"]
    assert set(detectors.resolve_groups("all")) == set(detectors.EXTRA_GROUPS)
    with pytest.raises(ValueError, match="unknown check group"):
        detectors.resolve_groups("privacy,nonsense")


def test_every_detector_declares_a_known_group():
    for name, det in detectors.REGISTRY.items():
        assert det.group in detectors.ALL_GROUPS, f"{name} has group {det.group!r}"


def test_governance_findings_are_categorised_apart():
    gov = Finding(type="missing_license", title="t", detail="d", detector="x")
    train = Finding(type="out_of_bounds", title="t", detail="d", detector="x")
    assert gov.category == GOVERNANCE
    assert train.category == TRAINABILITY


def test_governance_findings_do_not_change_the_verdict(clean_root):
    """A licensing question must not read as "do not train"."""
    from dsdoctor.card import deterministic_verdict
    from dsdoctor.findings import CRITICAL

    gov = Finding(type="missing_license", title="t", detail="d", detector="x")
    assert deterministic_verdict([gov]) == "usable_with_caveats"
    real = Finding(type="out_of_bounds", title="t", detail="d", detector="x",
                   severity=CRITICAL)
    assert deterministic_verdict([gov, real]) == "blocked"


def test_governance_findings_do_not_trip_the_ci_gate(clean_root):
    from dsdoctor.sweep import sweep, should_fail

    res = sweep(Dataset(clean_root), groups=["privacy"])
    assert any(f.type == "missing_license" for f in res.findings)
    assert should_fail(res, "critical") is False
    assert should_fail(res, "major") is False
    assert should_fail(res, "any") is False


def test_agent_cannot_run_a_detector_from_a_disabled_group(clean_root):
    from dsdoctor.tools import ToolBox

    tb = ToolBox(Dataset(clean_root))
    assert "not enabled" in tb.run_detector("privacy_scan")["error"]
    tb2 = ToolBox(Dataset(clean_root), groups=["privacy"])
    assert "error" not in tb2.run_detector("privacy_scan")


# ------------------------------------------------------------------ plugins

class _FakeEntryPoint:
    """Stands in for an installed package advertising `dsdoctor.detectors`."""

    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    def load(self):
        return self._fn


def test_plugin_detectors_are_registered_and_attributed(monkeypatch, clean_root):
    from dsdoctor.findings import Finding

    def register_all():
        @detectors.register("acme_naming_scan",
                            "check names against the house style",
                            covers=("empty_label_file",), group="core")
        def _scan(ds):
            return [Finding(type="empty_label_file", title="acme",
                            detail="d", detector="acme_naming_scan")]

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda **kw: [_FakeEntryPoint("acme", register_all)])
    monkeypatch.setattr(detectors, "REGISTRY", dict(detectors.REGISTRY))

    assert detectors.load_plugins() == ["acme"]
    assert detectors.REGISTRY["acme_naming_scan"].origin == "acme"
    assert "acme_naming_scan" in {d.name for d in detectors.available()}


def test_a_broken_plugin_does_not_take_the_audit_down(monkeypatch, capsys,
                                                      clean_root):
    from dsdoctor.dataset import Dataset
    from dsdoctor.sweep import sweep

    def explode():
        raise RuntimeError("bad plugin")

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda **kw: [_FakeEntryPoint("broken", explode)])
    assert detectors.load_plugins() == []
    assert "failed to load" in capsys.readouterr().out
    assert sweep(Dataset(clean_root)).detectors_run          # still works
