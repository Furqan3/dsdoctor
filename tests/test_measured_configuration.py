"""The published results must keep describing the shipped tool.

`runs/main/reports/*.script.json` is the deterministic arm of the evaluation
whose numbers appear in the README. It involves no language model, so it is
reproducible exactly — which makes it usable as a regression fixture rather
than only as a record.

This is the test that would have caught the two ways this project could
silently invalidate its own results table: adding a detector to the default
set, and changing what an existing detector reports. Both are legitimate
changes; neither may happen without the table being regenerated.

It skips when `data/cases/` has not been built (it is generated, not committed,
so a fresh clone does not have it). Build it with:

    python eval/build_corpus.py && python eval/run_eval.py --cases-only
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsdoctor.dataset import Dataset
from dsdoctor.detectors import available

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "cases"
REPORTS = ROOT / "runs" / "main" / "reports"

pytestmark = pytest.mark.skipif(
    not CASES.is_dir() or not REPORTS.is_dir(),
    reason="evaluation corpus not built; see eval/build_corpus.py")


def _signature(findings) -> list[tuple]:
    """Everything the report asserts, in a comparable shape."""
    return sorted((f["type"] if isinstance(f, dict) else f.type,
                   f["detector"] if isinstance(f, dict) else f.detector,
                   tuple(f["items"] if isinstance(f, dict) else f.items),
                   tuple(f["evidence"] if isinstance(f, dict) else f.evidence))
                  for f in findings)


def _sweep(ds: Dataset):
    out = []
    for det in available():          # the core group: the measured set
        if det.covers:
            out.extend(det.fn(ds))
    return out


def _cases():
    return sorted(p.name.split(".")[0] for p in REPORTS.glob("*.script.json")
                  if (CASES / p.name.split(".")[0]).is_dir())


@pytest.mark.parametrize("case", _cases() or ["<none>"])
def test_deterministic_arm_reproduces_the_published_run(case):
    if case == "<none>":
        pytest.skip("no evaluation cases available")
    stored = json.loads((REPORTS / f"{case}.script.json").read_text())["findings"]
    got = _sweep(Dataset(CASES / case))
    assert _signature(got) == _signature(stored), (
        f"the deterministic detectors no longer reproduce the published run "
        f"for {case!r}. If the change is intentional, regenerate the "
        f"evaluation (eval/run_eval.py) and the README tables "
        f"(eval/summarise.py) in the same commit.")
