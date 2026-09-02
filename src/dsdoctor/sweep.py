"""Run the deterministic detectors and hand back findings.

`scan`, `card`, `recheck` and `diff` all need the same thing: every
finding-producing detector in a chosen set of groups, run over a dataset, with
timings. That used to live inside the `scan` command, which meant every new
command either duplicated it or quietly ran a different set of checks than the
one the user thought they had asked for.

Nothing here involves a language model. `dsdoctor audit` is the path that adds
judgement; this is the path that establishes facts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import detectors
from .dataset import Dataset
from .findings import Finding, CRITICAL, MAJOR, TRAINABILITY, sort_findings


@dataclass
class SweepResult:
    findings: list[Finding] = field(default_factory=list)
    detectors_run: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    groups: list[str] = field(default_factory=list)

    @property
    def trainability(self) -> list[Finding]:
        return [f for f in self.findings if f.category == TRAINABILITY]

    def count(self, severity: str) -> int:
        return sum(1 for f in self.trainability if f.severity == severity)

    @property
    def worst_severity(self) -> str | None:
        for sev in (CRITICAL, MAJOR, "minor"):
            if self.count(sev):
                return sev
        return None


def sweep(ds: Dataset, *, groups: list[str] | None = None,
          experimental: bool = False,
          on_start=None, on_done=None) -> SweepResult:
    """Run every finding-producing detector in the selected groups.

    A detector that raises is recorded and skipped rather than taking the run
    down with it: a plugin or an optional extra failing on one dataset must
    not cost the user the twenty checks that would have worked.
    """
    res = SweepResult(groups=list(groups or []))
    for det in detectors.available(include_experimental=experimental,
                                   groups=groups):
        if not det.covers:
            continue          # informational tools, not checks
        if on_start:
            on_start(det)
        t0 = time.time()
        try:
            found = det.fn(ds)
        except Exception as exc:
            res.failed[det.name] = f"{type(exc).__name__}: {exc}"
            found = []
        res.timings[det.name] = time.time() - t0
        res.findings.extend(found)
        res.detectors_run.append(det.name)
        if on_done:
            on_done(det, found, res.timings[det.name])
    res.findings = sort_findings(res.findings)
    return res


# Ordered worst-first, so "--fail-on major" also fails on critical.
FAIL_LEVELS = {"critical": (CRITICAL,), "major": (CRITICAL, MAJOR),
               "minor": (CRITICAL, MAJOR, "minor"), "any": (CRITICAL, MAJOR, "minor")}


def should_fail(res: SweepResult, level: str | None) -> bool:
    """Whether a CI gate at `level` should reject this dataset.

    Governance findings never fail the gate on their own. A build server is
    not the right place to discover a licensing question, and wiring one into
    a red build teaches people to pass `--fail-on` a weaker value, which loses
    the trainability gate too.
    """
    if not level:
        return False
    wanted = FAIL_LEVELS.get(level)
    if wanted is None:
        raise ValueError(f"unknown --fail-on level {level!r}; "
                         f"choose from {', '.join(FAIL_LEVELS)}")
    return any(f.severity in wanted for f in res.trainability)
