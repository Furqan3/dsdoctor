"""Machine-readable renderings of a sweep.

The point of these is to let dsdoctor sit in a pipeline rather than in a
terminal. A dataset is a build artefact like any other; the checks in this
package are cheap enough to run on every change to it, and the only thing
standing between them and a CI job is an exit code and a format the CI system
already understands.

SARIF is that format for GitHub: results uploaded as SARIF are rendered as
annotations against the offending files, so a leaked validation image shows up
on the pull request that leaked it, next to the file, instead of in a log
nobody opens.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import __version__
from .dataset import Dataset
from .findings import DEFECT_TYPES, CRITICAL, MAJOR
from .sweep import SweepResult

# SARIF has three levels and no notion of "major". Mapping major to `error`
# would make a mislabelled-but-trainable dataset indistinguishable from one
# that crashes the run, which is the distinction this tool exists to draw.
SARIF_LEVEL = {CRITICAL: "error", MAJOR: "warning", "minor": "note"}


def to_json(ds: Dataset, res: SweepResult, *, elapsed: float = 0.0) -> str:
    s = ds.summary()
    return json.dumps({
        "schema": "dsdoctor/scan/1",
        "dsdoctor_version": __version__,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"path": s["root"], "num_classes": s["nc"],
                    "splits": s["splits"], "total_boxes": s["total_boxes"]},
        "checks": {"groups": ["core", *res.groups],
                   "detectors_run": res.detectors_run,
                   "detectors_failed": res.failed},
        "summary": {"critical": res.count(CRITICAL), "major": res.count(MAJOR),
                    "minor": res.count("minor"),
                    "governance": len(res.findings) - len(res.trainability)},
        "elapsed_seconds": round(elapsed, 3),
        "findings": [{
            "type": f.type,
            "severity": f.severity,
            "category": f.category,
            "detector": f.detector,
            "title": f.title,
            "detail": f.detail,
            "affected_files": f.n_items,
            "items": f.items,
            "evidence": f.evidence,
            "fix": f.fix,
        } for f in res.findings],
    }, indent=2)


def _artifact_uri(ds: Dataset, key: str) -> str:
    """A repo-relative path for a finding key, preferring the label file.

    Findings are keyed "split/stem"; an annotation defect belongs on the .txt
    that carries it, and an image defect on the image. Falling back to the
    dataset root keeps the result valid when neither file exists, which is
    itself one of the defects reported here.
    """
    sample = ds.get(key)
    path = None
    if sample is not None:
        path = sample.label_path or sample.image_path
    if path is None:
        return ds.root.name
    try:
        return path.relative_to(ds.root.parent).as_posix()
    except ValueError:  # pragma: no cover - path outside the tree
        return path.as_posix()


def to_sarif(ds: Dataset, res: SweepResult, *, max_locations: int = 100) -> str:
    rules, seen = [], set()
    for f in res.findings:
        if f.type in seen:
            continue
        seen.add(f.type)
        rules.append({
            "id": f.type,
            "name": f.type,
            "shortDescription": {"text": DEFECT_TYPES.get(f.type, ("", f.type))[1]},
            "fullDescription": {"text": f.detail},
            "defaultConfiguration": {
                "level": SARIF_LEVEL.get(f.severity, "note")},
            "properties": {"category": f.category, "detector": f.detector},
        })

    results = []
    for f in res.findings:
        locations = [{
            "physicalLocation": {"artifactLocation": {"uri": _artifact_uri(ds, k)}}
        } for k in f.items[:max_locations]]
        if not locations:
            # Dataset-level findings (a bad data.yaml, an unusable split
            # ratio) still need somewhere to land, or CI drops them entirely.
            locations = [{"physicalLocation": {"artifactLocation": {
                "uri": (ds.yaml_path.relative_to(ds.root.parent).as_posix()
                        if ds.yaml_path else ds.root.name)}}}]
        results.append({
            "ruleId": f.type,
            "level": SARIF_LEVEL.get(f.severity, "note"),
            "message": {"text": f"{f.title}. {f.detail}"},
            "locations": locations,
            "properties": {"affectedFiles": f.n_items,
                           "evidence": f.evidence[:5],
                           "category": f.category},
        })

    return json.dumps({
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spectool"
                   "/main/schemas/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "dsdoctor",
                "version": __version__,
                "informationUri": "https://github.com/Furqan3/dsdoctor",
                "rules": rules,
            }},
            "results": results,
        }],
    }, indent=2)
