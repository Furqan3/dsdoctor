"""The tool surface the auditing agent works through.

One decision shapes everything here: the agent never retypes a finding. Every
detector result is registered under a stable id, and the agent's report is a
list of *decisions* about those ids - keep this, suppress that, rank this
first. A 27B model asked to re-emit 40 findings with their file lists will
quietly drop a few, and a dataset audit that silently loses defects is worse
than no audit. Curating ids cannot lose evidence; it can only mis-rank it,
which is a failure the evaluation can actually see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .dataset import Dataset
from .findings import Finding, DEFECT_TYPES, sort_findings
from . import detectors


@dataclass
class ToolBox:
    ds: Dataset
    experimental: bool = False
    retype: bool = False
    findings: dict[str, Finding] = field(default_factory=dict)
    ran: list[str] = field(default_factory=list)
    report: dict | None = None

    # ----------------------------------------------------------- tool impls

    def dataset_summary(self) -> dict:
        s = self.ds.summary()
        return {
            "root": s["root"],
            "num_classes": s["nc"],
            "class_names": s["names"],
            "splits": s["splits"],
            "total_boxes": s["total_boxes"],
            "data_yaml_problem": s["yaml_error"],
        }

    def list_detectors(self) -> dict:
        out = []
        for d in detectors.available(include_experimental=self.experimental):
            out.append({
                "name": d.name,
                "description": d.description,
                "detects": list(d.covers),
                "cost": "slow" if d.heavy else ("medium" if d.reads_pixels else "fast"),
                "reliability": ("experimental - known to produce false "
                                "positives, treat output as a hypothesis"
                                if d.experimental else
                                "exact - reads the files directly"),
                "already_run": d.name in self.ran,
            })
        return {"detectors": out,
                "note": "Every detector is deterministic and reads the files "
                        "directly. None of them use a language model."}

    def run_detector(self, name: str) -> dict:
        if name not in detectors.REGISTRY:
            return {"error": f"unknown detector {name!r}",
                    "available": sorted(detectors.REGISTRY)}
        det = detectors.REGISTRY[name]
        if det.experimental and not self.experimental:
            return {"error": f"{name} is disabled in this run; it is an "
                             f"experimental detector and is off by default"}
        try:
            found = det.fn(self.ds)
        except Exception as exc:
            return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
        if name not in self.ran:
            self.ran.append(name)

        rows = []
        for i, f in enumerate(found):
            fid = f"{name}:{f.type}:{i}"
            self.findings[fid] = f
            rows.append({
                "finding_id": fid,
                "type": f.type,
                "detector_severity": f.severity,
                "title": f.title,
                "affected_files": f.n_items,
                "example_files": f.items[:3],
                "example_evidence": f.evidence[:2],
            })
        return {"detector": name, "findings_found": len(rows), "findings": rows,
                "hint": "Use inspect_finding to see the full evidence before "
                        "deciding whether a finding is real."}

    def inspect_finding(self, finding_id: str, max_evidence: int = 12) -> dict:
        f = self.findings.get(finding_id)
        if f is None:
            return {"error": f"no finding {finding_id!r}",
                    "known": sorted(self.findings)[:40]}
        return {
            "finding_id": finding_id,
            "type": f.type,
            "meaning": DEFECT_TYPES.get(f.type, ("", "unknown defect type"))[1],
            "detector": f.detector,
            "detector_severity": f.severity,
            "title": f.title,
            "explanation": f.detail,
            "affected_files": f.items[:60],
            "affected_file_count": f.n_items,
            "evidence": f.evidence[:max_evidence],
            "proposed_fix": f.fix,
        }

    def read_label_file(self, file_key: str) -> dict:
        """file_key is 'split/stem', e.g. 'train/000000012345'."""
        s = self.ds.get(file_key)
        if s is None:
            return {"error": f"no sample {file_key!r}"}
        if s.label is None:
            return {"file": file_key, "label_file": None,
                    "note": "image has no label file"}
        self.ds.ensure_image_meta(s)
        return {
            "file": file_key,
            "image_size": [s.width, s.height] if s.width else None,
            "image_error": s.image_error,
            "num_boxes": s.label.n_boxes,
            "rows": [{"line": b.line_no, "class_id": b.cls,
                      "class_name": self.ds.class_name(b.cls),
                      "xc": b.xc, "yc": b.yc, "w": b.w, "h": b.h}
                     for b in s.label.boxes[:80]],
            "parse_errors": [{"line": e.line_no, "reason": e.reason, "raw": e.raw}
                             for e in s.label.parse_errors],
        }

    def class_distribution(self) -> dict:
        per: dict[str, dict[str, int]] = {}
        for split in self.ds.splits:
            counts: dict[str, int] = {n: 0 for n in self.ds.names}
            for s in self.ds.in_split(split):
                if not s.label:
                    continue
                for b in s.label.boxes:
                    counts[self.ds.class_name(b.cls)] = \
                        counts.get(self.ds.class_name(b.cls), 0) + 1
            per[split] = counts
        return {"instances_per_class_per_split": per}

    def submit_report_retyped(self, verdict: str, headline: str,
                              findings: list, training_impact: str = "") -> dict:
        """Ablation-only: the model re-emits findings instead of curating ids.

        Kept so the claim that curate-by-id prevents evidence loss can be
        measured rather than asserted. Not used by the shipped agent.
        """
        self.report = {
            "verdict": verdict,
            "headline": headline,
            "retyped_findings": findings,
            "training_impact": training_impact,
        }
        return {"status": "report recorded", "findings": len(findings)}

    def submit_report(self, verdict: str, headline: str, decisions: list,
                      training_impact: str = "") -> dict:
        self.report = {
            "verdict": verdict,
            "headline": headline,
            "decisions": decisions,
            "training_impact": training_impact,
        }
        return {"status": "report recorded", "decisions": len(decisions)}

    # ------------------------------------------------------------ dispatch

    def call(self, name: str, args: dict) -> dict:
        fn = {
            "dataset_summary": self.dataset_summary,
            "list_detectors": self.list_detectors,
            "run_detector": self.run_detector,
            "inspect_finding": self.inspect_finding,
            "read_label_file": self.read_label_file,
            "class_distribution": self.class_distribution,
            "submit_report": (self.submit_report_retyped if self.retype
                              else self.submit_report),
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}

    def all_findings(self) -> list[Finding]:
        return sort_findings(list(self.findings.values()))


VERDICTS = ["blocked", "fix_before_training", "usable_with_caveats"]

SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "dataset_summary",
        "description": "Size, splits, class names and per-class totals for the "
                       "dataset under audit. Start here.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "list_detectors",
        "description": "List the deterministic checks available, what each one "
                       "detects, and how expensive it is to run.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "run_detector",
        "description": "Run one detector over the whole dataset and get back its "
                       "findings, each with a finding_id.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "detector name from list_detectors"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "inspect_finding",
        "description": "Full evidence for one finding: every affected file and the "
                       "raw rows that triggered it. Use this before suppressing "
                       "anything.",
        "parameters": {"type": "object", "properties": {
            "finding_id": {"type": "string"},
            "max_evidence": {"type": "integer",
                             "description": "how many evidence lines to return (default 12)"}},
            "required": ["finding_id"]}}},
    {"type": "function", "function": {
        "name": "read_label_file",
        "description": "Read one label file's parsed rows plus its image size, to "
                       "check a finding against the underlying data.",
        "parameters": {"type": "object", "properties": {
            "file_key": {"type": "string",
                         "description": "'split/stem', e.g. 'train/000000012345'"}},
            "required": ["file_key"]}}},
    {"type": "function", "function": {
        "name": "class_distribution",
        "description": "Instances of every class in every split.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "submit_report",
        "description": "Deliver the audit. Call this exactly once, at the end. "
                       "List decisions in the order the user should act on them.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": VERDICTS,
                        "description": "blocked = training will crash or the "
                                       "metrics are meaningless; "
                                       "fix_before_training = real defects that "
                                       "will cost accuracy; usable_with_caveats "
                                       "= only minor issues remain"},
            "headline": {"type": "string",
                         "description": "one sentence a busy engineer can act on"},
            "training_impact": {"type": "string",
                                "description": "what happens if they train on this as-is"},
            "decisions": {"type": "array",
                          "description": "one entry per finding_id you saw, most "
                                         "urgent first",
                          "items": {"type": "object", "properties": {
                              "finding_id": {"type": "string"},
                              "action": {"type": "string", "enum": ["report", "suppress"],
                                         "description": "suppress only when the "
                                                        "evidence shows it is not "
                                                        "a real defect"},
                              "severity": {"type": "string",
                                           "enum": ["critical", "major", "minor"]},
                              "rationale": {"type": "string",
                                            "description": "why it matters, or why "
                                                           "you are suppressing it"}},
                              "required": ["finding_id", "action", "rationale"]}}},
            "required": ["verdict", "headline", "decisions"]}}},
]


RETYPE_SUBMIT = {"type": "function", "function": {
    "name": "submit_report",
    "description": "Deliver the audit. Call this exactly once, at the end. "
                   "List every defect you are reporting, with every affected "
                   "file, most urgent first.",
    "parameters": {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "headline": {"type": "string"},
        "training_impact": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string",
                     "description": "the defect type id, e.g. out_of_bounds"},
            "severity": {"type": "string",
                         "enum": ["critical", "major", "minor"]},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "every affected file as 'split/stem'"},
            "rationale": {"type": "string"}},
            "required": ["type", "files", "rationale"]}}},
        "required": ["verdict", "headline", "findings"]}}}


def schemas(retype: bool = False) -> list[dict]:
    """Tool list for the shipped agent, or for the retype ablation."""
    if not retype:
        return SCHEMAS
    return [t for t in SCHEMAS
            if t["function"]["name"] != "submit_report"] + [RETYPE_SUBMIT]


def compact_json(obj) -> str:
    """Tool results go back into the prompt, so keep them tight."""
    return json.dumps(obj, separators=(",", ":"), default=str)
