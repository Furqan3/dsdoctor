"""Render the audit into something an engineer would actually read.

Two artefacts come out of here: a Markdown triage report meant for a human,
and a machine-readable fix plan that `dsdoctor apply` consumes. Nothing is ever
changed on disk by this module.

The report is deliberately not a dump of every finding in severity order. It
opens with the decision the reader has to make - train on this or not - and
then gives them an ordered list of work. Findings the agent chose to suppress
are still printed, in their own section, because a reviewer needs to be able to
overrule that judgement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .agent import AuditResult, Decision
from .dataset import Dataset
from .findings import CRITICAL, MAJOR

VERDICT_LINE = {
    "blocked": "**Do not train on this yet.** Something here will either stop "
               "the run or make its results meaningless.",
    "fix_before_training": "**Fix these before you train.** The run will "
                           "complete, but the model or the metrics will be "
                           "worse than they need to be.",
    "usable_with_caveats": "**Usable.** Only minor issues remain; read them and "
                           "decide if you care.",
}

SEVERITY_TAG = {CRITICAL: "CRITICAL", MAJOR: "MAJOR", "minor": "minor"}


def render_markdown(res: AuditResult, ds: Dataset, *,
                    model: str = "", elapsed: float = 0.0) -> str:
    s = ds.summary()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reported = res.reported
    crit = [d for d in reported if d.severity == CRITICAL]
    major = [d for d in reported if d.severity == MAJOR]
    minor = [d for d in reported if d.severity not in (CRITICAL, MAJOR)]

    L: list[str] = []
    L.append(f"# Trainability audit: `{Path(s['root']).name}`")
    L.append("")
    L.append(VERDICT_LINE.get(res.verdict, f"Verdict: {res.verdict}"))
    L.append("")
    if res.headline:
        L.append(res.headline)
        L.append("")

    files = sum(v["images"] for v in s["splits"].values())
    L.append(f"`{files}` images · `{s['total_boxes']}` boxes · "
             f"`{s['nc']}` classes · "
             + " · ".join(f"`{k}` {v['images']}" for k, v in s["splits"].items()))
    L.append("")
    L.append(f"**{len(crit)} critical · {len(major)} major · {len(minor)} minor**"
             + (f" · {len(res.suppressed)} suppressed" if res.suppressed else ""))
    L.append("")

    if res.incomplete:
        L.append(f"> Note: {res.incomplete}")
        L.append("")

    if res.training_impact:
        L.append("## If you train on this as it stands")
        L.append("")
        L.append(res.training_impact)
        L.append("")

    L.append("## What to fix, in order")
    L.append("")
    if not reported:
        L.append("Nothing. Every check passed.")
        L.append("")
    for i, d in enumerate(reported, 1):
        L.extend(_render_decision(i, d))

    if res.suppressed:
        L.append("## Considered and not reported")
        L.append("")
        L.append("These were raised by a detector and judged not to be real "
                 "defects. They are listed so you can overrule that call.")
        L.append("")
        for d in res.suppressed:
            L.append(f"- **{d.finding.type}** — {d.finding.title}  ")
            L.append(f"  {d.rationale}")
        L.append("")

    L.append("## How this was produced")
    L.append("")
    L.append(f"- Detectors run: {', '.join(res.detectors_run) or 'none'}")
    L.append(f"- Every finding above comes from a deterministic check reading "
             f"the files directly. The model ranked and filtered them; it did "
             f"not generate them.")
    if res.reinstated_total:
        L.append(f"- The verification pass reinstated "
                 f"{res.reinstated_total} finding(s) the auditor had suppressed.")
    if model:
        L.append(f"- Model: `{model}`")
    if elapsed:
        L.append(f"- Wall time: {elapsed:.0f}s")
    L.append(f"- Generated {now} by dsdoctor")
    L.append("")
    return "\n".join(L)


def _render_decision(i: int, d: Decision) -> list[str]:
    f = d.finding
    tag = SEVERITY_TAG.get(d.severity, d.severity)
    L = [f"### {i}. [{tag}] {f.title}", ""]
    if d.rationale and d.source != "default":
        L += [d.rationale, ""]
    L += [f.detail, ""]

    if f.items:
        shown = f.items[:8]
        L.append(f"<details><summary>{f.n_items} affected file(s)</summary>")
        L.append("")
        L += [f"- `{k}`" for k in shown]
        if f.n_items > len(shown):
            L.append(f"- … and {f.n_items - len(shown)} more")
        L.append("")
        L.append("</details>")
        L.append("")

    if f.evidence:
        L.append("```")
        L += [str(e) for e in f.evidence[:5]]
        if len(f.evidence) > 5:
            L.append(f"... {len(f.evidence) - 5} more")
        L.append("```")
        L.append("")
    if f.fix:
        L.append(f"**Suggested fix:** `{f.fix.get('action')}` "
                 f"({len(f.fix.get('targets') or [])} target(s)) — "
                 f"run `dsdoctor apply` to review and approve.")
        L.append("")
    return L


def build_fix_plan(res: AuditResult, ds: Dataset) -> dict:
    """Machine-readable, ordered, and explicitly not yet applied."""
    steps = []
    for d in res.reported:
        f = d.finding
        if not f.fix:
            continue
        steps.append({
            "finding_id": d.finding_id,
            "type": f.type,
            "severity": d.severity,
            "action": f.fix.get("action"),
            "targets": f.fix.get("targets") or [],
            "detail": f.fix.get("from") and
                      {"from": f.fix.get("from"), "to": f.fix.get("to")} or None,
            "why": f.title,
            "requires_human_review": f.type in ("class_swap", "empty_label_file",
                                                "extreme_class_imbalance"),
        })
    return {
        "dataset": str(ds.root),
        "verdict": res.verdict,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved": False,
        "steps": steps,
    }


def write_all(res: AuditResult, ds: Dataset, out_dir: str | Path, *,
              model: str = "", elapsed: float = 0.0) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "audit_report.md"
    md.write_text(render_markdown(res, ds, model=model, elapsed=elapsed))
    plan = out / "fix_plan.json"
    plan.write_text(json.dumps(build_fix_plan(res, ds), indent=2))
    paths = {"report": md, "fix_plan": plan}
    if res.trajectory is not None:
        paths["trajectory"] = res.trajectory.save(out / "trajectory.json")
    return paths
