"""Run the arms over every case and write a comparison table.

    python eval/run_eval.py --arms script,baseline,agent
    python eval/run_eval.py --arms agent --cases subtle_leak,ambiguity_trap

Three arms, the same cases and the same scoring for each:

  script    every detector, everything it finds reported verbatim. This is the
            "simple script" baseline and it is a strong one - it is the recall
            ceiling of the deterministic layer.
  baseline  one direct prompt with the dataset summary and a large sample of
            the raw label rows. The "one direct prompt" baseline.
  agent     the tool-using auditor plus the verification pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from cases import CASES, by_name                      # noqa: E402
from injector import build_case, DATASET_LEVEL        # noqa: E402
from score import score, aggregate                    # noqa: E402

from dsdoctor.dataset import Dataset                  # noqa: E402
from dsdoctor.detectors import REGISTRY               # noqa: E402
from dsdoctor.llm import LLM, Trajectory              # noqa: E402
from dsdoctor.agent import audit                      # noqa: E402
from dsdoctor.baseline import (run_baseline, baseline_facts,  # noqa: E402
                               baseline_scope)


# ------------------------------------------------------------------- arms

def arm_script(ds: Dataset, llm, experimental: bool):
    """Every detector, nothing filtered. No language model involved."""
    t0 = time.time()
    facts, findings = set(), []
    for det in REGISTRY.values():
        if det.experimental and not experimental:
            continue
        if not det.covers:
            continue
        for f in det.fn(ds):
            findings.append(f)
            for k in (f.items or [DATASET_LEVEL]):
                facts.add((f.type, k))
    from dsdoctor.findings import sort_findings, CRITICAL
    ranked = sort_findings(findings)
    return facts, {
        "wall_seconds": round(time.time() - t0, 2),
        "llm_calls": 0, "tool_calls": 0,
        "prompt_tokens": 0, "completion_tokens": 0,
        "verdict": "n/a",
        "top_finding_type": ranked[0].type if ranked else "",
        "top_is_critical": (ranked[0].severity == CRITICAL) if ranked else None,
    }, None, {"findings": [f.to_dict() for f in ranked]}


def arm_baseline(ds: Dataset, llm: LLM, experimental: bool):
    t0 = time.time()
    report, traj = run_baseline(ds, llm)
    facts = baseline_facts(report)
    scope = baseline_scope(report)
    from dsdoctor.findings import DEFECT_TYPES, CRITICAL
    first = (report.get("findings") or [{}])[0]
    ftype = str(first.get("type", ""))
    return facts, {
        "wall_seconds": round(time.time() - t0, 2),
        "llm_calls": traj.llm_calls, "tool_calls": 0,
        "prompt_tokens": traj.prompt_tokens,
        "completion_tokens": traj.completion_tokens,
        "verdict": str(report.get("verdict", "")),
        "top_finding_type": ftype,
        "top_is_critical": (DEFECT_TYPES.get(ftype, ("", ""))[0] == CRITICAL)
                           if ftype else None,
        "_scope": scope,
    }, traj, report


def arm_agent(ds: Dataset, llm: LLM, experimental: bool):
    return _agent_common(ds, llm, experimental)


def _agent_common(ds: Dataset, llm: LLM, experimental: bool,
                  retype: bool = False, verify: bool = True):
    t0 = time.time()
    res = audit(ds, llm, experimental=experimental, retype=retype, verify=verify)
    facts = res.as_facts()
    from dsdoctor.findings import CRITICAL
    top = res.reported[0] if res.reported else None
    return facts, {
        "wall_seconds": round(time.time() - t0, 2),
        "llm_calls": res.trajectory.llm_calls,
        "tool_calls": res.trajectory.tool_calls,
        "prompt_tokens": res.trajectory.prompt_tokens,
        "completion_tokens": res.trajectory.completion_tokens,
        "verdict": res.verdict,
        "top_finding_type": top.finding.type if top else "",
        "top_is_critical": (top.severity == CRITICAL) if top else None,
    }, res.trajectory, {
        "verdict": res.verdict, "headline": res.headline,
        "training_impact": res.training_impact,
        "detectors_run": res.detectors_run,
        "suppressed": res.suppressed_total,
        "reinstated": res.reinstated_total,
        "swept_fallback": res.swept_fallback,
        "incomplete": res.incomplete,
        "decisions": [{"finding_id": d.finding_id, "type": d.finding.type,
                       "action": d.action, "severity": d.severity,
                       "source": d.source, "rationale": d.rationale,
                       "files": d.finding.n_items}
                      for d in res.decisions],
    }


def arm_agent_retype(ds: Dataset, llm: LLM, experimental: bool):
    """Ablation: the agent retypes findings instead of curating ids."""
    return _agent_common(ds, llm, experimental, retype=True)


def arm_agent_noverify(ds: Dataset, llm: LLM, experimental: bool):
    """Ablation: no verification pass over the agent's suppressions."""
    return _agent_common(ds, llm, experimental, verify=False)


ARMS = {"script": arm_script, "baseline": arm_baseline, "agent": arm_agent,
        "agent_retype": arm_agent_retype, "agent_noverify": arm_agent_noverify}


# ------------------------------------------------------------------ driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus_clean")
    ap.add_argument("--workdir", default="data/cases")
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default="script,baseline,agent")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--experimental", action="store_true",
                    help="also run detectors measured as net-harmful "
                         "(currently: model_disagreement_scan)")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; have {sorted(ARMS)}")
    chosen = CASES if args.cases == "all" else [by_name(n) for n in args.cases.split(",")]

    kw = {}
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.model:
        kw["model"] = args.model
    llm = LLM(**kw)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out or f"runs/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories").mkdir(exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    all_scores = []
    print(f"corpus: {args.corpus}   arms: {', '.join(arms)}   "
          f"cases: {len(chosen)}   out: {out_dir}\n")

    for case in chosen:
        case_dir = workdir / case["name"]
        gt = build_case(Path(args.corpus), case_dir, case["recipe"], case["seed"])
        (out_dir / "reports" / f"{case['name']}.ground_truth.json").write_text(
            json.dumps(gt, indent=2))
        G = {tuple(x) for x in gt["ground_truth"]}
        C = {tuple(x) for x in gt["collateral"]}
        print(f"[{case['name']}] {len(G)} ground-truth fact(s)")

        for arm in arms:
            ds = Dataset(case_dir)
            try:
                facts, meta, traj, report = ARMS[arm](ds, llm, args.experimental)
            except Exception as exc:
                print(f"    {arm:9s} FAILED: {type(exc).__name__}: {exc}")
                continue
            scope = meta.pop("_scope", None)
            s = score(case["name"], arm, facts, G, C, scope=scope, **meta)
            all_scores.append(s)
            if traj is not None:
                traj.save(out_dir / "trajectories" / f"{case['name']}.{arm}.json")
            if report is not None:
                (out_dir / "reports" / f"{case['name']}.{arm}.json").write_text(
                    json.dumps(report, indent=2, default=str))
            r = "n/a" if s.recall is None else f"{s.recall:6.1%}"
            print(f"    {arm:9s} recall {r}  fp {s.false_positives:3d} "
                  f"(swap {s.class_swap_false_positives:2d})  "
                  f"{s.wall_seconds:6.1f}s  {s.llm_calls:2d} llm  "
                  f"{s.tool_calls:2d} tool")

    agg = aggregate(all_scores)
    (out_dir / "scores.json").write_text(json.dumps(
        {"generated": stamp,
         "corpus": args.corpus,
         "model": llm.model,
         "cases": [c["name"] for c in chosen],
         "aggregate": agg,
         "per_case": [s.to_dict() for s in all_scores]}, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"{'arm':10s}{'recall':>9s}{'precision':>11s}{'FP':>6s}{'critFP':>8s}"
          f"{'swapFP':>8s}{'seconds':>9s}{'llm':>6s}")
    print("-" * 78)
    for arm in arms:
        a = agg.get(arm)
        if not a:
            continue
        rec = "n/a" if a["recall"] is None else f"{a['recall']:.1%}"
        pre = "n/a" if a["precision"] is None else f"{a['precision']:.1%}"
        print(f"{arm:10s}{rec:>9s}{pre:>11s}{a['false_positives']:>6d}"
              f"{a['critical_false_positives']:>8d}"
              f"{a['class_swap_false_positives']:>8d}"
              f"{a['wall_seconds']:>9.1f}{a['llm_calls']:>6d}")
    print("=" * 78)
    print(f"\nwrote {out_dir}/scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
