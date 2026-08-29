"""Scoring: compare a set of reported (defect_type, file) facts to ground truth.

There is no model in the loop here and no rubric to interpret. An arm either
named the right defect type on the right file or it did not.

Three sets matter:
  ground truth  what the injector actually did
  collateral    second-order truths an injection creates (pixel coordinates are
                also out of bounds). Reporting these is correct, so they count
                neither for recall nor against precision.
  everything else that was reported is a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from dsdoctor.findings import DEFECT_TYPES, CRITICAL, MAJOR


@dataclass
class Score:
    case: str
    arm: str
    n_ground_truth: int
    n_reported: int
    hits: int
    misses: int
    false_positives: int
    critical_false_positives: int
    class_swap_false_positives: int
    recall: float | None
    precision: float | None
    missed_facts: list
    fp_facts: list
    n_in_scope: int | None = None
    hits_in_scope: int | None = None
    recall_in_scope: float | None = None
    wall_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verdict: str = ""
    expected_verdict: str = ""
    verdict_correct: bool | None = None
    top_finding_type: str = ""
    top_is_critical: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def expected_verdict(ground_truth: set) -> str:
    """What a correct audit should conclude, derived from what was injected."""
    types = {t for t, _ in ground_truth}
    if any(DEFECT_TYPES.get(t, ("", ""))[0] == CRITICAL for t in types):
        return "blocked"
    if any(DEFECT_TYPES.get(t, ("", ""))[0] == MAJOR for t in types):
        return "fix_before_training"
    return "usable_with_caveats"


def score(case: str, arm: str, reported: set, ground_truth: set,
          collateral: set, scope: set | None = None, **extra) -> Score:
    gt = {tuple(x) for x in ground_truth}
    col = {tuple(x) for x in collateral}
    rep = {tuple(x) for x in reported}

    hits = gt & rep
    misses = gt - rep
    fps = rep - gt - col

    crit_fp = [f for f in fps
               if DEFECT_TYPES.get(f[0], ("", ""))[0] == CRITICAL]
    swap_fp = [f for f in fps if f[0] == "class_swap"]

    # When an arm can only see part of the dataset, separate what it missed
    # because it never saw the file from what it missed despite seeing it.
    n_scope = hits_scope = rec_scope = None
    if scope is not None:
        in_scope = {(t, k) for t, k in gt if k == "<dataset>" or k in scope}
        n_scope = len(in_scope)
        hits_scope = len(in_scope & rep)
        rec_scope = (hits_scope / n_scope) if n_scope else None

    scored = rep - col
    want = expected_verdict(gt)
    got = str(extra.pop("verdict", "") or "")
    extra["verdict"] = got
    extra["expected_verdict"] = want
    extra["verdict_correct"] = (got == want) if got and got != "n/a" else None

    return Score(
        case=case, arm=arm,
        n_ground_truth=len(gt), n_reported=len(rep),
        hits=len(hits), misses=len(misses),
        false_positives=len(fps),
        critical_false_positives=len(crit_fp),
        class_swap_false_positives=len(swap_fp),
        recall=(len(hits) / len(gt)) if gt else None,
        precision=(len(hits) / len(scored)) if scored else None,
        n_in_scope=n_scope, hits_in_scope=hits_scope, recall_in_scope=rec_scope,
        missed_facts=sorted(misses),
        fp_facts=sorted(fps)[:40],
        **extra,
    )


def aggregate(scores: list[Score]) -> dict:
    """Micro-averaged over facts, so a big case counts more than a small one."""
    by_arm: dict[str, list[Score]] = {}
    for s in scores:
        by_arm.setdefault(s.arm, []).append(s)

    out = {}
    for arm, rows in by_arm.items():
        gt = sum(r.n_ground_truth for r in rows)
        hits = sum(r.hits for r in rows)
        fps = sum(r.false_positives for r in rows)
        scored = hits + fps
        out[arm] = {
            "cases": len(rows),
            "ground_truth_facts": gt,
            "hits": hits,
            "misses": sum(r.misses for r in rows),
            "recall": (hits / gt) if gt else None,
            "ground_truth_in_scope": (sum(r.n_in_scope or 0 for r in rows)
                                      if any(r.n_in_scope is not None for r in rows)
                                      else None),
            "recall_in_scope": (
                (sum(r.hits_in_scope or 0 for r in rows)
                 / max(sum(r.n_in_scope or 0 for r in rows), 1))
                if any(r.n_in_scope is not None for r in rows) else None),
            "false_positives": fps,
            "critical_false_positives": sum(r.critical_false_positives for r in rows),
            "class_swap_false_positives": sum(r.class_swap_false_positives for r in rows),
            "precision": (hits / scored) if scored else None,
            "f1": (2 * hits / (gt + scored)) if (gt + scored) else None,
            "verdict_correct": sum(1 for r in rows if r.verdict_correct is True),
            "verdict_scored": sum(1 for r in rows if r.verdict_correct is not None),
            "leads_with_critical": sum(1 for r in rows if r.top_is_critical is True),
            "wall_seconds": round(sum(r.wall_seconds for r in rows), 1),
            "llm_calls": sum(r.llm_calls for r in rows),
            "tool_calls": sum(r.tool_calls for r in rows),
            "prompt_tokens": sum(r.prompt_tokens for r in rows),
            "completion_tokens": sum(r.completion_tokens for r in rows),
        }
    return out
