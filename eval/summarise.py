"""Turn one or more runs' scores.json into the Markdown tables in the README.

    python eval/summarise.py runs/main
    python eval/summarise.py runs/main runs/main-2 runs/main-3

Kept as a script rather than hand-copied numbers so every figure in the
write-up can be regenerated from the run directories it came from.

With more than one run the tables report the mean and the observed range.
That matters here: `temperature=0` does not make a served model deterministic,
because continuous batching reorders floating-point reductions, so the model
arms move between runs while the deterministic arm does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ARM_LABEL = {
    "script": "`script` (all detectors, unfiltered)",
    "baseline": "`baseline` (one direct prompt)",
    "agent": "**`agent` (this project)**",
    "agent_retype": "`agent_retype` (ablation)",
    "agent_noverify": "`agent_noverify` (ablation)",
}
ORDER = ["baseline", "script", "agent", "agent_retype", "agent_noverify"]


def load(dirs: list[str]) -> list[dict]:
    out = []
    for d in dirs:
        p = Path(d) / "scores.json"
        if not p.is_file():
            raise SystemExit(f"no scores.json in {d}")
        out.append(json.loads(p.read_text()))
    return out


def spread(values: list, fmt) -> str:
    """One number if it never moved, otherwise mean with the observed range."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    lo, hi = fmt(min(vals)), fmt(max(vals))
    if lo == hi:
        # Collapse on the *rendered* value: 755.6s and 756.2s both print as
        # "756s", and "756s [756s-756s]" is noise, not information.
        return lo
    return f"{fmt(mean(vals))} [{lo}–{hi}]"


def pct(v) -> str:
    return f"{v:.1%}"


def num(v) -> str:
    return f"{v:.0f}"


def collect(runs: list[dict], arm: str, key: str) -> list:
    return [r["aggregate"][arm][key] for r in runs if arm in r["aggregate"]]


def headline_table(runs: list[dict], arms: list[str]) -> str:
    L = ["| arm | defect recall | precision | false positives | verdict correct | wall time |",
         "|---|---|---|---|---|---|"]
    for arm in arms:
        rec = spread(collect(runs, arm, "recall"), pct)
        pre = spread(collect(runs, arm, "precision"), pct)
        fps = spread(collect(runs, arm, "false_positives"), num)
        vc_n = collect(runs, arm, "verdict_correct")
        vc_d = collect(runs, arm, "verdict_scored")
        vc = (f"{spread(vc_n, num)} / {vc_d[0]}"
              if vc_d and vc_d[0] else "n/a")
        secs = spread(collect(runs, arm, "wall_seconds"), lambda v: f"{v:.0f}s")
        L.append(f"| {ARM_LABEL.get(arm, arm)} | {rec} | {pre} | {fps} | {vc} | {secs} |")
    return "\n".join(L)


def cost_table(runs: list[dict], arms: list[str]) -> str:
    L = ["| arm | model calls | tool calls | prompt tokens | completion tokens | s / dataset |",
         "|---|---|---|---|---|---|"]
    for arm in arms:
        cases = collect(runs, arm, "cases")
        n = cases[0] if cases else 1
        L.append(
            f"| {ARM_LABEL.get(arm, arm)} "
            f"| {spread(collect(runs, arm, 'llm_calls'), num)} "
            f"| {spread(collect(runs, arm, 'tool_calls'), num)} "
            f"| {spread(collect(runs, arm, 'prompt_tokens'), lambda v: f'{v:,.0f}')} "
            f"| {spread(collect(runs, arm, 'completion_tokens'), lambda v: f'{v:,.0f}')} "
            f"| {spread([v / max(n, 1) for v in collect(runs, arm, 'wall_seconds')], num)} |")
    return "\n".join(L)


def per_case_table(runs: list[dict], arms: list[str]) -> str:
    cases: dict[str, dict[str, list]] = {}
    for r in runs:
        for row in r["per_case"]:
            cases.setdefault(row["case"], {}).setdefault(row["arm"], []).append(row)
    L = ["| case | injected facts | " + " | ".join(f"`{a}`" for a in arms) + " |",
         "|---|---|" + "---|" * len(arms)]
    for case, by_arm in cases.items():
        any_row = next(iter(by_arm.values()))[0]
        gt = any_row["n_ground_truth"]
        cells = []
        for a in arms:
            rows = by_arm.get(a)
            if not rows:
                cells.append("—")
                continue
            recalls = [x["recall"] for x in rows]
            fps = [x["false_positives"] for x in rows]
            if all(v is None for v in recalls):
                cells.append(f"{spread(fps, num)} FP")
            else:
                cell = spread(recalls, lambda v: f"{v:.0%}")
                if any(f for f in fps):
                    cell += f" · {spread(fps, num)} FP"
                cells.append(cell)
        L.append(f"| `{case}` | {gt} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--plain", action="store_true",
                    help="aligned plain-text summary instead of Markdown "
                         "(readable in a terminal or on video)")
    args = ap.parse_args()
    runs = load(args.run_dirs)
    arms = [a for a in ORDER if any(a in r["aggregate"] for r in runs)]

    if args.plain:
        print()
        print(f"  {'arm':36s}{'recall':>9s}{'precision':>11s}"
              f"{'false pos':>11s}{'verdict':>10s}{'time':>9s}")
        print("  " + "-" * 86)
        for arm in arms:
            rec = spread(collect(runs, arm, "recall"), pct)
            pre = spread(collect(runs, arm, "precision"), pct)
            fps = spread(collect(runs, arm, "false_positives"), num)
            vc_n = collect(runs, arm, "verdict_correct")
            vc_d = collect(runs, arm, "verdict_scored")
            vc = f"{vc_n[0]}/{vc_d[0]}" if vc_d and vc_d[0] else "--"
            # plain mode is for reading on screen: one number per cell, so
            # the mean stands in for the range the Markdown table spells out
            w = collect(runs, arm, "wall_seconds")
            secs = f"{mean(w):.0f}s" if w else "n/a"
            label = {"baseline": "baseline  (one direct prompt)",
                     "script": "script    (all detectors, no model)",
                     "agent": "agent     (this project)"}.get(arm, arm)
            print(f"  {label:36s}{rec:>9s}{pre:>11s}{fps:>11s}{vc:>10s}{secs:>9s}")
        cases = collect(runs, arms[0], "cases")
        print()
        print(f"  {cases[0] if cases else '?'} cases · {len(runs)} independent run(s) · "
              f"objective (defect, file) matching · no LLM judge")
        print()
        return 0

    n = len(runs)
    print(f"<!-- generated by eval/summarise.py from "
          f"{', '.join(args.run_dirs)} -->")
    print(f"<!-- model: {runs[0].get('model')}  corpus: {runs[0].get('corpus')} -->\n")
    if n > 1:
        print(f"*{n} independent runs of the same twelve cases; cells show the "
              f"mean with the observed range in brackets where it moved.*\n")
    print("### Headline\n")
    print(headline_table(runs, arms))
    print("\n### Cost\n")
    print(cost_table(runs, arms))
    print("\n### Per case\n")
    print(per_case_table(runs, arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
