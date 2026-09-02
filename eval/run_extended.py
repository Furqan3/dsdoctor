"""Score the opt-in check groups against injected ground truth.

Kept separate from `run_eval.py` on purpose. That script produces the twelve
numbers in the README, and those describe the `core` detector set; folding
these cases into it would change every published figure's denominator while
appearing to be an addition.

There is no model in this loop. The opt-in groups are all deterministic, so
the only arm worth scoring is the detectors themselves, and the result is
reproducible offline in seconds.

**On measuring a property rather than a defect.** One of these checks does not
fit defect-injection scoring at all, and pretending otherwise would produce a
precision figure of 21% that means nothing. `undetectable_at_imgsz` reports
boxes too small for the network's finest stride, and COCO genuinely contains
hundreds of those before anything is injected - they are a true property of
the data, not a false positive. So the base rate on the provably clean corpus
is measured first, and a finding that the clean corpus also produces is
counted as neither a hit nor a miss. That number is reported alongside, since
it is the interesting one for that check.

    python eval/run_extended.py --out runs/extended
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import EXTENDED_CASES                       # noqa: E402
from injector import build_case, DATASET_LEVEL        # noqa: E402

from dsdoctor.dataset import Dataset                   # noqa: E402
from dsdoctor.sweep import sweep                       # noqa: E402

DEFAULT_IMGSZ = 640


def facts(root: Path, groups: list[str], imgsz: int = DEFAULT_IMGSZ) -> set:
    ds = Dataset(root)
    ds.imgsz = imgsz
    res = sweep(ds, groups=groups)
    return {(f.type, k) for f in res.findings for k in (f.items or [DATASET_LEVEL])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus_clean")
    ap.add_argument("--cases-dir", default="data/cases_extended")
    ap.add_argument("--out", default="runs/extended")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = ap.parse_args(argv)

    base = Path(args.corpus)
    if not base.is_dir():
        print(f"no corpus at {base}. Run eval/build_corpus.py first.",
              file=sys.stderr)
        return 2

    groups = sorted({g for c in EXTENDED_CASES for g in c["groups"]})
    base_by_type: dict[str, set] = {}
    for t, k in facts(base, groups, args.imgsz):
        base_by_type.setdefault(t, set()).add(k)

    print("base rate on the provably clean corpus")
    print("  (true properties of the data, not defects, and not false positives)")
    for t, v in sorted(base_by_type.items()):
        print(f"    {t:26s} {len(v):4d} file(s)")
    print()

    report = {"imgsz": args.imgsz,
              "base_rate_on_clean_corpus":
                  {t: len(v) for t, v in sorted(base_by_type.items())},
              "cases": {}}

    print(f"{'case':24s}{'recall':>10s}{'spurious':>10s}{'base-rate':>11s}")
    for case in EXTENDED_CASES:
        out = Path(args.cases_dir) / case["name"]
        gt = build_case(base, out, case["recipe"], seed=case["seed"])
        truth = {(t, k) for t, k in gt["ground_truth"] if t in case["recipe"]}
        got = {x for x in facts(out, case["groups"], args.imgsz)
               if x[0] in case["recipe"]}
        baseline = {x for x in got if x[1] in base_by_type.get(x[0], ())}

        tp = len(truth & got)
        fn = len(truth - got)
        fp = len(got - truth - baseline)
        report["cases"][case["name"]] = {
            "groups": case["groups"], "recipe": case["recipe"],
            "injected": len(truth), "detected": tp, "missed": fn,
            "spurious": fp, "explained_by_base_rate": len(baseline),
            "missed_examples": sorted(truth - got)[:5],
            "spurious_examples": sorted(got - truth - baseline)[:5],
        }
        print(f"{case['name']:24s}{f'{tp}/{tp + fn}':>10s}{fp:>10d}"
              f"{len(baseline):>11d}")

    tp = sum(c["detected"] for c in report["cases"].values())
    fn = sum(c["missed"] for c in report["cases"].values())
    fp = sum(c["spurious"] for c in report["cases"].values())
    report["total"] = {"detected": tp, "missed": fn, "spurious": fp,
                       "recall": round(tp / max(tp + fn, 1), 4)}
    print(f"\ntotal: {tp}/{tp + fn} injected defects found "
          f"({tp / max(tp + fn, 1):.0%}), {fp} spurious finding(s)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(json.dumps(report, indent=2))
    print(f"written to {out_dir / 'scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
