"""Command line entry point.

    dsdoctor scan  <dataset>              deterministic checks only, no model
    dsdoctor audit <dataset> --out DIR    full agent audit -> report + fix plan
    dsdoctor apply <fix_plan.json>        apply the plan, after you approve it
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .dataset import Dataset
from .findings import sort_findings, CRITICAL, MAJOR


def cmd_scan(args) -> int:
    from . import detectors
    ds = Dataset(args.dataset)
    s = ds.summary()
    print(f"{s['root']}: {sum(v['images'] for v in s['splits'].values())} images, "
          f"{s['total_boxes']} boxes, {s['nc']} classes")
    found = []
    for det in detectors.available(include_experimental=args.experimental):
        if not det.covers:
            continue
        t0 = time.time()
        got = det.fn(ds)
        found += got
        print(f"  {det.name:26s} {len(got):3d} finding(s)  {time.time() - t0:5.1f}s")
    print()
    for f in sort_findings(found):
        print(f.short())
    crit = sum(1 for f in found if f.severity == CRITICAL)
    print(f"\n{len(found)} finding(s), {crit} critical")
    print("Note: this is the raw detector output, unfiltered. `dsdoctor audit` "
          "triages it.")
    return 0


def cmd_audit(args) -> int:
    from .llm import LLM
    from .agent import audit
    from .report import write_all

    ds = Dataset(args.dataset)
    kw = {}
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.model:
        kw["model"] = args.model
    llm = LLM(**kw)

    t0 = time.time()
    print(f"auditing {ds.root} with {llm.model} ...")
    res = audit(ds, llm, experimental=args.experimental,
                verify=not args.no_verify)
    elapsed = time.time() - t0

    out = Path(args.out or (Path.cwd() / "audit_out"))
    paths = write_all(res, ds, out, model=llm.model, elapsed=elapsed)

    crit = sum(1 for d in res.reported if d.severity == CRITICAL)
    major = sum(1 for d in res.reported if d.severity == MAJOR)
    print(f"\nverdict: {res.verdict}")
    print(f"{crit} critical, {major} major, {len(res.suppressed)} suppressed"
          + (f", {res.reinstated_total} reinstated by the verifier"
             if res.reinstated_total else ""))
    print(f"{res.trajectory.llm_calls} model call(s), "
          f"{res.trajectory.tool_calls} tool call(s), {elapsed:.0f}s")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


def cmd_apply(args) -> int:
    from .apply import apply_plan
    apply_plan(args.plan, assume_yes=args.yes)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsdoctor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="run the deterministic detectors only")
    p.add_argument("dataset")
    p.add_argument("--experimental", action="store_true",
                   help="also run detectors measured as net-harmful "
                        "(currently: model_disagreement_scan)")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("audit", help="full agent audit")
    p.add_argument("dataset")
    p.add_argument("--out", default="")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint (default: local vLLM)")
    p.add_argument("--model", default=None)
    p.add_argument("--experimental", action="store_true",
                   help="also run detectors measured as net-harmful "
                        "(currently: model_disagreement_scan)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the verification pass over suppressions")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("apply", help="apply an approved fix plan")
    p.add_argument("plan")
    p.add_argument("--yes", action="store_true",
                   help="skip the interactive confirmation (for scripted use)")
    p.set_defaults(fn=cmd_apply)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
