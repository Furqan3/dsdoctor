"""Command line entry point.

    dsdoctor scan       <dataset>              deterministic checks, no model
    dsdoctor audit      <dataset> --out DIR    agent audit -> report + fix plan
    dsdoctor apply      <fix_plan.json>        apply the plan, after approval
    dsdoctor card       <dataset>              health card + content fingerprint
    dsdoctor verify-card <dataset>             does the card describe this data?
    dsdoctor recheck    <dataset> --against H  what changed since a health card
    dsdoctor diff       <a> <b>                compare two datasets
    dsdoctor resplit    <dataset> --out DIR    propose a leak-free split
    dsdoctor convert    <dataset> --out DIR    COCO / Pascal VOC -> YOLO
    dsdoctor detectors                         what checks exist, and in which group

Exit codes are meant for pipelines:

    0  fine
    1  the --fail-on threshold was met
    2  usage or runtime error
    3  a health card does not match the dataset
    4  audit could not reach a model and fell back to a deterministic scan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import detectors
from .dataset import Dataset
from .findings import sort_findings, CRITICAL, MAJOR, GOVERNANCE

EXIT_OK, EXIT_GATE, EXIT_ERROR, EXIT_MISMATCH, EXIT_DEGRADED = 0, 1, 2, 3, 4


# ------------------------------------------------------------------ helpers

def _groups(args) -> list[str]:
    return detectors.resolve_groups(getattr(args, "checks", None))


def _apply_training_config(ds, args) -> None:
    """Attach the training parameters the `training` group needs.

    These are not properties of the dataset, which is why they arrive as flags
    rather than being read off disk - the same dataset is fine at one input
    resolution and half-wasted at another.
    """
    imgsz = getattr(args, "imgsz", None)
    max_det = getattr(args, "max_det", None)
    if imgsz:
        ds.imgsz = imgsz
    if max_det:
        ds.max_det = max_det


def _open(path, *, allow_convert: bool = True):
    """Load a dataset in any supported layout, reporting any conversion.

    The returned Dataset may point at a converted view in a cache directory.
    Anything that describes the dataset *to someone else* - the health card
    above all - must use `_source_root` instead, or it will describe a file
    tree this tool generated rather than the one that was delivered.
    """
    if not allow_convert:
        return Dataset(path)
    from .formats import load_any, detect
    fmt = detect(path)
    if fmt in (None, "yolo"):
        return Dataset(path)     # let Dataset raise its own clear error
    ds, report = load_any(path)
    ds.source_root = Path(path).resolve()
    print(f"detected {report['format'].upper()} layout; reading it through a "
          f"YOLO view at {report['output']}")
    print(f"  {report['classes']} classes, {report['annotations']} annotations, "
          + ", ".join(f"{k} {v['images']} images"
                      for k, v in report["splits"].items()))
    if report.get("unknown_category_ids"):
        print(f"  note: category id(s) {report['unknown_category_ids']} are used "
              f"by annotations but never declared; kept as out-of-range ids so "
              f"class_scan reports them")
    return ds


def _source_root(ds) -> Path:
    """Where the data actually came from, converted view or not."""
    return getattr(ds, "source_root", None) or ds.root


def _print_findings(found: list) -> None:
    train = [f for f in found if f.category != GOVERNANCE]
    gov = [f for f in found if f.category == GOVERNANCE]
    for f in sort_findings(train):
        print(f.short())
    if gov:
        print("\n-- governance (does not affect the trainability verdict) --")
        for f in sort_findings(gov):
            print(f.short())


# -------------------------------------------------------------------- scan

def cmd_scan(args) -> int:
    from .sweep import sweep, should_fail
    from . import output

    ds = _open(args.dataset)
    groups = _groups(args)
    _apply_training_config(ds, args)
    quiet = args.format != "text"

    if not quiet:
        s = ds.summary()
        print(f"{s['root']}: {sum(v['images'] for v in s['splits'].values())} "
              f"images, {s['total_boxes']} boxes, {s['nc']} classes")

    from .progress import spinner

    t0 = time.time()
    with spinner("scanning", enabled=not quiet) as note:
        res = sweep(ds, groups=groups, experimental=args.experimental,
                    on_start=lambda d: note(d.name),
                    on_done=None if quiet else
                    (lambda d, f, secs: print(f"  {d.name:26s} {len(f):3d} "
                                              f"finding(s)  {secs:5.1f}s")))
    elapsed = time.time() - t0

    for name, err in res.failed.items():
        print(f"warning: detector {name} failed: {err}", file=sys.stderr)

    if args.format == "json":
        print(output.to_json(ds, res, elapsed=elapsed))
    elif args.format == "sarif":
        print(output.to_sarif(ds, res))
    else:
        print()
        _print_findings(res.findings)
        print(f"\n{len(res.findings)} finding(s), {res.count(CRITICAL)} critical")
        print("Note: this is the raw detector output, unfiltered. "
              "`dsdoctor audit` triages it.")

    if args.html:
        from . import htmlreport
        Path(args.html).write_text(htmlreport.render(
            ds, res.findings, detectors_run=res.detectors_run, elapsed=elapsed))
        if not quiet:
            print(f"  html: {args.html}")

    if args.out:
        Path(args.out).write_text(output.to_json(ds, res, elapsed=elapsed))
        if not quiet:
            print(f"  json: {args.out}")

    return EXIT_GATE if should_fail(res, args.fail_on) else EXIT_OK


# ------------------------------------------------------------------- audit

def cmd_audit(args) -> int:
    from .llm import LLM, EndpointUnavailable
    from .agent import audit
    from .report import write_all

    ds = _open(args.dataset)
    groups = _groups(args)
    kw = {}
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.model:
        kw["model"] = args.model
    llm = LLM(**kw)

    try:
        llm.preflight()
    except EndpointUnavailable as exc:
        # The deterministic half of this tool needs no model at all, so the
        # useful response to a missing endpoint is the scan, not a stack trace.
        print(f"{exc}\n", file=sys.stderr)
        print("Set DSDOCTOR_BASE_URL / DSDOCTOR_MODEL (or pass --base-url and "
              "--model) to point at any OpenAI-compatible endpoint.\n"
              "Running the deterministic checks instead - they need no model, "
              "and they are where every finding comes from anyway.\n",
              file=sys.stderr)
        args.format, args.fail_on, args.out = "text", None, ""
        cmd_scan(args)
        return EXIT_DEGRADED

    t0 = time.time()
    print(f"auditing {ds.root} with {llm.model} ...")
    res = audit(ds, llm, experimental=args.experimental,
                verify=not args.no_verify, groups=groups)
    elapsed = time.time() - t0

    out = Path(args.out or (Path.cwd() / "audit_out"))
    paths = write_all(res, ds, out, model=llm.model, elapsed=elapsed)

    if args.html:
        from . import htmlreport
        html_path = Path(args.html)
        html_path.write_text(htmlreport.render(
            ds, [d.finding for d in res.reported], verdict=res.verdict,
            headline=res.headline, detectors_run=res.detectors_run,
            model=llm.model, elapsed=elapsed))
        paths["html"] = html_path

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
    return EXIT_OK


# ------------------------------------------------------------------- apply

def cmd_apply(args) -> int:
    from .apply import apply_plan
    apply_plan(args.plan, assume_yes=args.yes)
    return EXIT_OK


# -------------------------------------------------------------------- card

def cmd_card(args) -> int:
    from .sweep import sweep
    from . import card

    ds = _open(args.dataset)
    groups = _groups(args)
    _apply_training_config(ds, args)
    res = sweep(ds, groups=groups, experimental=args.experimental)
    src = _source_root(ds)
    out = Path(args.out) if args.out else src
    paths = card.write(ds, res.findings, out, groups=groups,
                       detectors_run=res.detectors_run, source_root=src)
    if src != ds.root:
        print(f"read through a converted view; the card describes and "
              f"fingerprints the original at {src}")
    health = json.loads(paths["health"].read_text())
    print(f"verdict: {health['verdict']}  "
          f"({health['summary']['critical']} critical, "
          f"{health['summary']['major']} major, "
          f"{health['summary']['governance']} governance)")
    print(f"fingerprint: {health['fingerprint']['digest'][:24]}… over "
          f"{health['fingerprint']['files']:,} files")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return EXIT_OK


def cmd_verify_card(args) -> int:
    from . import card

    ds = Dataset(args.dataset)
    hp = Path(args.card) if args.card else ds.root / "health.json"
    # `Dataset` on a COCO/VOC delivery parses no samples, but verification only
    # needs the file tree, so it works on any layout.
    if not hp.is_file():
        print(f"no health card at {hp}. Run `dsdoctor card {args.dataset}` "
              f"to create one.", file=sys.stderr)
        return EXIT_ERROR
    health = json.loads(hp.read_text())
    result = card.verify(ds, health, hp.parent / "dataset_manifest.tsv")

    if result["match"]:
        print(f"MATCH — this card describes the dataset in front of you.")
        print(f"  digest {result['current_digest'][:24]}… over "
              f"{result['current_files']:,} files")
        print(f"  card written {health.get('generated', 'unknown')}, "
              f"verdict {health.get('verdict', 'unknown')}")
        return EXIT_OK

    print("MISMATCH — the dataset has changed since this card was written.")
    print(f"  card says   {result['claimed_digest'][:24]}… "
          f"({result['claimed_files']} files)")
    print(f"  actual      {result['current_digest'][:24]}… "
          f"({result['current_files']} files)")
    for label in ("added", "removed", "modified"):
        rows = result[label]
        if rows:
            print(f"\n  {len(rows)} {label}:")
            for r in rows[:10]:
                print(f"    {r}")
            if len(rows) > 10:
                print(f"    … and {len(rows) - 10} more")
    if not any(result[k] for k in ("added", "removed", "modified")):
        print("\n  (no manifest alongside the card, so the digests can be "
              "compared but the changes cannot be listed)")
    return EXIT_MISMATCH


def cmd_recheck(args) -> int:
    """What changed since a health card - the loop `apply` never closed."""
    from .sweep import sweep
    from . import card

    ds = _open(args.dataset)
    hp = Path(args.against) if args.against else ds.root / "health.json"
    if not hp.is_file():
        print(f"no health card at {hp}. Run `dsdoctor card {args.dataset}` "
              f"first, so there is something to compare against.",
              file=sys.stderr)
        return EXIT_ERROR
    before = json.loads(hp.read_text())

    groups = detectors.resolve_groups(args.checks) if args.checks else [
        g for g in before.get("checks", {}).get("groups", []) if g != "core"]
    res = sweep(ds, groups=groups, experimental=args.experimental)
    # rows=[] on purpose: recheck compares findings, and the fingerprint it
    # would otherwise compute is never printed. On the 600-image corpus that
    # hashing was 0.22s of a 2.6s command - 8%, not the dominant cost, and the
    # first measurement of it claimed otherwise by timing the whole command.
    # It is worth removing anyway, and more so at scale than the small figure
    # suggests: the sweep's own image hashing is cached on (path, size, mtime)
    # across runs, while a fingerprint is not cacheable by construction - it
    # has to re-read every byte to mean anything.
    after = card.build(ds, res.findings, groups=groups, rows=[],
                       detectors_run=res.detectors_run,
                       source_root=_source_root(ds))

    def index(h):
        out = {}
        for f in h["findings"] + h["governance_findings"]:
            for item in f["items"] or [f["type"]]:
                out[(f["type"], item)] = f
        return out

    b, a = index(before), index(after)
    resolved = sorted(set(b) - set(a))
    introduced = sorted(set(a) - set(b))
    remaining = sorted(set(a) & set(b))

    print(f"comparing against {hp}")
    print(f"  card written {before.get('generated')}, verdict "
          f"{before.get('verdict')}")
    print(f"  now                             verdict {after['verdict']}")
    print()
    print(f"  resolved:   {len(resolved)}")
    print(f"  remaining:  {len(remaining)}")
    print(f"  introduced: {len(introduced)}")

    for label, rows in (("resolved", resolved), ("introduced", introduced)):
        if rows:
            print(f"\n{label}:")
            for typ, item in rows[:15]:
                print(f"  {typ:24s} {item}")
            if len(rows) > 15:
                print(f"  … and {len(rows) - 15} more")

    if introduced:
        print("\nSomething got worse. If this ran after `dsdoctor apply`, "
              "the backup directory it printed holds the originals.")
        return EXIT_GATE
    return EXIT_OK


# -------------------------------------------------------------------- diff

def cmd_diff(args) -> int:
    from .sweep import sweep
    from . import card

    a, b = _open(args.a), _open(args.b)
    groups = _groups(args)
    ra, rb = (sweep(a, groups=groups, experimental=args.experimental),
              sweep(b, groups=groups, experimental=args.experimental))
    ca = card.build(a, ra.findings, groups=groups, rows=[],
                    source_root=_source_root(a))
    cb = card.build(b, rb.findings, groups=groups, rows=[],
                    source_root=_source_root(b))

    print(f"A  {a.root}")
    print(f"B  {b.root}\n")
    print(f"{'':28s} {'A':>10s} {'B':>10s} {'Δ':>10s}")
    rows = [("images", sum(v['images'] for v in ca['dataset']['splits'].values()),
             sum(v['images'] for v in cb['dataset']['splits'].values())),
            ("boxes", ca["dataset"]["total_boxes"], cb["dataset"]["total_boxes"]),
            ("classes", ca["dataset"]["num_classes"], cb["dataset"]["num_classes"]),
            ("critical", ca["summary"]["critical"], cb["summary"]["critical"]),
            ("major", ca["summary"]["major"], cb["summary"]["major"]),
            ("minor", ca["summary"]["minor"], cb["summary"]["minor"]),
            ("governance", ca["summary"]["governance"], cb["summary"]["governance"])]
    for name, x, y in rows:
        d = y - x
        print(f"{name:28s} {x:10d} {y:10d} {d:+10d}" if d else
              f"{name:28s} {x:10d} {y:10d} {'—':>10s}")

    ta = {f.type for f in ra.findings}
    tb = {f.type for f in rb.findings}
    if tb - ta:
        print(f"\nonly in B: {', '.join(sorted(tb - ta))}")
    if ta - tb:
        print(f"only in A: {', '.join(sorted(ta - tb))}")
    print(f"\nverdict  A: {ca['verdict']}   B: {cb['verdict']}")
    return EXIT_OK


# ----------------------------------------------------------------- resplit

def cmd_resplit(args) -> int:
    from . import resplit

    ds = _open(args.dataset)
    print(f"grouping {len(ds.samples)} sample(s) by near-duplicate content ...")
    proposal = resplit.propose(ds, val_fraction=args.val_fraction)
    print(f"  {proposal['groups']} duplicate group(s), largest holds "
          f"{proposal['largest_group']} image(s)")
    print(f"  val fraction {proposal['val_fraction_achieved']:.3f} "
          f"(requested {args.val_fraction})")
    if proposal["classes_missing_from_val"]:
        print(f"  warning: {len(proposal['classes_missing_from_val'])} class(es) "
              f"still absent from val: "
              f"{', '.join(proposal['classes_missing_from_val'][:8])}")

    out = Path(args.out) if args.out else None
    manifest = Path(args.manifest) if args.manifest else (
        (out / "split_proposal.json") if out else Path("split_proposal.json"))
    if out:
        written = resplit.materialise(ds, proposal, out)
        print(f"\nwrote {written} to {out} (symlinks; the source is untouched)")
        check = resplit.verify(out)
        print(f"  verified: leak_free={check['leak_free']}, "
              f"{check['leaked_pairs']} leaked pair(s) across "
              f"{check['images']} images")
        manifest.parent.mkdir(parents=True, exist_ok=True)
    resplit.write(ds, proposal, manifest)
    print(f"  proposal: {manifest}")
    return EXIT_OK


# ------------------------------------------------------------------- merge

def cmd_merge(args) -> int:
    from . import merge

    roots = args.datasets
    if len(roots) < 2:
        print("merge needs at least two datasets", file=sys.stderr)
        return EXIT_ERROR

    print(f"planning a merge of {len(roots)} dataset(s) ...")
    plan_doc = merge.plan(roots)
    print()
    print(merge.summarise(plan_doc))

    manifest = Path(args.manifest) if args.manifest else None
    if args.out:
        out = Path(args.out)
        if plan_doc["needs_a_decision"] and not args.force:
            print(f"\nNot writing. The name conflicts above are a relabelling "
                  f"decision, not a merge step: resolve them in the sources, "
                  f"or pass --force to merge with those classes kept separate.")
            return EXIT_GATE
        written = merge.materialise(roots, plan_doc, out)
        print(f"\nwrote {dict(written)} to {out} "
              f"(symlinked images, remapped labels; sources untouched)")
        manifest = manifest or (out / "merge_plan.json")
    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        merge.write(plan_doc, manifest)
        print(f"  plan: {manifest}")
    return EXIT_OK


# ----------------------------------------------------------------- convert

def cmd_convert(args) -> int:
    from .formats import convert, detect

    fmt = detect(args.dataset)
    if fmt is None:
        print(f"could not identify a dataset layout at {args.dataset}",
              file=sys.stderr)
        return EXIT_ERROR
    if fmt == "yolo":
        print(f"{args.dataset} is already in YOLO layout; nothing to convert.")
        return EXIT_OK
    out, report = convert(args.dataset, args.out, fmt)
    print(f"converted {report['format'].upper()} -> YOLO at {out}")
    print(f"  {report['classes']} classes, {report['annotations']} annotations")
    for split, v in report["splits"].items():
        print(f"  {split}: {v['images']} images, {v['annotations']} annotations")
    if report.get("unknown_category_ids"):
        print(f"  {len(report['unknown_category_ids'])} undeclared category "
              f"id(s) kept as out-of-range class ids: "
              f"{report['unknown_category_ids'][:10]}")
    if report.get("images_without_size"):
        print(f"  {report['images_without_size']} image(s) declared no size; "
              f"their coordinates were left unnormalised so the scan reports it")
    if report.get("malformed_xml"):
        print(f"  {len(report['malformed_xml'])} unreadable XML file(s)")
    return EXIT_OK


# --------------------------------------------------------------- detectors

def cmd_detectors(args) -> int:
    loaded = detectors.load_plugins()
    print(f"{'detector':28s} {'group':10s} {'cost':7s} detects")
    for group in detectors.ALL_GROUPS:
        for d in detectors.REGISTRY.values():
            if d.group != group:
                continue
            cost = "slow" if d.heavy else ("medium" if d.reads_pixels else "fast")
            tag = f"{d.group}{'*' if d.experimental else ''}"
            print(f"{d.name:28s} {tag:10s} {cost:7s} "
                  f"{', '.join(d.covers) or '(informational)'}")
    print(f"\ncore runs by default. Add others with --checks:")
    for name, why in detectors.EXTRA_GROUPS.items():
        print(f"  --checks {name:10s} {why}")
    print("  --checks all        everything above")
    print("\n* experimental: measured as net-harmful, needs --experimental")
    if loaded:
        print(f"\nplugins loaded: {', '.join(loaded)}")
    else:
        print("\nno detector plugins installed. A package can add its own "
              "checks by advertising a `dsdoctor.detectors` entry point.")
    return EXIT_OK


# --------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsdoctor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def checks_flag(p):
        p.add_argument("--checks", default=None, metavar="GROUPS",
                       help="extra check groups, comma separated: "
                            + ", ".join(detectors.EXTRA_GROUPS) + ", or 'all'")
        p.add_argument("--experimental", action="store_true",
                       help="also run detectors measured as net-harmful "
                            "(currently: model_disagreement_scan)")
        p.add_argument("--imgsz", type=int, default=None,
                       help="input resolution you intend to train at; used by "
                            "--checks training (default 640)")
        p.add_argument("--max-det", type=int, default=None,
                       help="max detections per image at validation; used by "
                            "--checks training (default 300)")

    p = sub.add_parser("scan", help="run the deterministic detectors only")
    p.add_argument("dataset")
    checks_flag(p)
    p.add_argument("--format", choices=("text", "json", "sarif"), default="text",
                   help="sarif is for CI: GitHub renders it against the files")
    p.add_argument("--fail-on", choices=("critical", "major", "minor", "any"),
                   default=None, help="exit 1 when a finding at or above this "
                                      "severity exists (governance findings "
                                      "never fail the gate)")
    p.add_argument("--html", default="", metavar="FILE",
                   help="also write a self-contained visual report")
    p.add_argument("--out", default="", metavar="FILE", help="write JSON here")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("audit", help="full agent audit")
    p.add_argument("dataset")
    checks_flag(p)
    p.add_argument("--out", default="")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint (default: $DSDOCTOR_BASE_URL, "
                        "else a local vLLM at :8000)")
    p.add_argument("--model", default=None, help="default: $DSDOCTOR_MODEL")
    p.add_argument("--html", default="", metavar="FILE",
                   help="also write a self-contained visual report")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the verification pass over suppressions")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("apply", help="apply an approved fix plan")
    p.add_argument("plan")
    p.add_argument("--yes", action="store_true",
                   help="skip the interactive confirmation (for scripted use)")
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("card", help="write a health card that can travel with "
                                    "the dataset")
    p.add_argument("dataset")
    checks_flag(p)
    p.add_argument("--out", default="", help="default: the dataset directory")
    p.set_defaults(fn=cmd_card)

    p = sub.add_parser("verify-card",
                       help="check a health card still describes this dataset")
    p.add_argument("dataset")
    p.add_argument("--card", default="", help="default: <dataset>/health.json")
    p.set_defaults(fn=cmd_verify_card)

    p = sub.add_parser("recheck",
                       help="what changed since a health card was written")
    p.add_argument("dataset")
    checks_flag(p)
    p.add_argument("--against", default="", help="default: <dataset>/health.json")
    p.set_defaults(fn=cmd_recheck)

    p = sub.add_parser("diff", help="compare two datasets or two versions")
    p.add_argument("a")
    p.add_argument("b")
    checks_flag(p)
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("resplit", help="propose a train/val split that cannot leak")
    p.add_argument("dataset")
    p.add_argument("--out", default="", help="materialise the split here "
                                             "(symlinks; source untouched)")
    p.add_argument("--manifest", default="", help="where to write the proposal")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.set_defaults(fn=cmd_resplit)

    p = sub.add_parser("merge", help="merge datasets, reporting class "
                                     "conflicts before they corrupt anything")
    p.add_argument("datasets", nargs="+")
    p.add_argument("--out", default="", help="materialise the merge here")
    p.add_argument("--manifest", default="", help="where to write the plan")
    p.add_argument("--force", action="store_true",
                   help="merge even when class names need a human decision")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("convert", help="COCO or Pascal VOC -> YOLO")
    p.add_argument("dataset")
    p.add_argument("--out", default="", help="default: a cache directory")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("detectors", help="list every check and its group")
    p.set_defaults(fn=cmd_detectors)

    args = ap.parse_args(argv)
    detectors.load_plugins()
    try:
        return args.fn(args)
    except ValueError as exc:            # bad --checks, bad --fail-on
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
