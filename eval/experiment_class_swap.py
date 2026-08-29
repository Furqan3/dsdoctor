"""Reproduce the retired experiment: can a reference detector find class swaps?

This script exists so that the decision recorded in the README's changelog
(iteration 4 - added a vision-based mislabel detector, measured it, removed it)
can be checked rather than taken on trust.

It injects a known class swap into several cases, runs the experimental
`model_disagreement_scan` at a range of operating points, and reports precision
and recall at the file level against the injected ground truth.

    pip install -e ".[vision]"
    python eval/experiment_class_swap.py

Expect it to confirm that there is no threshold worth shipping: high recall
comes with single-digit precision, and usable precision comes with a third of
the recall. It also reports what the detector claims on a corpus with no swaps
in it at all, which is the number that actually decided the question.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from injector import build_case                       # noqa: E402
from dsdoctor.dataset import Dataset, file_sha1       # noqa: E402
from dsdoctor.detectors import consistency as C       # noqa: E402

# The operating points swept. The shipped defaults were the first row.
GRID = [
    # (conf, iou, min_boxes_in_group, min_disagreement_rate)
    (0.55, 0.60, 2, 0.60),
    (0.25, 0.45, 2, 0.60),
    (0.10, 0.35, 2, 0.60),
    (0.10, 0.35, 2, 0.75),
    (0.10, 0.35, 2, 1.00),
    (0.10, 0.35, 3, 0.75),
    (0.10, 0.35, 3, 0.90),
    (0.10, 0.35, 3, 1.00),
]

CASES = [("swap_a", {"class_swap": 6}, 701),
         ("swap_b", {"class_swap": 5}, 702),
         ("swap_c", {"class_swap": 4}, 703),
         ("no_swap_a", {}, 704),
         ("no_swap_b", {"out_of_bounds": 3}, 705)]


def predictions(model, ref_names, path: Path, conf: float, cache: dict) -> list:
    key = f"{file_sha1(path)}:{conf}"
    if key not in cache:
        r = model.predict(str(path), conf=conf, verbose=False)[0]
        w, h = r.orig_shape[1], r.orig_shape[0]
        cache[key] = [[float(b.xyxy[0][0]) / w, float(b.xyxy[0][1]) / h,
                       float(b.xyxy[0][2]) / w, float(b.xyxy[0][3]) / h,
                       ref_names.get(int(b.cls[0]), "?"), float(b.conf[0])]
                      for b in r.boxes]
    return [((q[0], q[1], q[2], q[3]), q[4], q[5]) for q in cache[key]]


def groups(ds: Dataset, model, ref_names, conf: float, iou: float, cache: dict):
    """(file, labelled class) -> (matched boxes, disagreeing boxes)."""
    ours = set(ds.names)
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for s in ds.samples:
        if not (s.image_path and s.label and s.label.boxes):
            continue
        try:
            preds = predictions(model, ref_names, s.image_path, conf, cache)
        except Exception:
            continue
        per = defaultdict(lambda: [0, 0])
        for b in s.label.boxes:
            gtn = ds.class_name(b.cls)
            best, best_iou = None, 0.0
            for pxy, pn, pc in preds:
                v = C._iou(b.xyxy, pxy)
                if v > best_iou:
                    best, best_iou = pn, v
            if best is None or best_iou < iou or best not in ours:
                continue
            per[gtn][0] += 1
            if best != gtn:
                per[gtn][1] += 1
        for cls, (tot, dis) in per.items():
            out[(s.key(), cls)] = (tot, dis)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_clean")
    ap.add_argument("--workdir", default="data/cases_swap")
    ap.add_argument("--out", default="runs/experiment-class-swap.json")
    args = ap.parse_args()

    try:
        model = C._load_model("yolov8n.pt")
    except Exception as exc:
        print(f"needs the vision extra: pip install -e '.[vision]'  ({exc})")
        return 1
    ref_names = {i: n for i, n in model.names.items()}

    base = Path(args.corpus)
    work = Path(args.workdir)
    cache: dict = {}
    built = []
    for name, recipe, seed in CASES:
        d = work / name
        gt = build_case(base, d, recipe, seed)
        swaps = {k for t, k in gt["ground_truth"] if t == "class_swap"}
        built.append((name, d, swaps))
        print(f"{name}: {len(swaps)} injected swap file(s)")

    # What does it claim when there is nothing to find? This is the number
    # that decided the question.
    print("\n--- on a corpus with no swaps at all ---")
    clean = Dataset(base)
    t0 = time.time()
    clean_findings = C.model_disagreement_scan(clean)
    clean_files = {k for f in clean_findings for k in f.items}
    print(f"{len(clean_findings)} class_swap group(s) claimed across "
          f"{len(clean_files)} file(s) in {time.time() - t0:.0f}s "
          f"- every one of them a false positive")
    for f in clean_findings:
        print(f"   {f.title}")

    print(f"\n--- sweep over {len(GRID)} operating points ---")
    print(f"{'conf':>6}{'iou':>6}{'minbox':>8}{'rate':>7} | "
          f"{'TP':>5}{'FP':>6}{'recall':>9}{'precision':>11}")
    rows = []
    for conf, iou, minbox, rate in GRID:
        tp = fp = total = 0
        for name, d, swaps in built:
            g = groups(Dataset(d), model, ref_names, conf, iou, cache)
            flagged = {k for (k, _), (tot, dis) in g.items()
                       if tot >= minbox and dis / tot >= rate}
            tp += len(flagged & swaps)
            fp += len(flagged - swaps)
            total += len(swaps)
        rec = tp / total if total else 0.0
        pre = tp / (tp + fp) if (tp + fp) else 0.0
        rows.append({"conf": conf, "iou": iou, "min_boxes": minbox,
                     "min_rate": rate, "tp": tp, "fp": fp,
                     "recall": rec, "precision": pre})
        print(f"{conf:>6.2f}{iou:>6.2f}{minbox:>8}{rate:>7.2f} | "
              f"{tp:>5}{fp:>6}{rec:>9.1%}{pre:>11.1%}")

    best_p = max(rows, key=lambda r: r["precision"])
    best_r = max(rows, key=lambda r: r["recall"])
    print(f"\nbest precision: {best_p['precision']:.1%} at "
          f"{best_p['recall']:.1%} recall")
    print(f"best recall:    {best_r['recall']:.1%} at "
          f"{best_r['precision']:.1%} precision")
    print("\nConclusion: no operating point is worth shipping. The detector is "
          "registered as experimental and excluded from the default audit.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "clean_corpus_false_positive_groups": len(clean_findings),
        "clean_corpus_false_positive_files": len(clean_files),
        "clean_corpus_claims": [f.title for f in clean_findings],
        "sweep": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
