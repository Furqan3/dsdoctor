"""Class-level checks: ids that do not exist, and distributions too thin to
train or to validate against."""

from __future__ import annotations

from collections import Counter, defaultdict

from ..dataset import Dataset
from ..findings import Finding, CRITICAL, MAJOR
from . import register

# A class needs enough instances to learn from, and enough in val for the
# reported per-class mAP to mean anything at all.
MIN_TRAIN_INSTANCES = 10
MIN_VAL_INSTANCES = 3
IMBALANCE_RATIO = 50.0


@register("class_scan",
          "Check class ids against data.yaml and measure the per-class, "
          "per-split instance distribution. Detects class_id_out_of_range and "
          "extreme_class_imbalance.",
          covers=("class_id_out_of_range", "extreme_class_imbalance"))
def class_scan(ds: Dataset) -> list[Finding]:
    out: list[Finding] = []
    bad_ids: list[tuple[str, str]] = []
    per_split: dict[str, Counter] = defaultdict(Counter)

    for s in ds.samples:
        if not s.label:
            continue
        for b in s.label.boxes:
            if b.cls < 0 or (ds.nc and b.cls >= ds.nc):
                bad_ids.append((s.key(),
                                f"{s.key()} line {b.line_no}: class id {b.cls} "
                                f"but data.yaml defines {ds.nc} classes (0..{ds.nc - 1})"))
            per_split[s.split][b.cls] += 1

    if bad_ids:
        keys = sorted({k for k, _ in bad_ids})
        out.append(Finding(
            type="class_id_out_of_range", severity=CRITICAL,
            title=f"{len(bad_ids)} box(es) use a class id that does not exist",
            detail="An id at or above nc indexes past the model's classification "
                   "head. Ultralytics raises on this during dataset verification; "
                   "a hand-rolled loader will index out of bounds mid-epoch.",
            detector="class_scan", items=keys,
            evidence=[m for _, m in bad_ids[:12]],
            fix={"action": "remap_or_drop_class_ids", "targets": keys}))

    train_like = next((s for s in ("train", "train2017", "training")
                       if s in per_split), None)
    val_like = next((s for s in ("val", "valid", "val2017", "validation", "test")
                     if s in per_split), None)

    starved: list[str] = []
    starved_ids: list[int] = []
    if train_like:
        counts = per_split[train_like]
        present = {c: n for c, n in counts.items() if n > 0}
        if present:
            busiest = max(present.values())
            for cls in range(ds.nc) if ds.nc else sorted(present):
                n_train = counts.get(cls, 0)
                n_val = per_split[val_like].get(cls, 0) if val_like else None
                reasons = []
                if n_train == 0:
                    reasons.append("0 instances in train")
                elif n_train < MIN_TRAIN_INSTANCES:
                    reasons.append(f"only {n_train} instances in train")
                if n_train and busiest / max(n_train, 1) >= IMBALANCE_RATIO:
                    reasons.append(f"{busiest / n_train:.0f}x rarer than the "
                                   f"most common class")
                if val_like is not None and n_val is not None and n_val < MIN_VAL_INSTANCES:
                    reasons.append(f"only {n_val} instances in {val_like}, so its "
                                   f"per-class mAP is not meaningful")
                if reasons:
                    starved.append(f"'{ds.class_name(cls)}' (id {cls}): "
                                   + "; ".join(reasons))
                    starved_ids.append(cls)

    if starved:
        out.append(Finding(
            type="extreme_class_imbalance", severity=MAJOR,
            title=f"{len(starved)} class(es) have too few instances to train or validate",
            detail="Under-represented classes do not simply score badly, they make "
                   "the headline mAP misleading: a class with two validation "
                   "instances moves the average by whole points on a single "
                   "detection. Decide up front whether to merge, drop or collect "
                   "more of these.",
            detector="class_scan", items=[],
            evidence=starved[:15],
            fix={"action": "rebalance_or_merge_classes", "targets": [],
                 "class_ids": starved_ids}))

    return out


@register("class_distribution",
          "Report the full per-class instance count for every split. Purely "
          "informational - emits no findings, used to reason about balance.",
          covers=())
def class_distribution(ds: Dataset) -> list[Finding]:
    # Exposed as a tool so the agent can look at the numbers without a
    # detector deciding for it what counts as a problem.
    return []
