"""Split integrity: is the train/val division capable of measuring anything?

`class_scan` asks whether a class can be *learned*. These checks ask the
adjacent question - whether the split can *report* on what was learned. A
class with three hundred training instances and none in val is not a training
problem at all; the model will learn it fine. It is a measurement problem, and
it is invisible in the headline number, because most frameworks fold an
undefined per-class AP into the mean as a zero and the team reads the result
as "the model is bad at that class" rather than "we never tested it".
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..dataset import Dataset, split_role
from ..findings import Finding, MAJOR
from . import register

# Below this, a val split cannot resolve much: at 2% of 600 images a single
# image moves mAP by a visible amount. Above the upper bound you are usually
# looking at a split that was never done, or was done the wrong way round.
MIN_VAL_FRACTION = 0.05
MAX_VAL_FRACTION = 0.50


def _counts_by_role(ds: Dataset) -> dict[str, Counter]:
    per_role: dict[str, Counter] = defaultdict(Counter)
    for s in ds.samples:
        if not s.label:
            continue
        role = split_role(s.split)
        for b in s.label.boxes:
            per_role[role][b.cls] += 1
    return per_role


@register("split_scan",
          "Check that the train/val division can actually measure the model: "
          "classes present in train but absent from val, and split ratios too "
          "extreme to validate against. Detects class_absent_from_val and "
          "split_ratio_extreme.",
          covers=("class_absent_from_val", "split_ratio_extreme"),
          group="split")
def split_scan(ds: Dataset) -> list[Finding]:
    out: list[Finding] = []
    per_role = _counts_by_role(ds)
    train, val = per_role.get("train", Counter()), per_role.get("val", Counter())

    # --- classes that exist in train and nowhere in val ------------------
    #
    # `class_scan` also mentions thin val classes as one reason among several
    # inside extreme_class_imbalance. This isolates the qualitatively worse
    # case - not "thin", but "undefined" - because the remedy differs: a thin
    # class needs more val data, an absent one needs the split redone.
    if train and val is not None:
        absent = [c for c, n in sorted(train.items()) if n > 0 and val.get(c, 0) == 0]
        if absent and val:  # a val split exists but omits these classes
            ev = [f"'{ds.class_name(c)}' (id {c}): {train[c]} instance(s) in "
                  f"train, 0 in val" for c in absent]
            out.append(Finding(
                type="class_absent_from_val", severity=MAJOR,
                title=f"{len(absent)} class(es) appear in train but never in val",
                detail="Average precision is undefined for a class with no "
                       "ground-truth instances in the validation set. Most "
                       "frameworks fold that undefined value into the mean as "
                       "a zero, so the headline mAP is dragged down by classes "
                       "that were never actually evaluated - and the team reads "
                       "that as a model weakness rather than a split defect. "
                       "Re-split so every class is represented on both sides.",
                detector="split_scan", items=[],
                evidence=ev[:15],
                fix={"action": "restratify_split", "targets": [],
                     "class_ids": absent}))

    # --- a split ratio that cannot support a decision --------------------
    n_train = sum(1 for s in ds.samples if split_role(s.split) == "train"
                  and s.image_path)
    n_val = sum(1 for s in ds.samples if split_role(s.split) == "val"
                and s.image_path)
    total = n_train + n_val
    if total >= 20:  # below this the ratio is not a meaningful statistic
        frac = n_val / total
        problem = ""
        if n_val == 0:
            problem = ("there is no validation split at all, so nothing "
                       "measures whether training worked")
        elif frac < MIN_VAL_FRACTION:
            problem = (f"only {frac:.1%} of images are in val ({n_val} of "
                       f"{total}); a single image moves mAP by "
                       f"{100 / max(n_val, 1):.1f}% of the per-image weight")
        elif frac > MAX_VAL_FRACTION:
            problem = (f"{frac:.1%} of images are in val ({n_val} of {total}), "
                       f"which usually means the split was never done or was "
                       f"written the wrong way round")
        if problem:
            out.append(Finding(
                type="split_ratio_extreme", severity=MAJOR,
                title="the train/val ratio cannot support a training decision",
                detail=f"Found {n_train} train and {n_val} val image(s): "
                       f"{problem}. A validation set exists to answer one "
                       f"question - would this model work on data it has not "
                       f"seen - and at this ratio the answer carries an error "
                       f"bar wider than the differences you will be comparing.",
                detector="split_scan", items=[],
                evidence=[f"train images: {n_train}", f"val images: {n_val}",
                          f"val fraction: {frac:.3f}",
                          f"usual range: {MIN_VAL_FRACTION:.0%}-{MAX_VAL_FRACTION:.0%}"],
                fix={"action": "restratify_split", "targets": []}))

    return out
