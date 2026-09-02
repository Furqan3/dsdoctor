"""Checks that depend on how you intend to train, not only on what is on disk.

Every other detector answers a question about the dataset alone. These two need
one number the dataset does not carry - the input resolution you will train at -
and in exchange they answer something no amount of staring at label files will:
how much of this annotation effort the model is structurally unable to use.

A YOLO detection head predicts on feature maps at strides 8, 16 and 32. Stride 8
is the finest, so an object that occupies fewer than a few pixels at your chosen
`imgsz` has no cell that can represent it. It is not "hard"; it is unlearnable at
that resolution, and every one of those instances is counted against recall.

This is the check that changes a decision rather than fixing a file. "6% of your
annotations are invisible at 640 and visible at 1280" is the input to choosing
between them, and it costs a scan rather than two training runs.
"""

from __future__ import annotations

from collections import Counter

from ..dataset import Dataset
from ..findings import Finding, MAJOR
from . import register

# Ultralytics' P3 head. An object needs to span at least one stride-8 cell to
# be representable at all, and roughly two to be learned with any reliability.
FINEST_STRIDE = 8
MIN_CELLS = 1.0

# Ultralytics predicts at most this many boxes per image at validation time.
# An image with more ground-truth objects than this cannot be fully scored.
DEFAULT_MAX_DET = 300

DEFAULT_IMGSZ = 640


@register("training_fit_scan",
          "Measure the dataset against the input resolution you intend to "
          "train at: boxes too small for the finest feature stride, and images "
          "with more objects than max_det can score. Detects "
          "undetectable_at_imgsz and over_max_detections. Takes --imgsz.",
          covers=("undetectable_at_imgsz", "over_max_detections"),
          group="training")
def training_fit_scan(ds: Dataset) -> list[Finding]:
    imgsz = getattr(ds, "imgsz", None) or DEFAULT_IMGSZ
    max_det = getattr(ds, "max_det", None) or DEFAULT_MAX_DET

    # A normalised side maps to imgsz * side pixels once letterboxed.
    min_side = (FINEST_STRIDE * MIN_CELLS) / imgsz

    too_small: list[tuple[str, str]] = []
    n_boxes = 0
    per_class: Counter = Counter()
    crowded: list[tuple[str, str]] = []

    for s in ds.samples:
        if not s.label:
            continue
        if s.label.n_boxes > max_det:
            crowded.append((s.key(),
                            f"{s.key()}: {s.label.n_boxes} objects, "
                            f"max_det={max_det}"))
        for b in s.label.boxes:
            n_boxes += 1
            if b.w <= 0 or b.h <= 0:
                continue          # geometry_scan owns degenerate boxes
            if b.w < min_side or b.h < min_side:
                too_small.append((s.key(),
                                  f"{s.key()} line {b.line_no}: "
                                  f"{b.w * imgsz:.1f}x{b.h * imgsz:.1f}px at "
                                  f"imgsz={imgsz} (needs >= "
                                  f"{FINEST_STRIDE * MIN_CELLS:.0f}px)"))
                per_class[b.cls] += 1

    out: list[Finding] = []
    if too_small:
        share = len(too_small) / max(n_boxes, 1)
        worst = ", ".join(f"'{ds.class_name(c)}' {n}"
                          for c, n in per_class.most_common(5))
        doubled = sum(1 for _k, m in too_small) and imgsz * 2
        out.append(Finding(
            type="undetectable_at_imgsz", severity=MAJOR,
            title=f"{len(too_small)} box(es) ({share:.1%}) are too small to "
                  f"detect at imgsz={imgsz}",
            detail=f"At an input size of {imgsz} these objects span fewer than "
                   f"{FINEST_STRIDE * MIN_CELLS:.0f} pixels, which is below the "
                   f"finest feature stride the detection head predicts on "
                   f"(P3, stride {FINEST_STRIDE}). No cell in the network can "
                   f"represent them, so they are not hard examples - they are "
                   f"unlearnable at this resolution, and every one is counted "
                   f"as a false negative against your recall. This is "
                   f"{share:.1%} of all annotations in the dataset. Training "
                   f"at {doubled} would halve the threshold; the alternative "
                   f"is to accept that these instances are not being learned "
                   f"and stop treating the recall number as a model problem. "
                   f"Most affected classes: {worst}.",
            detector="training_fit_scan",
            items=sorted({k for k, _ in too_small}),
            evidence=[m for _, m in too_small[:12]],
            fix={"action": "raise_imgsz_or_accept_loss", "targets": []}))

    if crowded:
        out.append(Finding(
            type="over_max_detections", severity=MAJOR,
            title=f"{len(crowded)} image(s) hold more objects than max_det="
                  f"{max_det}",
            detail=f"Validation predicts at most {max_det} boxes per image by "
                   f"default. On these images some ground-truth objects cannot "
                   f"be matched no matter how good the model is, so the "
                   f"reported recall has a ceiling below 1.0 that has nothing "
                   f"to do with the weights. Raise max_det for evaluation, or "
                   f"know that the number is capped.",
            detector="training_fit_scan",
            items=sorted({k for k, _ in crowded}),
            evidence=[m for _, m in crowded[:12]],
            fix={"action": "raise_max_det", "targets": []}))
    return out
