"""Annotation smells: patterns that are legal, plausible, and usually a mistake.

Nothing here is a defect in the sense the rest of this package uses the word.
Every finding is a shape in the data that has an innocent explanation and a
much more common guilty one, so each is phrased as a question with the
evidence attached, and rated no higher than the evidence supports.

The guilty explanations are worth naming, because they are invisible
otherwise: a labelling tool whose default box was accepted rather than drawn,
a pre-annotation pass whose suggestions were never corrected, a script that
wrote a placeholder for every image it could not process.
"""

from __future__ import annotations

from collections import defaultdict

from ..dataset import Dataset
from ..findings import Finding, MAJOR, MINOR
from . import register

# How many *different* images must share a byte-identical box before repetition
# stops being a coincidence. Fixed-camera datasets legitimately repeat boxes,
# which is why this is deliberately high and why the finding says so.
TEMPLATE_MIN_IMAGES = 5
COORD_ROUNDING = 6

# A box covering essentially the whole frame.
#
# The first version of this reported every such box and fired 13 times on the
# provably clean corpus - all of them COCO's `dining table`, which really does
# fill the frame it is photographed in. Those are true observations and useless
# findings, and by this project's own standard a check with a false-positive
# rate is worse than no check.
#
# The fix is to report the *pattern* rather than the box. One whole-frame box
# is a close-up. A pipeline that converted image-level labels into boxes
# produces them at a rate no photographic subject explains, so the finding
# needs a share of the dataset behind it before it says anything. On the clean
# corpus that share is 0.19% and the check is silent; the threshold sits an
# order of magnitude above it.
WHOLE_FRAME_SIDE = 0.98
WHOLE_FRAME_MIN_SHARE = 0.02
WHOLE_FRAME_MIN_COUNT = 5


@register("provenance_scan",
          "Look for annotation patterns that are usually artefacts of the "
          "labelling process rather than descriptions of the image: one box "
          "repeated verbatim across many images, and boxes spanning the whole "
          "frame. Detects template_annotation and whole_frame_box.",
          covers=("template_annotation", "whole_frame_box"),
          group="annotations")
def provenance_scan(ds: Dataset) -> list[Finding]:
    by_signature: dict[tuple, set[str]] = defaultdict(set)
    whole_frame: list[tuple[str, str]] = []
    n_boxes = 0

    for s in ds.samples:
        if not s.label:
            continue
        for b in s.label.boxes:
            n_boxes += 1
            sig = (b.cls, round(b.xc, COORD_ROUNDING), round(b.yc, COORD_ROUNDING),
                   round(b.w, COORD_ROUNDING), round(b.h, COORD_ROUNDING))
            by_signature[sig].add(s.key())
            if b.w >= WHOLE_FRAME_SIDE and b.h >= WHOLE_FRAME_SIDE:
                whole_frame.append((s.key(),
                                    f"{s.key()} line {b.line_no}: "
                                    f"{b.w:.3f}x{b.h:.3f} of the frame, "
                                    f"class '{ds.class_name(b.cls)}'"))

    out: list[Finding] = []

    repeated = {sig: keys for sig, keys in by_signature.items()
                if len(keys) >= TEMPLATE_MIN_IMAGES}
    if repeated:
        items: set[str] = set()
        evidence = []
        for sig, keys in sorted(repeated.items(),
                                key=lambda kv: -len(kv[1]))[:10]:
            cls, xc, yc, w, h = sig
            items |= keys
            evidence.append(
                f"class '{ds.class_name(cls)}' at ({xc:g},{yc:g}) "
                f"{w:g}x{h:g} appears in {len(keys)} different images, "
                f"e.g. {', '.join(sorted(keys)[:3])}")
        # The union, not the sum. Summing per-signature counts double-counts
        # any image carrying two repeated boxes, and produced a title claiming
        # more affected images than the dataset contains.
        total = len(items)
        out.append(Finding(
            type="template_annotation", severity=MAJOR,
            title=f"{len(repeated)} box(es) repeat verbatim across "
                  f"{total} image(s)",
            detail="The same class at the same coordinates with the same size, "
                   "to six decimal places, on images that are otherwise "
                   "different. Two things produce this: a labelling tool whose "
                   "suggested default box was accepted rather than adjusted, "
                   "and a script that wrote a placeholder annotation for every "
                   "image it failed to process. Both teach the model that the "
                   "class lives at a fixed screen position. It is also what a "
                   "genuinely fixed camera looks like - if that is what this "
                   "is, the finding is noise and you can say so; if it is not, "
                   "these annotations describe the tool rather than the image.",
            detector="provenance_scan", items=sorted(items),
            evidence=evidence,
            fix={"action": "review_repeated_annotations",
                 "targets": sorted(items)}))

    share = len(whole_frame) / max(n_boxes, 1)
    if (len(whole_frame) >= WHOLE_FRAME_MIN_COUNT
            and share >= WHOLE_FRAME_MIN_SHARE):
        keys = sorted({k for k, _ in whole_frame})
        out.append(Finding(
            type="whole_frame_box", severity=MINOR,
            title=f"{len(whole_frame)} box(es) ({share:.1%}) span essentially "
                  f"the whole image",
            detail=f"A box covering the entire frame carries almost no "
                   f"localisation signal - the model is asked to learn that "
                   f"the object is everywhere. One of these is a close-up and "
                   f"means nothing. {share:.1%} of every annotation in the "
                   f"dataset is a pattern, and the usual cause is an "
                   f"image-level label that was converted into a box because "
                   f"the format demanded one. Check a few before deciding.",
            detector="provenance_scan", items=keys,
            evidence=[m for _, m in whole_frame[:12]],
            fix={"action": "review_whole_frame_boxes", "targets": keys}))
    return out
