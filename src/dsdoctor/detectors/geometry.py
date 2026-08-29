"""Box geometry: coordinates that are out of range, inverted, degenerate,
duplicated, or never normalised in the first place."""

from __future__ import annotations

from collections import defaultdict

from ..dataset import Dataset
from ..findings import Finding, CRITICAL, MAJOR
from . import register

# Below this normalised side length a box is under ~2px on a 640px letterbox,
# which Ultralytics' own dataloader will discard.
TINY_SIDE = 0.003
TINY_AREA = 1e-5
EPS = 1e-9


@register("geometry_scan",
          "Validate every box's normalised coordinates. Detects out_of_bounds, "
          "degenerate_box, tiny_box and duplicate_annotation.",
          covers=("out_of_bounds", "degenerate_box", "tiny_box",
                  "duplicate_annotation"))
def geometry_scan(ds: Dataset) -> list[Finding]:
    oob: list[tuple[str, str]] = []
    degen: list[tuple[str, str]] = []
    tiny: list[tuple[str, str]] = []
    dupes: list[tuple[str, str]] = []

    for s in ds.samples:
        if not s.label:
            continue
        seen: dict[tuple, int] = {}
        for b in s.label.boxes:
            where = f"{s.key()} line {b.line_no}"
            x1, y1, x2, y2 = b.xyxy

            if b.w <= EPS or b.h <= EPS:
                degen.append((s.key(), f"{where}: w={b.w:g} h={b.h:g} -> {b.raw!r}"))
            elif b.w < TINY_SIDE or b.h < TINY_SIDE or b.area < TINY_AREA:
                tiny.append((s.key(),
                             f"{where}: w={b.w:g} h={b.h:g} area={b.area:.2e} -> {b.raw!r}"))

            if (x1 < -EPS or y1 < -EPS or x2 > 1 + EPS or y2 > 1 + EPS
                    or not (0 - EPS <= b.xc <= 1 + EPS)
                    or not (0 - EPS <= b.yc <= 1 + EPS)):
                oob.append((s.key(),
                            f"{where}: xyxy=({x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}) -> {b.raw!r}"))

            sig = (b.cls, round(b.xc, 6), round(b.yc, 6), round(b.w, 6), round(b.h, 6))
            if sig in seen:
                dupes.append((s.key(),
                              f"{where} repeats line {seen[sig]}: {b.raw!r}"))
            else:
                seen[sig] = b.line_no

    out: list[Finding] = []
    if oob:
        out.append(_group(
            "out_of_bounds", CRITICAL, oob,
            "box(es) fall outside the normalised [0,1] range",
            "Coordinates outside [0,1] are clipped or rejected depending on the "
            "loader. Where they are clipped the box silently changes shape, so "
            "the model is trained against a target the annotator never drew.",
            "clip_or_drop_boxes"))
    if degen:
        out.append(_group(
            "degenerate_box", CRITICAL, degen,
            "box(es) have zero or negative width/height",
            "A zero-area box produces a NaN in most IoU implementations, which "
            "propagates into the loss and ends the run.",
            "drop_degenerate_boxes"))
    if tiny:
        out.append(_group(
            "tiny_box", MAJOR, tiny,
            "box(es) are too small to survive the dataloader",
            f"Boxes with a normalised side below {TINY_SIDE} are dropped during "
            "letterboxing. They inflate the apparent annotation count while "
            "contributing nothing, which makes per-class coverage look better "
            "than it is.",
            "drop_tiny_boxes"))
    if dupes:
        out.append(_group(
            "duplicate_annotation", MAJOR, dupes,
            "box(es) are listed more than once in the same file",
            "Duplicated boxes double-count in the loss and break the "
            "one-target-per-object assumption used by NMS-free matchers.",
            "dedupe_annotations"))
    return out


@register("normalisation_scan",
          "Check whether coordinates were ever normalised, by comparing label "
          "value ranges against the image dimensions. Detects "
          "denormalised_coords. Reads image headers.",
          reads_pixels=True, covers=("denormalised_coords",))
def normalisation_scan(ds: Dataset) -> list[Finding]:
    """Pixel coordinates written into a YOLO file are the classic export bug.

    A single >1 value is ambiguous (it could be one sloppy box), so we only
    claim this when a file's values are *systematically* larger than 1 and
    plausibly bounded by the real image size.
    """
    hits: list[tuple[str, str]] = []
    for s in ds.samples:
        if not s.label or not s.label.boxes:
            continue
        over = [b for b in s.label.boxes if max(b.xc, b.yc, b.w, b.h) > 1.5]
        if len(over) < max(1, len(s.label.boxes) // 2):
            continue
        ds.ensure_image_meta(s)
        if not s.width or not s.height:
            continue
        max_x = max(max(b.xc, b.w) for b in over)
        max_y = max(max(b.yc, b.h) for b in over)
        if max_x <= s.width * 1.02 and max_y <= s.height * 1.02:
            hits.append((s.key(),
                         f"{s.key()}: {len(over)}/{len(s.label.boxes)} rows exceed 1.0; "
                         f"max x={max_x:g} y={max_y:g} vs image {s.width}x{s.height} "
                         f"-> {over[0].raw!r}"))
    if not hits:
        return []
    return [_group(
        "denormalised_coords", CRITICAL, hits,
        "file(s) hold pixel coordinates instead of normalised fractions",
        "The values are bounded by the image dimensions rather than by 1.0, so "
        "this is an export that skipped the divide-by-width/height step. Training "
        "on it produces boxes that collapse to the top-left corner.",
        "normalise_coordinates")]


def _group(dtype: str, severity: str, hits: list[tuple[str, str]],
           title_tail: str, detail: str, action: str) -> Finding:
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, msg in hits:
        by_file[key].append(msg)
    return Finding(
        type=dtype, severity=severity,
        title=f"{len(hits)} {title_tail}",
        detail=detail, detector="geometry_scan" if dtype != "denormalised_coords"
        else "normalisation_scan",
        items=sorted(by_file),
        evidence=[m for _, m in hits[:12]],
        fix={"action": action, "targets": sorted(by_file)})
