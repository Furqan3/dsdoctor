"""Polygon and keypoint geometry, for segmentation and pose datasets.

`geometry_scan` already applies to these datasets in full, because the parser
derives a bounding box from every polygon - so out-of-bounds, degenerate,
tiny and duplicated are all covered without a second implementation. What is
left is the set of defects that only exist once a shape has more than four
numbers in it, and each one is a way for a mask to be rasterised differently
by the training code than by the tool that drew it:

  * fewer than three points, which encloses nothing at all;
  * three or more points that are collinear, which also encloses nothing but
    looks like a real polygon in every text editor;
  * a polygon that crosses itself, which has no well-defined interior - the
    resulting mask depends on whether the rasteriser uses an even-odd or a
    non-zero winding fill rule, and the two disagree exactly where the
    crossing is.

On a detection dataset every check here returns immediately: there are no
polygons and no keypoints to look at.
"""

from __future__ import annotations

from ..dataset import (Dataset, SEGMENT, POSE, polygon_area,
                       self_intersections)
from ..findings import Finding, CRITICAL, MAJOR, MINOR
from . import register

# Self-intersection is quadratic in the vertex count. Real polygons from an
# annotation tool have tens of points; a machine-generated contour can have
# thousands, and checking one of those costs more than the rest of the audit.
# Above this the polygon is counted and reported as unchecked rather than
# silently skipped - an unchecked shape must not read as a clean one.
MAX_VERTICES_FOR_CROSSING_CHECK = 160

def _locations(hits) -> dict[str, list[int]]:
    """(key, line_no, message) tuples -> the row map the visual report uses."""
    out: dict[str, list[int]] = {}
    for key, line_no, _msg in hits:
        out.setdefault(key, []).append(line_no)
    return {k: sorted(set(v)) for k, v in out.items()}


AREA_EPS = 1e-9
KEYPOINT_MARGIN = 0.02          # 2% of the image, to allow honest annotation slop


@register("polygon_scan",
          "Validate segmentation polygons: point counts, zero-area shapes and "
          "self-intersection. Detects polygon_too_few_points, "
          "polygon_zero_area and polygon_self_intersecting. Inert on "
          "detection datasets.",
          covers=("polygon_too_few_points", "polygon_zero_area",
                  "polygon_self_intersecting", "polygon_unverified"))
def polygon_scan(ds: Dataset) -> list[Finding]:
    if ds.task != SEGMENT:
        return []

    few: list[tuple[str, str]] = []
    zero: list[tuple[str, str]] = []
    crossing: list[tuple[str, str]] = []
    unchecked: list[tuple[str, str]] = []

    for s in ds.samples:
        if not s.label:
            continue
        for b in s.label.boxes:
            if b.polygon is None:
                continue
            where = f"{s.key()} line {b.line_no}"
            n = len(b.polygon)

            if n < 3:
                few.append((s.key(), b.line_no, f"{where}: {n} point(s) -> {b.raw[:80]!r}"))
                continue

            # Crossing is tested before area, and the order is load-bearing.
            # A symmetric self-intersecting polygon - the classic bow tie -
            # has a signed area of exactly zero, because its two lobes wind in
            # opposite directions and cancel. Testing area first reports it as
            # "zero area", which is true of the arithmetic and misleading
            # about the defect: the shape does enclose region, the crossing is
            # why that region is undefined. Most specific diagnosis wins.
            if n <= MAX_VERTICES_FOR_CROSSING_CHECK:
                hits = self_intersections(b.polygon)
                if hits:
                    crossing.append((s.key(), b.line_no,
                                     f"{where}: {len(hits)} self-crossing edge "
                                     f"pair(s), first at edges {hits[0]}"))
                    continue
            else:
                unchecked.append((s.key(), b.line_no,
                                  f"{where}: {n} vertices, above the "
                                  f"{MAX_VERTICES_FOR_CROSSING_CHECK}-vertex "
                                  f"limit for the crossing check"))

            area = abs(polygon_area(b.polygon))
            if area <= AREA_EPS:
                zero.append((s.key(), b.line_no,
                             f"{where}: {n} points enclosing area {area:.2e} "
                             f"(collinear or coincident)"))

    out: list[Finding] = []
    if few:
        out.append(Finding(
            type="polygon_too_few_points", severity=CRITICAL,
            title=f"{len(few)} polygon(s) have fewer than three points",
            detail="A polygon needs three points to enclose any area at all. "
                   "Two points describe a line and one describes a dot, and "
                   "both rasterise to an empty mask - so the object is present "
                   "in the label file, absent from the training target, and "
                   "taught to the model as background.",
            detector="polygon_scan", items=sorted({k for k, _, _ in few}),
            evidence=[m for _, _, m in few[:12]],
            locations=_locations(few),
            fix={"action": "drop_or_redraw_polygons",
                 "targets": sorted({k for k, _, _ in few})}))
    if zero:
        out.append(Finding(
            type="polygon_zero_area", severity=CRITICAL,
            title=f"{len(zero)} polygon(s) enclose zero area",
            detail="These have enough points to look like real shapes but all "
                   "of them fall on one line, or repeat the same coordinate. "
                   "The mask is empty. This is the segmentation equivalent of "
                   "a degenerate box, and it is harder to spot because the row "
                   "is long and looks plausible.",
            detector="polygon_scan", items=sorted({k for k, _, _ in zero}),
            evidence=[m for _, _, m in zero[:12]],
            locations=_locations(zero),
            fix={"action": "drop_or_redraw_polygons",
                 "targets": sorted({k for k, _, _ in zero})}))
    if crossing:
        detail = ("A polygon that crosses itself has no well-defined interior. "
                  "Whether a pixel near the crossing ends up inside the mask "
                  "depends on the fill rule the rasteriser uses - even-odd and "
                  "non-zero winding give different answers - so the training "
                  "target silently differs from what the annotation tool "
                  "displayed, and differs again between libraries.")
        out.append(Finding(
            type="polygon_self_intersecting", severity=CRITICAL,
            title=f"{len(crossing)} polygon(s) cross themselves",
            detail=detail,
            detector="polygon_scan", items=sorted({k for k, _, _ in crossing}),
            evidence=[m for _, _, m in crossing[:12]],
            locations=_locations(crossing),
            fix={"action": "drop_or_redraw_polygons",
                 "targets": sorted({k for k, _, _ in crossing})}))

    if unchecked:
        # This used to be a sentence appended to the self-intersection
        # finding, which meant that when there were no crossings there was no
        # finding to append it to and the limitation vanished entirely - an
        # unchecked polygon read as a clean one, which is the single thing the
        # comment above the constant says must not happen.
        out.append(Finding(
            type="polygon_unverified", severity=MINOR,
            title=f"{len(unchecked)} polygon(s) were too complex to check for "
                  f"self-intersection",
            detail=f"Checking whether a polygon crosses itself costs time "
                   f"quadratic in its vertex count, so shapes above "
                   f"{MAX_VERTICES_FOR_CROSSING_CHECK} vertices are skipped "
                   f"rather than allowed to dominate the audit. These are not "
                   f"known to be clean; they are unexamined. Machine-generated "
                   f"contours are the usual source, and they are also the "
                   f"least likely to self-intersect - but this says so rather "
                   f"than letting silence imply a pass.",
            detector="polygon_scan", items=sorted({k for k, _, _ in unchecked}),
            evidence=[m for _, _, m in unchecked[:12]],
            locations=_locations(unchecked),
            fix={"action": "simplify_or_verify_polygons",
                 "targets": sorted({k for k, _, _ in unchecked})}))
    return out


@register("keypoint_scan",
          "Validate pose keypoints: visibility flags and points that fall "
          "outside their own box. Detects keypoint_visibility_invalid and "
          "keypoint_outside_box. Inert on detection datasets.",
          covers=("keypoint_visibility_invalid", "keypoint_outside_box"))
def keypoint_scan(ds: Dataset) -> list[Finding]:
    if ds.task != POSE:
        return []

    bad_vis: list[tuple[str, str]] = []
    outside: list[tuple[str, str]] = []

    for s in ds.samples:
        if not s.label:
            continue
        for b in s.label.boxes:
            if not b.keypoints:
                continue
            x1, y1, x2, y2 = b.xyxy
            for j, (kx, ky, kv) in enumerate(b.keypoints):
                where = f"{s.key()} line {b.line_no} keypoint {j}"
                if kv not in (0.0, 1.0, 2.0):
                    bad_vis.append((s.key(), b.line_no, f"{where}: visibility={kv:g}"))
                    continue
                if kv == 0.0:
                    continue      # not labelled; its coordinates carry no claim
                if not (x1 - KEYPOINT_MARGIN <= kx <= x2 + KEYPOINT_MARGIN
                        and y1 - KEYPOINT_MARGIN <= ky <= y2 + KEYPOINT_MARGIN):
                    outside.append((s.key(), b.line_no,
                                    f"{where}: at ({kx:.3f},{ky:.3f}), box is "
                                    f"({x1:.3f},{y1:.3f})-({x2:.3f},{y2:.3f})"))

    out: list[Finding] = []
    if bad_vis:
        out.append(Finding(
            type="keypoint_visibility_invalid", severity=MAJOR,
            title=f"{len(bad_vis)} keypoint(s) have an invalid visibility flag",
            detail="The third value of a keypoint must be 0 (unlabelled), "
                   "1 (labelled but occluded) or 2 (visible). Anything else is "
                   "usually a confidence score from a prediction dump that was "
                   "mistaken for ground truth, and the loss will weight those "
                   "points by a number that means something else entirely.",
            detector="keypoint_scan", items=sorted({k for k, _, _ in bad_vis}),
            evidence=[m for _, _, m in bad_vis[:12]],
            locations=_locations(bad_vis),
            fix={"action": "repair_keypoint_visibility",
                 "targets": sorted({k for k, _, _ in bad_vis})}))
    if outside:
        out.append(Finding(
            type="keypoint_outside_box", severity=MINOR,
            title=f"{len(outside)} visible keypoint(s) fall outside their box",
            detail="A keypoint marked visible should lie within the instance it "
                   "belongs to. A few of these are annotation slop near an "
                   "edge; a systematic pattern usually means keypoints and "
                   "boxes were normalised against different image dimensions, "
                   "or that instances were matched to the wrong boxes during "
                   "an export.",
            detector="keypoint_scan", items=sorted({k for k, _, _ in outside}),
            evidence=[m for _, _, m in outside[:12]],
            locations=_locations(outside),
            fix={"action": "review_keypoint_assignment",
                 "targets": sorted({k for k, _, _ in outside})}))
    return out
