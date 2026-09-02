"""Capture metadata that silently contradicts the labels.

This is the class of defect the README calls "no symptom at all". An image
with an EXIF orientation tag is stored one way and displayed another. The
annotator drew boxes on the displayed image; the dataloader may read the
stored one. Nothing errors, nothing looks wrong in a viewer - the viewer is
the thing applying the rotation - and the model learns a systematically
transformed version of every box on those files.

Whether it bites depends on the loader: Ultralytics applies
``ImageOps.exif_transpose`` on some paths and not others, and a hand-rolled
`PIL.Image.open` applies none. That is precisely why it is worth reporting -
the dataset is only correct under an assumption nobody wrote down.
"""

from __future__ import annotations

from PIL import Image, UnidentifiedImageError

from ..dataset import Dataset
from ..findings import Finding, CRITICAL
from . import register

EXIF_ORIENTATION_TAG = 274

# 1 is "as stored". 2-4 flip or rotate without changing the aspect ratio;
# 5-8 additionally transpose width and height, which is the dangerous set -
# normalised coordinates written against one aspect ratio are meaningless
# against the other.
ORIENTATION_MEANING = {
    2: "mirrored horizontally",
    3: "rotated 180°",
    4: "mirrored vertically",
    5: "transposed (width/height swap)",
    6: "rotated 90° clockwise (width/height swap)",
    7: "transverse (width/height swap)",
    8: "rotated 90° counter-clockwise (width/height swap)",
}
DIMENSION_SWAPPING = {5, 6, 7, 8}


def _orientation(path) -> int | None:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            value = exif.get(EXIF_ORIENTATION_TAG)
    except (UnidentifiedImageError, OSError, ValueError, AttributeError):
        return None  # image_integrity_scan owns reporting unreadable files
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value in ORIENTATION_MEANING else None


@register("exif_orientation_scan",
          "Read the EXIF orientation tag of every image to find files whose "
          "stored pixels differ from what an annotation tool displayed. "
          "Detects exif_orientation. Reads image headers, so medium cost.",
          reads_pixels=True, covers=("exif_orientation",), group="metadata")
def exif_orientation_scan(ds: Dataset) -> list[Finding]:
    tagged: list[tuple[str, int]] = []
    for s in ds.samples:
        if s.image_path is None:
            continue
        value = _orientation(s.image_path)
        if value is not None:
            tagged.append((s.key(), value))

    if not tagged:
        return []

    swapping = [(k, v) for k, v in tagged if v in DIMENSION_SWAPPING]
    keys = sorted({k for k, _ in tagged})
    evidence = [f"{k}: EXIF orientation {v} ({ORIENTATION_MEANING[v]})"
                for k, v in tagged[:12]]

    detail = (
        f"{len(tagged)} image(s) carry a non-trivial EXIF orientation tag. The "
        f"bytes on disk are not what an annotation tool showed the person who "
        f"drew these boxes, so the labels are correct only if the training "
        f"loader applies the same rotation the annotator saw. Ultralytics does "
        f"this on some code paths and not others, and a plain `Image.open` does "
        f"not do it at all.")
    if swapping:
        detail += (
            f" {len(swapping)} of them use an orientation that swaps width and "
            f"height, which is the case that cannot go unnoticed: normalised "
            f"coordinates written against one aspect ratio describe a different "
            f"box entirely against the other.")
    detail += (" The durable fix is to bake the rotation into the pixels and "
               "strip the tag, so the file no longer depends on who reads it.")

    return [Finding(
        type="exif_orientation", severity=CRITICAL,
        title=f"{len(tagged)} image(s) have an EXIF orientation tag",
        detail=detail,
        detector="exif_orientation_scan", items=keys,
        evidence=evidence,
        fix={"action": "bake_exif_orientation", "targets": keys})]
