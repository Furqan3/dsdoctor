"""Read datasets that are not in YOLO layout.

The tool's own framing is an engineer holding a dataset they did not create -
a vendor delivery, a client hand-off, a merge of three internal collections.
Those arrive as COCO JSON or Pascal VOC XML at least as often as they arrive
as YOLO text files, and a checker that only reads one layout does nothing at
all for the others.

Rather than teach every detector three schemas, an incoming dataset is
converted to a YOLO *view*: label files and a data.yaml written into a cache
directory, with images symlinked rather than copied. Every detector, the
agent, the scorer and the fix plan then work unchanged.

**The conversion is deliberately faithful, not corrective.** This is the part
that is easy to get wrong and would quietly destroy the tool's purpose. A
converter written for convenience clamps coordinates to the image, drops
degenerate boxes and skips annotations whose category is unknown - and every
one of those is a defect this tool exists to report. So the converter
normalises coordinates and nothing else: an out-of-bounds box stays out of
bounds, a zero-area box stays zero-area, and an annotation referring to a
category the file never declares is mapped to an id past the end of the class
list, which is precisely how `class_scan` detects the same defect natively.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from ..dataset import Dataset

YOLO, COCO, VOC = "yolo", "coco", "voc"


def detect(root: str | Path) -> str | None:
    """Identify the layout at `root`, or None if nothing is recognised."""
    root = Path(root)
    if not root.is_dir():
        return None
    for cand in ("data.yaml", "dataset.yaml", "data.yml"):
        if (root / cand).is_file():
            return YOLO
    from . import coco, voc
    if coco.find_annotation_files(root):
        return COCO
    if voc.find_annotation_dirs(root):
        return VOC
    return None


def cache_dir_for(root: Path) -> Path:
    """A stable per-dataset location for the converted view.

    Keyed by the absolute source path so repeated runs reuse one directory,
    and placed outside the dataset so that reading a dataset never writes to
    it - the same read-only guarantee `scan` and `audit` make everywhere else.
    """
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    tag = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
    return Path(base) / "dsdoctor" / "converted" / f"{root.name}-{tag}"


def snap_subpixel(v: float, extent: float) -> float:
    """Pull a coordinate back onto the image when it is off by under half a pixel.

    Found by round-tripping a dataset through this converter and diffing the
    findings against the YOLO original: 99 boxes acquired a CRITICAL
    ``out_of_bounds`` that the source did not have, each exceeding the frame by
    5e-9 of a normalised unit.

    They are boxes that legitimately touch the edge. ``xmax == width`` is the
    ordinary way for a pixel-corner format to say "this box reaches the right
    edge", and dividing by the width lands a few float ULPs above 1.0, which is
    over the detector's 1e-9 tolerance. A converter that reports those is
    manufacturing defects out of arithmetic - the exact failure this tool is
    built to be trusted not to commit.

    Half a pixel is the honest threshold, and it is a statement about the source
    format rather than a tolerance chosen to make a number look good: a format
    whose coordinates *are* pixel corners cannot express "outside the image by
    0.4 pixels", so any excess below that is quantisation, not a claim. Anything
    at or beyond half a pixel is left exactly where it was found - the five real
    out-of-bounds boxes in the case that surfaced this overhang by 265 to 288
    pixels and pass through untouched.
    """
    if extent <= 0:
        return v
    if -0.5 < v < 0:
        return 0.0
    if extent < v < extent + 0.5:
        return extent
    return v


# Decimal places used when writing converted label rows. This is not a
# cosmetic choice, and the obvious value is wrong.
#
# A YOLO row stores centre and side, while every geometry check reconstructs
# corners: ``y2 = yc + h/2``. Rounding ``yc`` and ``h`` independently to *p*
# decimals leaves that reconstruction off by up to 0.75e-p. At the conventional
# p=8 that is 7.5e-9, which is over ``geometry.EPS`` (1e-9) - so a box sitting
# exactly on the image edge is written correctly, read back at 1.000000005, and
# reported as a CRITICAL out-of-bounds defect that the source never contained.
# Measured on a real round trip: 124 of them in one 600-image case.
#
# p=10 puts the worst-case reconstruction error at 7.5e-11, two orders of
# magnitude inside the tolerance, and costs two characters per number. The
# excess precision is meaningless in pixels (1e-10 of a 640px image is 6e-8 of
# a pixel) and that is precisely why it is safe to spend.
COORD_DECIMALS = 10


def yolo_row(cls: int, xc: float, yc: float, w: float, h: float) -> str:
    """One YOLO label row, at a precision that survives corner reconstruction."""
    return (f"{cls} {xc:.{COORD_DECIMALS}f} {yc:.{COORD_DECIMALS}f} "
            f"{w:.{COORD_DECIMALS}f} {h:.{COORD_DECIMALS}f}")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        shutil.copy2(src, dst)


def convert(root: str | Path, out: str | Path | None = None,
            fmt: str | None = None) -> tuple[Path, dict]:
    """Convert a dataset into a YOLO view. Returns (path, conversion report)."""
    root = Path(root).resolve()
    fmt = fmt or detect(root)
    if fmt == YOLO:
        return root, {"format": YOLO, "converted": False}
    if fmt is None:
        raise ValueError(
            f"could not identify a dataset layout at {root}. Expected a YOLO "
            f"data.yaml, COCO annotation JSON, or Pascal VOC Annotations/.")
    out = Path(out) if out else cache_dir_for(root)
    if fmt == COCO:
        from . import coco
        return coco.convert(root, out)
    if fmt == VOC:
        from . import voc
        return voc.convert(root, out)
    raise ValueError(f"unsupported format {fmt!r}")


def load_any(root: str | Path, *, out: str | Path | None = None
             ) -> tuple[Dataset, dict]:
    """Load a dataset in any supported layout, converting when necessary."""
    path, report = convert(root, out)
    return Dataset(path), report
