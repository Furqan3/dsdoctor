"""In-memory model of a YOLO-format object-detection dataset.

Layout we expect (the standard Ultralytics one)::

    root/
      data.yaml              names: [...]        (nc optional, derived from names)
      images/<split>/*.jpg
      labels/<split>/*.txt

Every label row is ``class_id xc yc w h`` with the four geometry values
normalised to ``[0, 1]`` and relative to the image.

Loading is deliberately forgiving: a dataset that is *broken* is exactly the
input this tool exists to describe, so nothing here raises on malformed rows.
Parse failures are captured as data (``LabelFile.parse_errors``) and handed to
the detectors, which decide how much they matter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageFile, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Canonical split-name aliases. Datasets in the wild disagree about what the
# folders are called and a check that only understands "train"/"val" quietly
# does nothing on half of them.
TRAIN_NAMES = {"train", "train2017", "training"}
VAL_NAMES = {"val", "valid", "val2017", "validation", "test"}


def split_role(split: str) -> str:
    """"train", "val" or "other" for a split directory name."""
    s = split.lower()
    if s in TRAIN_NAMES:
        return "train"
    if s in VAL_NAMES:
        return "val"
    return "other"


# The three label geometries this parser understands. A dataset is one of
# them, and guessing per-row would be a mistake: a detection row with a stray
# sixth column and a three-point polygon are not distinguishable in isolation,
# and one of them is a defect this tool exists to report.
DETECT, SEGMENT, POSE = "detect", "segment", "pose"
TASKS = (DETECT, SEGMENT, POSE)


@dataclass
class Box:
    """One annotation row, kept alongside the text it was parsed from.

    Segmentation and pose rows are also represented here, with `xc/yc/w/h`
    derived from the polygon's bounds. That is deliberate: it means every
    geometry, class and duplicate check written for detection applies
    unchanged to a segmentation dataset, and the polygon-specific checks are
    additive rather than a parallel implementation.
    """

    cls: int
    xc: float
    yc: float
    w: float
    h: float
    line_no: int
    raw: str
    polygon: tuple[tuple[float, float], ...] | None = None
    keypoints: tuple[tuple[float, float, float], ...] | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.xc - self.w / 2, self.yc - self.h / 2,
                self.xc + self.w / 2, self.yc + self.h / 2)

    @property
    def area(self) -> float:
        return self.w * self.h


def polygon_area(points) -> float:
    """Signed shoelace area. The sign carries the winding direction."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _segments_cross(a, b, c, d) -> bool:
    """Do segments ab and cd properly cross (not merely touch at an endpoint)?"""
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 != o2 and o3 != o4 and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0


def self_intersections(points) -> list[tuple[int, int]]:
    """Pairs of non-adjacent edges that cross.

    A self-intersecting polygon has no well-defined interior, so the mask
    rasterised from it depends on the fill rule the training code happens to
    use - which is exactly the kind of defect that trains a model on a target
    nobody drew.
    """
    n = len(points)
    if n < 4:
        return []
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    out = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue          # adjacent through the wrap-around
            if _segments_cross(*edges[i], *edges[j]):
                out.append((i, j))
    return out


@dataclass
class ParseError:
    line_no: int
    raw: str
    reason: str


@dataclass
class LabelFile:
    path: Path
    boxes: list[Box] = field(default_factory=list)
    parse_errors: list[ParseError] = field(default_factory=list)
    exists: bool = True

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)


@dataclass
class Sample:
    """An image and the label file that should accompany it."""

    stem: str
    split: str
    image_path: Path | None
    label_path: Path | None
    label: LabelFile | None = None

    # Filled lazily by detectors that need pixels.
    width: int | None = None
    height: int | None = None
    image_error: str | None = None

    @property
    def rel_image(self) -> str:
        return str(self.image_path) if self.image_path else f"<missing image {self.stem}>"

    def key(self) -> str:
        return f"{self.split}/{self.stem}"


class Dataset:
    """A loaded YOLO dataset. Cheap to construct; pixels are read on demand."""

    def __init__(self, root: str | Path, task: str | None = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        self.names: list[str] = []
        self.yaml_path: Path | None = None
        self.yaml_error: str | None = None
        self.samples: list[Sample] = []
        self._by_key: dict[str, Sample] = {}
        self.task: str = task or DETECT
        self.task_source: str = "argument" if task else "default"
        self.n_kpt: int = 0
        self.kpt_dim: int = 3
        self._load_yaml()
        if task is None:
            self._resolve_task()
        self._load_samples()

    def _resolve_task(self) -> None:
        """data.yaml wins; otherwise infer from the label rows."""
        if self.n_kpt:
            self.task, self.task_source = POSE, "data.yaml kpt_shape"
            return
        declared = getattr(self, "_declared_task", None)
        if declared in TASKS:
            self.task, self.task_source = declared, "data.yaml task"
            return
        labels_root = self.root / "labels"
        paths: list[Path] = []
        if labels_root.is_dir():
            for split_dir in sorted(p for p in labels_root.iterdir() if p.is_dir()):
                paths += sorted(split_dir.glob("*.txt"))[:20]
        inferred = infer_task(paths)
        self.task = inferred
        self.task_source = "inferred from label rows"

    # ---------------------------------------------------------------- config

    def _load_yaml(self) -> None:
        for cand in ("data.yaml", "dataset.yaml", "data.yml"):
            p = self.root / cand
            if p.is_file():
                self.yaml_path = p
                break
        if self.yaml_path is None:
            self.yaml_error = "no data.yaml found at dataset root"
            return
        try:
            cfg = yaml.safe_load(self.yaml_path.read_text()) or {}
        except yaml.YAMLError as exc:
            self.yaml_error = f"data.yaml is not valid YAML: {exc}"
            return
        names = cfg.get("names")
        if isinstance(names, dict):
            # {0: 'person', 1: 'bicycle'} -> ordered list
            self.names = [names[k] for k in sorted(names, key=int)]
        elif isinstance(names, list):
            self.names = list(names)
        else:
            self.yaml_error = "data.yaml has no usable 'names' entry"
        self._declared_task = str(cfg.get("task") or "").strip().lower() or None
        kpt = cfg.get("kpt_shape")
        if isinstance(kpt, (list, tuple)) and len(kpt) >= 1:
            try:
                self.n_kpt = int(kpt[0])
                self.kpt_dim = int(kpt[1]) if len(kpt) > 1 else 3
            except (TypeError, ValueError):
                self.n_kpt, self.kpt_dim = 0, 3
        declared = cfg.get("nc")
        if declared is not None and self.names and int(declared) != len(self.names):
            self.yaml_error = (
                f"data.yaml declares nc={declared} but lists {len(self.names)} names"
            )

    @property
    def nc(self) -> int:
        return len(self.names)

    # --------------------------------------------------------------- samples

    def _load_samples(self) -> None:
        images_root = self.root / "images"
        labels_root = self.root / "labels"
        splits: set[str] = set()
        for base in (images_root, labels_root):
            if base.is_dir():
                splits.update(p.name for p in base.iterdir() if p.is_dir())

        for split in sorted(splits):
            img_dir = images_root / split
            lbl_dir = labels_root / split
            images: dict[str, Path] = {}
            if img_dir.is_dir():
                for p in sorted(img_dir.iterdir()):
                    if p.suffix.lower() in IMAGE_SUFFIXES:
                        images[p.stem] = p
            labels: dict[str, Path] = {}
            if lbl_dir.is_dir():
                for p in sorted(lbl_dir.iterdir()):
                    if p.suffix.lower() == ".txt":
                        labels[p.stem] = p

            for stem in sorted(set(images) | set(labels)):
                s = Sample(
                    stem=stem,
                    split=split,
                    image_path=images.get(stem),
                    label_path=labels.get(stem),
                )
                s.label = (_parse_label(labels[stem], self.task,
                                        self.n_kpt, self.kpt_dim)
                           if stem in labels else None)
                self.samples.append(s)
                self._by_key[s.key()] = s

    # ----------------------------------------------------------------- views

    @property
    def splits(self) -> list[str]:
        return sorted({s.split for s in self.samples})

    def in_split(self, split: str) -> list[Sample]:
        return [s for s in self.samples if s.split == split]

    def get(self, key: str) -> Sample | None:
        return self._by_key.get(key)

    def class_name(self, cls: int) -> str:
        if 0 <= cls < len(self.names):
            return self.names[cls]
        return f"<id {cls} not in data.yaml>"

    def ensure_image_meta(self, sample: Sample) -> None:
        """Read width/height from the header. Cheap: does not decode pixels."""
        if sample.width is not None or sample.image_error is not None:
            return
        if sample.image_path is None:
            sample.image_error = "no image file"
            return
        try:
            with Image.open(sample.image_path) as im:
                sample.width, sample.height = im.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            sample.image_error = f"{type(exc).__name__}: {exc}"

    def verify_decodable(self, sample: Sample) -> None:
        """Force a full decode to catch truncation.

        ``Image.verify()`` only checks the header and container structure, so a
        JPEG whose scan data is cut off part way through sails past it and then
        blows up inside the training loop instead. The only reliable test is to
        decode the pixels, with PIL's truncated-image tolerance switched off.
        """
        if sample.image_error is not None or sample.image_path is None:
            self.ensure_image_meta(sample)
            if sample.image_error:
                return
        prev = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            with Image.open(sample.image_path) as im:
                im.load()
                sample.width, sample.height = im.size
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            sample.image_error = f"{type(exc).__name__}: {exc}"
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev

    def summary(self) -> dict:
        """Small, cheap dict used both in prompts and in the report header."""
        per_split: dict[str, dict] = {}
        class_counts: dict[int, int] = {}
        total_boxes = 0
        for split in self.splits:
            samples = self.in_split(split)
            boxes = sum(s.label.n_boxes for s in samples if s.label)
            total_boxes += boxes
            per_split[split] = {"images": sum(1 for s in samples if s.image_path),
                                "label_files": sum(1 for s in samples if s.label_path),
                                "boxes": boxes}
            for s in samples:
                if s.label:
                    for b in s.label.boxes:
                        class_counts[b.cls] = class_counts.get(b.cls, 0) + 1
        return {
            "root": str(self.root),
            "task": self.task,
            "task_source": self.task_source,
            "nc": self.nc,
            "names": self.names,
            "splits": per_split,
            "total_boxes": total_boxes,
            "class_counts": {self.class_name(c): n
                             for c, n in sorted(class_counts.items())},
            "yaml_error": self.yaml_error,
        }


def looks_like_polygon(n_fields: int) -> bool:
    """Could a row of this width be `cls x1 y1 x2 y2 x3 y3 ...`?

    Needs at least three points, and an even number of coordinates. A
    detection row with one stray trailing column has six fields, which is odd
    after the class id and therefore never mistaken for a polygon - that
    matters, because a trailing confidence column is a defect the evaluation
    injects on purpose.
    """
    return n_fields >= 7 and (n_fields - 1) % 2 == 0


def _parse_label(path: Path, task: str = DETECT, n_kpt: int = 0,
                 kpt_dim: int = 3) -> LabelFile:
    lf = LabelFile(path=path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        lf.parse_errors.append(ParseError(0, "", f"unreadable: {exc}"))
        return lf

    expected = {DETECT: 5, POSE: 5 + n_kpt * kpt_dim}.get(task)

    for i, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()

        if task == SEGMENT:
            if not looks_like_polygon(len(parts)):
                lf.parse_errors.append(ParseError(
                    i, raw, f"expected an odd count of at least 7 fields "
                            f"(class plus >=3 xy pairs), found {len(parts)}"))
                continue
        elif expected is not None and len(parts) != expected:
            lf.parse_errors.append(
                ParseError(i, raw, f"expected {expected} fields, "
                                   f"found {len(parts)}"))
            continue

        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            lf.parse_errors.append(ParseError(i, raw, "non-numeric field"))
            continue

        if task == SEGMENT:
            pts = tuple(zip(values[0::2], values[1::2]))
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            # The box is derived from the polygon's extent, so every existing
            # detection check reads a segmentation dataset correctly.
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            lf.boxes.append(Box(cls=cls, xc=(x1 + x2) / 2, yc=(y1 + y2) / 2,
                                w=x2 - x1, h=y2 - y1, line_no=i, raw=raw,
                                polygon=pts))
            continue

        xc, yc, w, h = values[:4]
        kpts = None
        if task == POSE and n_kpt:
            flat = values[4:]
            if kpt_dim == 2:
                kpts = tuple((flat[j], flat[j + 1], 2.0)
                             for j in range(0, len(flat), 2))
            else:
                kpts = tuple((flat[j], flat[j + 1], flat[j + 2])
                             for j in range(0, len(flat), 3))
        lf.boxes.append(Box(cls=cls, xc=xc, yc=yc, w=w, h=h, line_no=i,
                            raw=raw, keypoints=kpts))
    return lf


def infer_task(label_paths: list[Path], limit: int = 40) -> str:
    """Guess the label geometry from the files themselves.

    Only a strong majority counts. A detection dataset containing a handful of
    polygon-width rows is a *malformed* detection dataset, not a segmentation
    one, and silently reinterpreting it would hide the defect.
    """
    polygon = plain = 0
    for path in label_paths[:limit]:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            if looks_like_polygon(len(parts)):
                polygon += 1
            else:
                plain += 1
    total = polygon + plain
    if total >= 5 and polygon / total >= 0.9:
        return SEGMENT
    return DETECT


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
