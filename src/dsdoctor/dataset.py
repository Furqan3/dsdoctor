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


@dataclass
class Box:
    """One annotation row, kept alongside the text it was parsed from."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float
    line_no: int
    raw: str

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.xc - self.w / 2, self.yc - self.h / 2,
                self.xc + self.w / 2, self.yc + self.h / 2)

    @property
    def area(self) -> float:
        return self.w * self.h


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

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        self.names: list[str] = []
        self.yaml_path: Path | None = None
        self.yaml_error: str | None = None
        self.samples: list[Sample] = []
        self._by_key: dict[str, Sample] = {}
        self._load_yaml()
        self._load_samples()

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
                s.label = _parse_label(labels[stem]) if stem in labels else None
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
            "nc": self.nc,
            "names": self.names,
            "splits": per_split,
            "total_boxes": total_boxes,
            "class_counts": {self.class_name(c): n
                             for c, n in sorted(class_counts.items())},
            "yaml_error": self.yaml_error,
        }


def _parse_label(path: Path) -> LabelFile:
    lf = LabelFile(path=path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        lf.parse_errors.append(ParseError(0, "", f"unreadable: {exc}"))
        return lf
    for i, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            lf.parse_errors.append(
                ParseError(i, raw, f"expected 5 fields, found {len(parts)}"))
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            lf.parse_errors.append(ParseError(i, raw, "non-numeric field"))
            continue
        lf.boxes.append(Box(cls=cls, xc=xc, yc=yc, w=w, h=h, line_no=i, raw=raw))
    return lf


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
