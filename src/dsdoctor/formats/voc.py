"""Pascal VOC XML -> YOLO view.

Layout::

    root/Annotations/*.xml
    root/JPEGImages/*.jpg          (or root/images/)
    root/ImageSets/Main/train.txt  (optional; without it everything is one split)

VOC coordinates are absolute pixel corners and, by the original specification,
**one-based** - ``xmin`` of 1 is the leftmost column. Converters that ignore
this shift every box by half a pixel and, more importantly, turn a legitimate
box touching the right edge into one that exceeds the image width. The offset
is subtracted here, and boxes that still fall outside the declared size are
left outside it, because that is a defect the geometry detector should see and
report rather than something a converter should quietly absorb.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from . import link_or_copy, snap_subpixel, yolo_row

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_annotation_dirs(root: Path) -> Path | None:
    for cand in ("Annotations", "annotations"):
        p = root / cand
        if p.is_dir() and any(p.glob("*.xml")):
            return p
    if any(root.glob("*.xml")):
        return root
    return None


def _image_dirs(root: Path) -> list[Path]:
    out = [root / c for c in ("JPEGImages", "images", "Images", "JPEGimages")]
    return [p for p in out if p.is_dir()] or [root]


def _splits(root: Path, stems: list[str]) -> dict[str, list[str]]:
    main = root / "ImageSets" / "Main"
    if not main.is_dir():
        return {"train": stems}
    out: dict[str, list[str]] = {}
    for p in sorted(main.glob("*.txt")):
        # Class-specific files look like `aeroplane_train.txt`; only the plain
        # split lists describe the split membership itself.
        if "_" in p.stem:
            continue
        listed = [ln.split()[0] for ln in p.read_text().splitlines() if ln.strip()]
        if listed:
            out[p.stem] = listed
    return out or {"train": stems}


def _text(node, tag, default=""):
    el = node.find(tag)
    return (el.text or default).strip() if el is not None and el.text else default


def convert(root: Path, out: Path) -> tuple[Path, dict]:
    ann_dir = find_annotation_dirs(root)
    if ann_dir is None:
        raise ValueError(f"no Pascal VOC Annotations directory found under {root}")

    xmls = {p.stem: p for p in sorted(ann_dir.glob("*.xml"))}
    parsed: dict[str, tuple[float, float, list[tuple[str, float, float, float, float]]]] = {}
    names: set[str] = set()
    malformed: list[str] = []

    for stem, path in xmls.items():
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            malformed.append(f"{path.name}: {exc}")
            continue
        r = tree.getroot()
        size = r.find("size")
        w = float(_text(size, "width", "0") or 0) if size is not None else 0.0
        h = float(_text(size, "height", "0") or 0) if size is not None else 0.0
        objs = []
        for obj in r.findall("object"):
            name = _text(obj, "name")
            bnd = obj.find("bndbox")
            if not name or bnd is None:
                continue
            try:
                x1 = float(_text(bnd, "xmin", "0")) - 1.0   # VOC is 1-based
                y1 = float(_text(bnd, "ymin", "0")) - 1.0
                x2 = float(_text(bnd, "xmax", "0")) - 1.0
                y2 = float(_text(bnd, "ymax", "0")) - 1.0
            except ValueError:
                malformed.append(f"{path.name}: non-numeric bndbox")
                continue
            names.add(name)
            objs.append((name, x1, y1, x2, y2))
        parsed[stem] = (w, h, objs)

    class_names = sorted(names)
    index = {n: i for i, n in enumerate(class_names)}

    image_files: dict[str, Path] = {}
    for d in _image_dirs(root):
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                image_files.setdefault(p.stem, p)

    splits = _splits(root, sorted(parsed))
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"names": class_names, "nc": len(class_names),
         "train": "images/train", "val": "images/val"}, sort_keys=False))

    report = {"format": "voc", "converted": True, "source": str(root),
              "output": str(out), "classes": len(class_names), "splits": {},
              "malformed_xml": malformed[:10], "images_without_size": 0,
              "annotations": 0}

    for split, stems in splits.items():
        lbl_out, img_out = out / "labels" / split, out / "images" / split
        lbl_out.mkdir(parents=True, exist_ok=True)
        img_out.mkdir(parents=True, exist_ok=True)
        n_img = n_ann = 0
        for stem in stems:
            src = image_files.get(stem)
            if src is not None:
                link_or_copy(src, img_out / src.name)
                n_img += 1
            if stem not in parsed:
                continue    # image with no XML: structure_scan reports it
            w, h, objs = parsed[stem]
            if not (w > 0 and h > 0):
                report["images_without_size"] += 1
            rows = []
            for name, x1, y1, x2, y2 in objs:
                if w > 0 and h > 0:
                    # Snap only sub-pixel overhang; real excursions survive.
                    x1, x2 = snap_subpixel(x1, w), snap_subpixel(x2, w)
                    y1, y2 = snap_subpixel(y1, h), snap_subpixel(y2, h)
                bw, bh = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    rows.append(yolo_row(index[name], ((x1 + x2) / 2) / w,
                                         ((y1 + y2) / 2) / h, bw / w, bh / h))
                else:
                    rows.append(yolo_row(index[name], (x1 + x2) / 2,
                                         (y1 + y2) / 2, bw, bh))
                n_ann += 1
            (lbl_out / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""))
        report["splits"][split] = {"images": n_img, "annotations": n_ann}
        report["annotations"] += n_ann

    return out, report
