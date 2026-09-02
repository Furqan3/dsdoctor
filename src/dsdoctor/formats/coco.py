"""COCO detection JSON -> YOLO view.

Two layouts cover almost everything seen in the wild:

  Roboflow-style   root/<split>/_annotations.coco.json  with images beside it
  COCO-style       root/annotations/instances_<split>.json, images in
                   root/<split>/ or root/images/<split>/

Category ids in COCO are arbitrary and sparse (the canonical set runs 1..90
with gaps), while YOLO requires contiguous 0..nc-1. The remap is by sorted
category id, which is deterministic and therefore reproducible across runs.

An annotation whose ``category_id`` is not declared in ``categories`` is not
dropped. It is assigned an id past the end of the class list, so that
``class_scan`` reports it as ``class_id_out_of_range`` - the same defect, in
the same vocabulary, as if the dataset had arrived in YOLO form.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import link_or_copy, snap_subpixel, yolo_row

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_annotation_files(root: Path) -> dict[str, Path]:
    """split name -> annotation json."""
    found: dict[str, Path] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for name in ("_annotations.coco.json", "annotations.json"):
            if (sub / name).is_file():
                found[sub.name] = sub / name
    ann_dir = root / "annotations"
    if ann_dir.is_dir():
        for p in sorted(ann_dir.glob("instances_*.json")):
            found.setdefault(p.stem[len("instances_"):], p)
        for p in sorted(ann_dir.glob("*.json")):
            if p.stem.startswith("instances_"):
                continue
            found.setdefault(p.stem, p)
    for p in sorted(root.glob("*.json")):
        if p.name.endswith(".coco.json") or p.stem.startswith("instances_"):
            found.setdefault(p.stem.replace("instances_", "") or "train", p)
    return found


def _image_dir(root: Path, split: str, ann_path: Path) -> Path:
    for cand in (ann_path.parent, root / split, root / "images" / split,
                 root / "images"):
        if cand.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES
                                 for p in cand.iterdir() if p.is_file()):
            return cand
    return ann_path.parent


def convert(root: Path, out: Path) -> tuple[Path, dict]:
    ann_files = find_annotation_files(root)
    if not ann_files:
        raise ValueError(f"no COCO annotation JSON found under {root}")

    # One class list across all splits: a per-split list would renumber the
    # same class differently on each side, which is itself a defect.
    categories: dict[int, str] = {}
    parsed: dict[str, dict] = {}
    for split, path in ann_files.items():
        with path.open() as fh:
            data = json.load(fh)
        parsed[split] = data
        for c in data.get("categories") or []:
            categories.setdefault(int(c["id"]), str(c.get("name", c["id"])))

    order = sorted(categories)
    remap = {cid: i for i, cid in enumerate(order)}
    names = [categories[c] for c in order]

    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"names": names, "nc": len(names),
         "train": "images/train", "val": "images/val"}, sort_keys=False))

    report = {"format": "coco", "converted": True, "source": str(root),
              "output": str(out), "classes": len(names), "splits": {},
              "unknown_category_ids": [], "images_without_size": 0,
              "annotations": 0}
    unknown: dict[int, int] = {}

    for split, data in parsed.items():
        img_dir = _image_dir(root, split, ann_files[split])
        images = {int(im["id"]): im for im in data.get("images") or []}
        by_image: dict[int, list] = {i: [] for i in images}
        for ann in data.get("annotations") or []:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)

        lbl_out = out / "labels" / split
        img_out = out / "images" / split
        lbl_out.mkdir(parents=True, exist_ok=True)
        img_out.mkdir(parents=True, exist_ok=True)

        n_img = n_ann = 0
        for image_id, im in images.items():
            file_name = Path(str(im.get("file_name", ""))).name
            if not file_name:
                continue
            stem = Path(file_name).stem
            src = img_dir / file_name
            if src.is_file():
                link_or_copy(src, img_out / file_name)
                n_img += 1

            w = float(im.get("width") or 0)
            h = float(im.get("height") or 0)
            if not (w > 0 and h > 0):
                report["images_without_size"] += 1

            rows: list[str] = []
            for ann in by_image.get(image_id, []):
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, bw, bh = (float(v) for v in bbox)
                cid = int(ann.get("category_id", -1))
                if cid in remap:
                    cls = remap[cid]
                else:
                    # Preserve the defect rather than the annotation: an id
                    # past nc is exactly what class_scan looks for.
                    unknown[cid] = unknown.get(cid, 0) + 1
                    cls = len(names) + sorted(unknown).index(cid)

                if w > 0 and h > 0:
                    # Snap only sub-pixel overhang; real excursions survive.
                    x1, y1 = snap_subpixel(x, w), snap_subpixel(y, h)
                    x2, y2 = snap_subpixel(x + bw, w), snap_subpixel(y + bh, h)
                    xc, yc = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                    nw, nh = (x2 - x1) / w, (y2 - y1) / h
                else:
                    # No declared size: emit the raw pixel values untouched so
                    # normalisation_scan reports them, instead of guessing.
                    xc, yc, nw, nh = x + bw / 2, y + bh / 2, bw, bh
                rows.append(yolo_row(cls, xc, yc, nw, nh))
                n_ann += 1

            (lbl_out / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""))

        report["splits"][split] = {"images": n_img, "annotations": n_ann}
        report["annotations"] += n_ann

    report["unknown_category_ids"] = sorted(unknown)
    return out, report
