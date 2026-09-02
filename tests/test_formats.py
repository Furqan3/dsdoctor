"""COCO and Pascal VOC adapters.

The property that matters is not "the converter runs". It is that the audit
reaches the same conclusion about the same data regardless of the container it
arrived in - and, more sharply, that the converter neither hides a defect nor
manufactures one. Both failures happened during development and both are
pinned here.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from conftest import write_sample, NAMES
from dsdoctor.dataset import Dataset
from dsdoctor.formats import detect, convert, load_any, snap_subpixel, yolo_row
from dsdoctor.sweep import sweep


def _export_coco(src: Dataset, root: Path) -> Path:
    for split in src.splits:
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        images, anns, aid = [], [], 1
        for i, s in enumerate(src.in_split(split), 1):
            if not s.image_path:
                continue
            src.ensure_image_meta(s)
            shutil.copy2(s.image_path, d / s.image_path.name)
            images.append({"id": i, "file_name": s.image_path.name,
                           "width": s.width, "height": s.height})
            for b in (s.label.boxes if s.label else []):
                W, H = s.width, s.height
                anns.append({"id": aid, "image_id": i, "category_id": b.cls + 1,
                             "bbox": [(b.xc - b.w / 2) * W, (b.yc - b.h / 2) * H,
                                      b.w * W, b.h * H]})
                aid += 1
        (d / "_annotations.coco.json").write_text(json.dumps({
            "images": images, "annotations": anns,
            "categories": [{"id": c + 1, "name": n}
                           for c, n in enumerate(src.names)]}))
    return root


def _export_voc(src: Dataset, root: Path) -> Path:
    (root / "Annotations").mkdir(parents=True)
    (root / "JPEGImages").mkdir(parents=True)
    (root / "ImageSets" / "Main").mkdir(parents=True)
    for split in src.splits:
        stems = []
        for s in src.in_split(split):
            if not s.image_path:
                continue
            src.ensure_image_meta(s)
            W, H = s.width, s.height
            shutil.copy2(s.image_path, root / "JPEGImages" / s.image_path.name)
            stems.append(s.stem)
            r = ET.Element("annotation")
            sz = ET.SubElement(r, "size")
            ET.SubElement(sz, "width").text = str(W)
            ET.SubElement(sz, "height").text = str(H)
            for b in (s.label.boxes if s.label else []):
                o = ET.SubElement(r, "object")
                ET.SubElement(o, "name").text = src.names[b.cls]
                bb = ET.SubElement(o, "bndbox")
                # VOC is one-based, so put the offset back on the way out.
                ET.SubElement(bb, "xmin").text = f"{(b.xc - b.w / 2) * W + 1:.6f}"
                ET.SubElement(bb, "ymin").text = f"{(b.yc - b.h / 2) * H + 1:.6f}"
                ET.SubElement(bb, "xmax").text = f"{(b.xc + b.w / 2) * W + 1:.6f}"
                ET.SubElement(bb, "ymax").text = f"{(b.yc + b.h / 2) * H + 1:.6f}"
            ET.ElementTree(r).write(root / "Annotations" / f"{s.stem}.xml")
        (root / "ImageSets" / "Main" / f"{split}.txt").write_text(
            "\n".join(stems) + "\n")
    return root


def _geometry_profile(ds: Dataset) -> dict:
    """Only the type/file pairs, which is what the scorer compares."""
    return {f.type: tuple(sorted(f.items)) for f in sweep(ds).findings}


@pytest.fixture
def defective_root(clean_root: Path) -> Path:
    """A clean dataset with three unmistakable geometry defects added."""
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text(p.read_text()
                 + "0 0.9 0.5 0.6 0.2\n"      # out of bounds
                 + "1 0.5 0.5 0.0 0.2\n"      # degenerate
                 + "2 0.5 0.5 0.002 0.002\n") # tiny (TINY_SIDE is 0.003)
    return clean_root


def test_format_detection(defective_root, tmp_path):
    assert detect(defective_root) == "yolo"
    assert detect(_export_coco(Dataset(defective_root), tmp_path / "c")) == "coco"
    assert detect(_export_voc(Dataset(defective_root), tmp_path / "v")) == "voc"
    assert detect(tmp_path / "nothing_here") is None


def test_coco_round_trip_preserves_findings_exactly(defective_root, tmp_path):
    base = _geometry_profile(Dataset(defective_root))
    assert {"out_of_bounds", "degenerate_box", "tiny_box"} <= set(base)

    ds, report = load_any(_export_coco(Dataset(defective_root), tmp_path / "c"),
                          out=tmp_path / "c_out")
    assert report["format"] == "coco"
    assert _geometry_profile(ds) == base


def test_voc_round_trip_preserves_findings_exactly(defective_root, tmp_path):
    base = _geometry_profile(Dataset(defective_root))
    ds, report = load_any(_export_voc(Dataset(defective_root), tmp_path / "v"),
                          out=tmp_path / "v_out")
    assert report["format"] == "voc"
    assert _geometry_profile(ds) == base


def test_converter_does_not_manufacture_out_of_bounds(clean_root, tmp_path):
    """The bug this guards against produced 124 false criticals on one case.

    A box touching the right edge is normal and legal. Converting it must not
    turn it into a CRITICAL out_of_bounds finding through float arithmetic.
    """
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.5 0.5 1.0 1.0\n")     # exactly the full frame
    assert not [f for f in sweep(Dataset(clean_root)).findings
                if f.type == "out_of_bounds"]

    for exporter, name in ((_export_coco, "c"), (_export_voc, "v")):
        ds, _ = load_any(exporter(Dataset(clean_root), tmp_path / name),
                         out=tmp_path / f"{name}_out")
        assert not [f for f in sweep(ds).findings if f.type == "out_of_bounds"], \
            f"{name}: converter invented an out-of-bounds box"


def test_converter_does_not_hide_a_real_excursion(clean_root, tmp_path):
    """Half a pixel is the snapping limit; a real defect is orders above it."""
    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.9 0.5 0.6 0.2\n")      # overhangs by ~0.2 of the width
    ds, _ = load_any(_export_coco(Dataset(clean_root), tmp_path / "c"),
                     out=tmp_path / "c_out")
    assert [f for f in sweep(ds).findings if f.type == "out_of_bounds"]


def test_snap_subpixel_boundaries():
    assert snap_subpixel(-0.4, 100) == 0.0        # quantisation
    assert snap_subpixel(100.4, 100) == 100.0     # quantisation
    assert snap_subpixel(-0.6, 100) == -0.6       # a real excursion survives
    assert snap_subpixel(100.6, 100) == 100.6
    assert snap_subpixel(-50.0, 100) == -50.0
    assert snap_subpixel(5.0, 0) == 5.0           # unknown extent: no opinion


def test_yolo_row_survives_corner_reconstruction():
    """yc + h/2 must not cross 1.0 through decimal rounding alone."""
    row = yolo_row(0, 0.5, 0.76981297, 0.2, 0.46037406)
    cls, xc, yc, w, h = row.split()
    assert float(yc) + float(h) / 2 <= 1.0 + 1e-9


def test_unknown_coco_category_becomes_an_out_of_range_class(clean_root, tmp_path):
    """A defect in the source must survive as the same defect, not vanish."""
    root = _export_coco(Dataset(clean_root), tmp_path / "c")
    ann = root / "train" / "_annotations.coco.json"
    data = json.loads(ann.read_text())
    data["annotations"][0]["category_id"] = 999      # never declared
    ann.write_text(json.dumps(data))

    ds, report = load_any(root, out=tmp_path / "c_out")
    assert report["unknown_category_ids"] == [999]
    assert [f for f in sweep(ds).findings if f.type == "class_id_out_of_range"]


def test_conversion_never_writes_to_the_source(defective_root, tmp_path):
    src = _export_coco(Dataset(defective_root), tmp_path / "c")
    before = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}
    load_any(src, out=tmp_path / "c_out")
    after = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}
    assert before == after
