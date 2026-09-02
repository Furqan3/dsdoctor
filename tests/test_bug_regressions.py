"""One test per bug that was actually found in this code, after it shipped.

Each of these passed review and passed the suite before it was caught. They
are grouped here rather than scattered so the list stays legible, and because
the shape they share is worth seeing at once: every one produced output that
looked right.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml
from PIL import Image

from conftest import write_sample
from dsdoctor import card
from dsdoctor.apply import strip_exif_jpeg
from dsdoctor.dataset import Dataset
from dsdoctor.detectors import run
from dsdoctor.sweep import sweep


# --------------------------------------------------------------------------
# The health card fingerprinted the converted cache, not the delivery.
#
# `dsdoctor card` on a COCO dataset wrote the card into ~/.cache, fingerprinted
# 1,201 generated files instead of the 602 received, and recorded a cache path
# as the thing it described - so `verify-card` against the real delivery could
# never match, which is the card's entire purpose.
# --------------------------------------------------------------------------

def _coco_delivery(clean_root: Path, out: Path) -> Path:
    src = Dataset(clean_root)
    for split in src.splits:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        images, anns, aid = [], [], 1
        for i, s in enumerate(src.in_split(split), 1):
            if not s.image_path:
                continue
            src.ensure_image_meta(s)
            (d / s.image_path.name).write_bytes(s.image_path.read_bytes())
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
    return out


def test_card_fingerprints_the_delivery_not_the_converted_view(clean_root,
                                                               tmp_path):
    from dsdoctor.cli import main, EXIT_OK, EXIT_MISMATCH

    delivery = _coco_delivery(clean_root, tmp_path / "delivery")
    n_delivered = sum(1 for p in delivery.rglob("*") if p.is_file())

    assert main(["card", str(delivery)]) == EXIT_OK
    health = json.loads((delivery / "health.json").read_text())

    assert (delivery / "health.json").is_file(), "card must land beside the data"
    assert health["fingerprint"]["files"] == n_delivered
    assert Path(health["dataset"]["path"]) == delivery.resolve()
    assert health["dataset"]["read_through"], "the conversion must be disclosed"

    # and it must verify against the delivery it describes
    assert main(["verify-card", str(delivery)]) == EXIT_OK
    (delivery / "train" / "_annotations.coco.json").write_text("{}")
    assert main(["verify-card", str(delivery)]) == EXIT_MISMATCH


# --------------------------------------------------------------------------
# strip_exif_jpeg dropped a trailing byte.
#
# A `while ... else` set the cursor past the end when the loop finished
# naturally, discarding whatever was unconsumed. Silent data loss inside a
# function whose entire contract is that it is lossless.
# --------------------------------------------------------------------------

def _jpeg_with_exif(path: Path) -> bytes:
    rng = np.random.default_rng(3)
    im = Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[274] = 6
    im.save(path, "JPEG", exif=exif, quality=95)
    return path.read_bytes()


def test_exif_strip_keeps_every_byte_it_did_not_mean_to_remove(tmp_path):
    data = _jpeg_with_exif(tmp_path / "src.jpg")

    # walk to a segment boundary before the scan, then end one byte past it
    i = 2
    while True:
        if data[i + 1] == 0xDA:
            break
        nxt = i + 2 + int.from_bytes(data[i + 2:i + 4], "big")
        if data[nxt + 1] == 0xDA:
            break
        i = nxt
    boundary = i + 2 + int.from_bytes(data[i + 2:i + 4], "big")

    p = tmp_path / "t.jpg"
    p.write_bytes(data[:boundary] + b"\xff")
    assert strip_exif_jpeg(p) is True
    assert p.read_bytes().endswith(b"\xff"), "the trailing byte was dropped"


def test_exif_strip_is_still_lossless_on_a_normal_file(tmp_path):
    p = tmp_path / "t.jpg"
    _jpeg_with_exif(p)
    before = Image.open(p).convert("RGB").tobytes()
    assert strip_exif_jpeg(p) is True
    assert Image.open(p).convert("RGB").tobytes() == before


# --------------------------------------------------------------------------
# An unchecked polygon read as a clean one.
#
# The "not checked for crossings" note was appended to the self-intersection
# finding, so when there were no crossings there was no finding to carry it
# and the limitation vanished entirely.
# --------------------------------------------------------------------------

def test_unchecked_polygons_are_reported_even_with_no_crossings(tmp_path):
    from dsdoctor.detectors.shapes import MAX_VERTICES_FOR_CROSSING_CHECK

    root = tmp_path / "seg"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8) + 128).save(
        root / "images" / "train" / "a.jpg")

    n = MAX_VERTICES_FOR_CROSSING_CHECK + 20
    pts = []
    for i in range(n):                       # a clean convex circle
        a = 2 * math.pi * i / n
        pts += [f"{0.5 + 0.4 * math.cos(a):.6f}", f"{0.5 + 0.4 * math.sin(a):.6f}"]
    (root / "labels" / "train" / "a.txt").write_text("0 " + " ".join(pts) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"names": ["t"], "nc": 1, "task": "segment"}))

    found = run("polygon_scan", Dataset(root))
    types = {f.type for f in found}
    assert "polygon_unverified" in types, "silence implied a pass"
    note = next(f for f in found if f.type == "polygon_unverified")
    assert "not known to be clean" in note.detail
    assert note.items == ["train/a"]


# --------------------------------------------------------------------------
# The template finding claimed more affected images than the dataset had.
#
# Per-signature counts were summed, so an image carrying two repeated boxes
# was counted twice.
# --------------------------------------------------------------------------

def test_template_count_is_images_not_occurrences(tmp_path):
    root = tmp_path / "ds"
    for i in range(8):
        write_sample(root, "train", f"t{i}",
                     ["0 0.100000 0.100000 0.100000 0.100000",
                      "0 0.900000 0.900000 0.100000 0.100000"], seed=i)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["a"], "nc": 1}))

    f = next(x for x in run("provenance_scan", Dataset(root))
             if x.type == "template_annotation")
    n_images = sum(1 for s in Dataset(root).samples if s.image_path)
    assert f.n_items == 8
    assert "8 image(s)" in f.title
    assert f.n_items <= n_images, "claimed more images than exist"


# --------------------------------------------------------------------------
# recheck hashed the whole dataset for a fingerprint it never printed.
# --------------------------------------------------------------------------

def test_recheck_does_not_fingerprint(clean_root, capsys):
    from dsdoctor.cli import main, EXIT_OK

    assert main(["card", str(clean_root)]) == EXIT_OK
    capsys.readouterr()

    calls = {"n": 0}
    real = card.manifest

    def counting(root, workers=8):
        calls["n"] += 1
        return real(root, workers)

    with mock.patch.object(card, "manifest", counting):
        assert main(["recheck", str(clean_root)]) == EXIT_OK
    assert calls["n"] == 0, "recheck hashed every file for a digest it discards"


# --------------------------------------------------------------------------
# The visual report outlined every box in red.
#
# For a finding about one box among twenty, the picture asserted something the
# finding did not.
# --------------------------------------------------------------------------

def test_report_marks_only_the_implicated_box(clean_root):
    from dsdoctor import htmlreport

    p = clean_root / "labels" / "train" / "t000.txt"
    p.write_text("0 0.2 0.2 0.1 0.1\n"
                 "1 0.5 0.5 0.1 0.1\n"
                 "2 0.9 0.5 0.6 0.2\n")        # only this one is out of bounds

    res = sweep(Dataset(clean_root))
    oob = next(f for f in res.findings if f.type == "out_of_bounds")
    assert oob.locations == {"train/t000": [3]}, \
        "the finding must record which row it is about"

    html = htmlreport.render(Dataset(clean_root), res.findings)
    assert "red outlines the annotation this finding is about" in html


def test_file_level_findings_do_not_claim_box_precision(clean_root):
    from dsdoctor import htmlreport

    (clean_root / "labels" / "train" / "t000.txt").write_text("")
    res = sweep(Dataset(clean_root))
    empty = next(f for f in res.findings if f.type == "empty_label_file")
    assert empty.locations is None
    html = htmlreport.render(Dataset(clean_root), res.findings)
    assert "about the file rather than one annotation" in html
