"""Duplicate and leakage detection via a perceptual hash.

Train/val leakage is the defect this whole tool most wants to catch. It never
throws an error, it never looks wrong in a viewer, and its only symptom is a
validation score that is quietly too good - which is the one number the team
uses to decide the model is ready.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from PIL import Image

from ..dataset import Dataset, Sample, file_sha1
from ..findings import Finding, CRITICAL, MAJOR
from . import register

HASH_SIDE = 8            # produces a 64-bit difference hash
NEAR_DUP_DISTANCE = 5    # Hamming distance at/below which we call it a near-duplicate
TRAIN_NAMES = {"train", "train2017", "training"}
VAL_NAMES = {"val", "valid", "val2017", "validation", "test"}


def dhash(path) -> int | None:
    """Row-wise difference hash: robust to rescaling and mild recompression."""
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((HASH_SIDE + 1, HASH_SIDE),
                                           Image.Resampling.LANCZOS)
            arr = np.asarray(small, dtype=np.int16)
    except Exception:
        return None
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hash_all(ds: Dataset) -> dict[str, tuple[Sample, int, str]]:
    table: dict[str, tuple[Sample, int, str]] = {}
    for s in ds.samples:
        if s.image_path is None:
            continue
        h = dhash(s.image_path)
        if h is None:
            continue  # image_integrity_scan owns reporting unreadable files
        table[s.key()] = (s, h, file_sha1(s.image_path))
    return table


@register("duplicate_scan",
          "Perceptually hash every image to find near-identical pairs inside a "
          "split and identical content shared across splits. Detects "
          "train_val_leakage and near_duplicate_image. Reads pixels.",
          reads_pixels=True,
          covers=("train_val_leakage", "near_duplicate_image"))
def duplicate_scan(ds: Dataset) -> list[Finding]:
    table = _hash_all(ds)
    out: list[Finding] = []

    train = {k: v for k, v in table.items() if v[0].split in TRAIN_NAMES}
    val = {k: v for k, v in table.items() if v[0].split in VAL_NAMES}

    leaks: list[str] = []
    leak_items: set[str] = set()
    for vk, (vs, vh, vsha) in val.items():
        for tk, (ts, th, tsha) in train.items():
            if vsha == tsha:
                leaks.append(f"{vk} is byte-identical to {tk} (sha1 {vsha[:12]})")
            else:
                d = _hamming(vh, th)
                if d <= NEAR_DUP_DISTANCE:
                    leaks.append(f"{vk} ~ {tk} (perceptual distance {d}/64)")
                else:
                    continue
            leak_items.update((vk, tk))

    if leaks:
        out.append(Finding(
            type="train_val_leakage", severity=CRITICAL,
            title=f"{len(leaks)} image pair(s) leak between train and val",
            detail="The validation split contains content the model trained on. "
                   "Every metric computed against it is optimistic by an unknown "
                   "margin, so it cannot be used to compare checkpoints or to "
                   "decide the model is ready to ship. Re-split before doing "
                   "anything else - fixing this after training means retraining.",
            detector="duplicate_scan", items=sorted(leak_items),
            evidence=leaks[:15],
            fix={"action": "resplit_removing_leaks",
                 "targets": sorted(k for k in leak_items if table[k][0].split in VAL_NAMES)}))

    near: list[str] = []
    near_items: set[str] = set()
    for split in ds.splits:
        members = [(k, v) for k, v in table.items() if v[0].split == split]
        for (ka, (sa, ha, sha_a)), (kb, (sb, hb, sha_b)) in combinations(members, 2):
            if sha_a == sha_b:
                near.append(f"{ka} and {kb} are byte-identical (sha1 {sha_a[:12]})")
            else:
                d = _hamming(ha, hb)
                if d > NEAR_DUP_DISTANCE:
                    continue
                near.append(f"{ka} ~ {kb} (perceptual distance {d}/64)")
            near_items.update((ka, kb))

    if near:
        out.append(Finding(
            type="near_duplicate_image", severity=MAJOR,
            title=f"{len(near)} near-duplicate image pair(s) inside a split",
            detail="Duplicates inside train over-weight whatever they contain and "
                   "inflate the dataset size the team reports. Duplicates inside "
                   "val make one scene count several times toward the score.",
            detector="duplicate_scan", items=sorted(near_items),
            evidence=near[:15],
            fix={"action": "deduplicate_images", "targets": sorted(near_items)}))

    return out
