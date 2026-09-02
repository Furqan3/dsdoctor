"""Duplicate and leakage detection via a perceptual hash.

Train/val leakage is the defect this whole tool most wants to catch. It never
throws an error, it never looks wrong in a viewer, and its only symptom is a
validation score that is quietly too good - which is the one number the team
uses to decide the model is ready.

**On scale.** The obvious implementation compares every pair, which is 179,700
comparisons at 600 images and about 50 million at 10,000 - the size at which
this check stops being run at all, and an unrun check catches nothing. So
candidates come from banded LSH instead.

That banding is *exact*, not approximate, and the reason is worth stating
because it is the whole justification for using it here. Split each 64-bit
hash into ``LSH_BANDS`` disjoint bands and bucket by each band separately. Two
hashes within Hamming distance ``d`` differ in at most ``d`` bits, so they can
disturb at most ``d`` bands. With ``LSH_BANDS > NEAR_DUP_DISTANCE`` at least
one band must survive untouched, and the pair therefore collides in that
band's bucket. No pair within the threshold can be missed. Candidates are then
re-checked with the true Hamming distance, so nothing outside it gets through
either. The pair set is identical to brute force - ``tests/test_detectors.py``
asserts exactly that - and the cost drops from quadratic to near-linear.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from ..dataset import Dataset, Sample, file_sha1, TRAIN_NAMES, VAL_NAMES
from ..findings import Finding, CRITICAL, MAJOR
from . import register

HASH_SIDE = 8            # produces a 64-bit difference hash
NEAR_DUP_DISTANCE = 5    # Hamming distance at/below which we call it a near-duplicate

# Strictly greater than NEAR_DUP_DISTANCE, which is what makes the banding
# lossless. Bands are as even as 64 bits allow: 11,11,11,11,10,10.
LSH_BANDS = 6

HASH_WORKERS = int(os.environ.get("DSDOCTOR_WORKERS", "8"))


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


# ------------------------------------------------------------------- cache

def _cache_path() -> Path | None:
    """Where hashes are remembered between runs.

    Deliberately **not** inside the dataset: `scan` and `audit` are documented
    as read-only, and a cache file written into the directory being audited
    would break that promise (and change the dataset's own fingerprint, which
    `dsdoctor card` then reports as a modification). Set DSDOCTOR_NO_CACHE=1
    to disable entirely.
    """
    if os.environ.get("DSDOCTOR_NO_CACHE"):
        return None
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    try:
        d = Path(base) / "dsdoctor"
        d.mkdir(parents=True, exist_ok=True)
        return d / "image-hashes-v1.json"
    except OSError:  # pragma: no cover - read-only or absent home directory
        return None


def _load_cache() -> dict:
    p = _cache_path()
    if p is None or not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):  # pragma: no cover - corrupt cache
        return {}


def _save_cache(cache: dict) -> None:
    p = _cache_path()
    if p is None:
        return
    try:
        # Bound the file so a long-lived cache cannot grow without limit.
        if len(cache) > 200_000:
            cache = dict(list(cache.items())[-200_000:])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(p)
    except OSError:  # pragma: no cover - unwritable cache directory
        pass


def _cache_key(path: Path) -> str | None:
    try:
        st = path.stat()
    except OSError:  # pragma: no cover - file vanished mid-scan
        return None
    return f"{path.resolve().as_posix()}|{st.st_size}|{st.st_mtime_ns}"


def _hash_all(ds: Dataset) -> dict[str, tuple[Sample, int, str]]:
    """Perceptual hash and sha1 for every readable image.

    Threaded because it is pure I/O plus a small decode, and cached on
    (path, size, mtime) because re-auditing a dataset after a fix should not
    re-read every byte of it.
    """
    cache = _load_cache()
    targets = [s for s in ds.samples if s.image_path is not None]

    def compute(s: Sample):
        key = _cache_key(s.image_path)
        if key is not None and key in cache:
            h, sha = cache[key]
            return s, (h if h is not None else None), sha
        h = dhash(s.image_path)
        sha = file_sha1(s.image_path) if h is not None else ""
        if key is not None:
            cache[key] = (h, sha)
        return s, h, sha

    table: dict[str, tuple[Sample, int, str]] = {}
    if targets:
        workers = max(1, min(HASH_WORKERS, len(targets)))
        from ..progress import track

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for s, h, sha in track(pool.map(compute, targets),
                                   "hashing images", total=len(targets)):
                if h is None:
                    continue  # image_integrity_scan owns reporting unreadable files
                table[s.key()] = (s, h, sha)
    _save_cache(cache)
    return table


# --------------------------------------------------------------- candidates

def _bands(value: int) -> list[int]:
    """Split a 64-bit hash into LSH_BANDS disjoint pieces."""
    total = HASH_SIDE * HASH_SIDE
    base, extra = divmod(total, LSH_BANDS)
    out, shift = [], 0
    for i in range(LSH_BANDS):
        width = base + (1 if i < extra else 0)
        out.append((value >> shift) & ((1 << width) - 1))
        shift += width
    return out


def _candidate_pairs(left: list[tuple[str, int]],
                     right: list[tuple[str, int]] | None = None
                     ) -> set[tuple[int, int]]:
    """Index pairs that could be within NEAR_DUP_DISTANCE.

    With `right` given, pairs cross the two collections (indices refer to
    left and right respectively); without it, pairs are within `left` and
    always ordered (i < j), matching `itertools.combinations`.
    """
    cross = right is not None
    other = right if cross else left

    buckets: list[dict[int, list[int]]] = [{} for _ in range(LSH_BANDS)]
    for j, (_k, h) in enumerate(other):
        for bi, bv in enumerate(_bands(h)):
            buckets[bi].setdefault(bv, []).append(j)

    pairs: set[tuple[int, int]] = set()
    for i, (_k, h) in enumerate(left):
        hit: set[int] = set()
        for bi, bv in enumerate(_bands(h)):
            hit.update(buckets[bi].get(bv, ()))
        for j in hit:
            if cross:
                pairs.add((i, j))
            elif i < j:
                pairs.add((i, j))
            elif j < i:
                pairs.add((j, i))
    return pairs


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

    # Evidence is ordered by the original enumeration order (val-major, then
    # train) rather than by whatever order the buckets happen to produce, so
    # that reports stay stable and comparable across runs and versions.
    val_items = list(val.items())
    train_items = list(train.items())

    leaks: list[str] = []
    leak_items: set[str] = set()
    for i, j in sorted(_candidate_pairs([(k, v[1]) for k, v in val_items],
                                        [(k, v[1]) for k, v in train_items])):
        vk, (vs, vh, vsha) = val_items[i]
        tk, (ts, th, tsha) = train_items[j]
        if vsha and vsha == tsha:
            leaks.append(f"{vk} is byte-identical to {tk} (sha1 {vsha[:12]})")
        else:
            d = _hamming(vh, th)
            if d > NEAR_DUP_DISTANCE:
                continue
            leaks.append(f"{vk} ~ {tk} (perceptual distance {d}/64)")
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
        pairs = _candidate_pairs([(k, v[1]) for k, v in members])
        for ia, ib in sorted(pairs):
            ka, (sa, ha, sha_a) = members[ia]
            kb, (sb, hb, sha_b) = members[ib]
            if sha_a and sha_a == sha_b:
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


def brute_force_pairs(hashes: list[int]) -> set[tuple[int, int]]:
    """Reference implementation, kept for the equivalence test only."""
    return {(i, j) for (i, a), (j, b) in combinations(enumerate(hashes), 2)
            if _hamming(a, b) <= NEAR_DUP_DISTANCE}
