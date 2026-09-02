"""Build the clean base corpus that every evaluation case is derived from.

The point of this script is a dataset with *no* defects in it. Recall and
false-positive rate only mean something if we know exactly what is wrong with
the input, so we start from public COCO data, keep the classes that are well
enough populated to actually train on, repair or discard everything a detector
would legitimately complain about, and then assert that a full detector sweep
comes back empty. If it does not, this script fails rather than shipping a
corpus with unknown defects in it.

Source data (public, downloaded at build time, never redistributed here):
  * YOLO-format labels for COCO 2017 - Ultralytics `coco2017labels.zip`
  * the val2017 JPEGs themselves - images.cocodataset.org
COCO images are Flickr-sourced under their original licences; the annotations
are CC BY 4.0.

    python eval/build_corpus.py --images 600 --out data/corpus_clean
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from dsdoctor.dataset import Dataset  # noqa: E402
from dsdoctor.detectors import available            # noqa: E402
from dsdoctor.detectors.duplicates import dhash, _hamming  # noqa: E402

LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{stem}.jpg"

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

N_CLASSES = 12
MIN_SIDE = 0.01            # stricter than the tiny_box detector's own threshold
NEAR_DUP_DISTANCE = 5
VAL_FRACTION = 0.22
VAL_CAP_FRACTION = 0.40
MIN_VAL_INSTANCES = 4      # detector wants 3; leave a margin


# --------------------------------------------------------------------- source

def fetch_labels(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    lbl_dir = cache / "coco" / "labels" / "val2017"
    if lbl_dir.is_dir():
        return lbl_dir
    zpath = cache / "coco2017labels.zip"
    if not zpath.is_file():
        print(f"downloading COCO YOLO labels (~48 MB) ...")
        urllib.request.urlretrieve(LABELS_URL, zpath)
    print("extracting val2017 labels ...")
    with zipfile.ZipFile(zpath) as zf:
        members = [m for m in zf.namelist() if "/labels/val2017/" in m and m.endswith(".txt")]
        zf.extractall(cache, members=members)
    if not lbl_dir.is_dir():  # archive layout differs between releases
        found = next((p for p in (cache).rglob("labels/val2017") if p.is_dir()), None)
        if found is None:
            raise SystemExit("could not locate labels/val2017 inside the archive")
        return found
    return lbl_dir


def read_labels(lbl_dir: Path) -> dict[str, list]:
    out: dict[str, list] = {}
    for p in sorted(lbl_dir.iterdir()):
        if p.suffix != ".txt":
            continue
        rows = []
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                rows.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
            except ValueError:
                continue
        if rows:
            out[p.stem] = rows
    return out


def download_images(stems: list[str], dst: Path, workers: int = 16) -> set[str]:
    dst.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (dst / f"{s}.jpg").is_file()]
    if todo:
        print(f"downloading {len(todo)} image(s) from images.cocodataset.org ...")

    ok: set[str] = {s for s in stems if (dst / f"{s}.jpg").is_file()}

    def grab(stem: str):
        target = dst / f"{stem}.jpg"
        try:
            urllib.request.urlretrieve(IMAGE_URL.format(stem=stem), target)
            return stem
        except Exception:
            target.unlink(missing_ok=True)
            return None

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, res in enumerate(pool.map(grab, todo), 1):
                if res:
                    ok.add(res)
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)}")
    return ok


# ---------------------------------------------------------------- selection

def choose_classes(raw: dict, k: int) -> list[int]:
    images = Counter()
    for rows in raw.values():
        for cls in {c for c, *_ in rows}:
            images[cls] += 1
    return [c for c, _ in images.most_common(k)]


def select_images(raw: dict, keep: set[int], n: int, seed: int) -> list[str]:
    """Greedily pick images so the rare classes are not left starved.

    Score each candidate by how much it helps the classes we have least of;
    1/(1+count) makes an image carrying a scarce class outrank an image
    carrying twenty more people.
    """
    rng = random.Random(seed)
    pool = [s for s, rows in raw.items() if {c for c, *_ in rows} & keep]
    rng.shuffle(pool)
    counts: Counter = Counter()
    chosen: list[str] = []
    remaining = pool[:]

    while remaining and len(chosen) < n:
        best, best_score = None, -1.0
        # Scanning the whole pool every pick is O(n*pool); at these sizes that
        # is a couple of seconds and keeps the selection easy to reason about.
        for stem in remaining:
            score = sum(1.0 / (1.0 + counts[c])
                        for c in {c for c, *_ in raw[stem]} & keep)
            if score > best_score:
                best, best_score = stem, score
        chosen.append(best)
        remaining.remove(best)
        for c, *_ in raw[best]:
            if c in keep:
                counts[c] += 1
    return sorted(chosen)


def sanitise(rows, keep_map: dict[int, int]):
    """Keep the selected classes and drop anything a detector would flag."""
    out, seen = [], set()
    for cls, xc, yc, w, h in rows:
        if cls not in keep_map:
            continue
        x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
        x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
        # Quantise the *corners*, then derive centre and size from them.
        # Rounding xc and w independently - or letting the writer re-round to
        # 6dp - can push xc + w/2 a few 1e-7 past 1.0, and out_of_bounds is
        # right to flag that. Corners at 6dp put xc on 7dp and w on 6dp, both
        # exact at the 8dp the writer uses.
        x1, y1, x2, y2 = round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)
        nw, nh = x2 - x1, y2 - y1
        if nw < MIN_SIDE or nh < MIN_SIDE:
            continue
        nxc, nyc = (x1 + x2) / 2, (y1 + y2) / 2
        sig = (keep_map[cls], round(nxc, 5), round(nyc, 5), round(nw, 5), round(nh, 5))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((keep_map[cls], nxc, nyc, nw, nh))
    return out


def drop_near_duplicates(stems: list[str], img_dir: Path) -> list[str]:
    hashes: dict[str, int] = {}
    for stem in stems:
        p = img_dir / f"{stem}.jpg"
        if not p.is_file():
            continue
        h = dhash(p)
        if h is not None:
            hashes[stem] = h
    kept: list[str] = []
    for stem, h in hashes.items():
        if any(_hamming(h, hashes[k]) <= NEAR_DUP_DISTANCE for k in kept):
            continue
        kept.append(stem)
    return sorted(kept)


def split(labels: dict[str, list], n_classes: int, seed: int):
    """Grow the val split until every class clears its quota.

    A fixed 80/20 cut leaves rare classes with one or two validation
    instances, which the class detector correctly calls unusable. So the
    fraction is a floor, not a target: keep promoting whichever image serves
    the most still-unmet classes until the quotas are met or we hit the cap.
    """
    rng = random.Random(seed)
    stems = sorted(labels)
    rng.shuffle(stems)
    floor = max(1, int(len(stems) * VAL_FRACTION))
    cap = max(floor, int(len(stems) * VAL_CAP_FRACTION))

    val: list[str] = []
    counts: Counter = Counter()

    def unmet() -> set[int]:
        return {c for c in range(n_classes) if counts[c] < MIN_VAL_INSTANCES}

    remaining = list(stems)
    while remaining and len(val) < cap and (unmet() or len(val) < floor):
        need = unmet()
        if need:
            remaining.sort(key=lambda s: -len({c for c, *_ in labels[s]} & need))
            if not ({c for c, *_ in labels[remaining[0]]} & need):
                break
        pick = remaining.pop(0)
        val.append(pick)
        for c, *_ in labels[pick]:
            counts[c] += 1

    return sorted(set(stems) - set(val)), sorted(val)


# ------------------------------------------------------------------- output

def write_split(dst: Path, split_name: str, stems: list[str],
                labels: dict[str, list], src_img: Path) -> None:
    (dst / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = src_img / f"{stem}.jpg"
        if not src.is_file():
            continue
        shutil.copy2(src, dst / "images" / split_name / src.name)
        body = "\n".join(f"{c} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}"
                         for c, xc, yc, w, h in labels[stem])
        (dst / "labels" / split_name / f"{stem}.txt").write_text(body + "\n")


def materialise(dst: Path, train, val, labels, names, img_dir) -> Dataset:
    if dst.exists():
        shutil.rmtree(dst)
    write_split(dst, "train", train, labels, img_dir)
    write_split(dst, "val", val, labels, img_dir)
    (dst / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "images/train", "val": "images/val",
         "nc": len(names), "names": names}, sort_keys=False))
    return Dataset(dst)


def sweep(ds: Dataset):
    # Core group only: "provably clean" is a claim about the checks the
    # evaluation scores against, and an opt-in group must not be able to
    # redefine it. `missing_license` in particular is true of this corpus and
    # is not a defect the injector can create or the scorer can match.
    out = []
    for det in available():
        if det.heavy:
            continue
        out.extend(det.fn(ds))
    return out


def drop_classes(labels: dict, names: list[str], drop: set[int]):
    remap, new_names = {}, []
    for c, n in enumerate(names):
        if c in drop:
            continue
        remap[c] = len(new_names)
        new_names.append(n)
    new_labels = {s: [(remap[c], *rest) for c, *rest in rows if c in remap]
                  for s, rows in labels.items()}
    return {s: r for s, r in new_labels.items() if r}, new_names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpus_clean")
    ap.add_argument("--cache", default="data/_cache")
    ap.add_argument("--images", type=int, default=600)
    ap.add_argument("--classes", type=int, default=N_CLASSES)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cache = Path(args.cache)
    lbl_dir = fetch_labels(cache)
    raw = read_labels(lbl_dir)
    print(f"source: {len(raw)} val2017 label files")

    keep = choose_classes(raw, args.classes)
    keep_map = {orig: i for i, orig in enumerate(keep)}
    names = [COCO80[c] for c in keep]
    print(f"candidate classes ({len(names)}): {', '.join(names)}")

    stems = select_images(raw, set(keep), args.images, args.seed)
    print(f"selected {len(stems)} images")

    img_cache = cache / "coco" / "images" / "val2017"
    have = download_images(stems, img_cache)
    stems = [s for s in stems if s in have]
    print(f"have {len(stems)} images on disk")

    labels = {s: sanitise(raw[s], keep_map) for s in stems}
    labels = {s: r for s, r in labels.items() if r}
    print(f"after class filter + sanitise: {len(labels)} images, "
          f"{sum(len(v) for v in labels.values())} boxes")

    kept = drop_near_duplicates(sorted(labels), img_cache)
    labels = {s: labels[s] for s in kept}
    print(f"after near-duplicate removal: {len(labels)} images")

    # The detector suite is the oracle for what "clean" means: rather than
    # hand-tuning thresholds until they agree, let it name the classes that
    # are too thin and drop them until a full sweep comes back empty.
    dst = Path(args.out)
    ds = None
    for attempt in range(1, 12):
        train, val = split(labels, len(names), args.seed)
        ds = materialise(dst, train, val, labels, names, img_cache)
        residual = sweep(ds)
        print(f"\n[attempt {attempt}] {len(names)} classes, "
              f"{len(train)} train / {len(val)} val -> {len(residual)} finding(s)")
        if not residual:
            break
        starved, other = set(), []
        for f in residual:
            if f.type == "extreme_class_imbalance" and f.fix:
                starved.update(f.fix.get("class_ids", []))
            else:
                other.append(f)
        if other:
            for f in other:
                print("  UNFIXABLE:", f.short())
            print("\nFAILED: residual defects the builder cannot resolve.")
            return 1
        if not starved:
            print("\nFAILED: imbalance reported but no class ids returned.")
            return 1
        print(f"  dropping {len(starved)} thin class(es): "
              f"{', '.join(names[c] for c in sorted(starved))}")
        labels, names = drop_classes(labels, names, starved)
        if not names:
            print("\nFAILED: every class was dropped.")
            return 1
    else:
        print("\nFAILED: did not converge on a clean corpus.")
        return 1

    s = ds.summary()
    n_det = len([d for d in available() if not d.heavy])
    print("\n--- clean corpus ---")
    print(f"  0 findings across {n_det} detectors")
    print(f"  classes ({s['nc']}): {', '.join(s['names'])}")
    print(f"  splits: {s['splits']}")
    print(f"  boxes: {s['total_boxes']}")
    print(f"  per class: {s['class_counts']}")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
