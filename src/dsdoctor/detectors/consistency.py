"""Label-vs-model consistency: the only detector that needs a neural network.

We run a small pretrained COCO detector over each image, match its predictions
to the stored labels by IoU, and flag boxes where the model is confident and
disagrees about the class. This is the standard way to surface systematic
mislabels (a whole batch exported with two classes swapped) that no amount of
geometry checking can see.

Honest caveat, repeated in the README: the reference detector was trained on
COCO and our evaluation corpus is drawn from COCO, so this detector is
unusually strong here. On a genuinely custom dataset it degrades to a weak
prior, which is exactly why it is registered as optional and why the agent is
told to treat its output as a hypothesis rather than as ground truth.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from ..dataset import Dataset, file_sha1
from ..findings import Finding, MAJOR
from . import register

CONF_THRESHOLD = 0.55     # only trust confident predictions
IOU_MATCH = 0.60          # localisation must clearly agree before we compare class
MIN_GROUP = 2             # one odd box is noise; a repeated pattern is a defect
_MODEL_CACHE: dict[str, object] = {}

# Predictions depend only on the image bytes, and every evaluation case reuses
# the same 600 images, so caching by content hash turns a 33s scan into a
# lookup. Keyed by sha1 so a mutated (e.g. truncated) image misses correctly.
CACHE_PATH = Path(os.environ.get(
    "DSDOCTOR_PRED_CACHE", "data/_cache/yolo_preds.json"))
_PRED_CACHE: dict[str, list] | None = None


def _load_cache() -> dict:
    global _PRED_CACHE
    if _PRED_CACHE is None:
        try:
            _PRED_CACHE = json.loads(CACHE_PATH.read_text())
        except (OSError, ValueError):
            _PRED_CACHE = {}
    return _PRED_CACHE


def _save_cache() -> None:
    if _PRED_CACHE is None:
        return
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_PRED_CACHE))
    except OSError:
        pass


def _load_model(weights: str):
    if weights not in _MODEL_CACHE:
        from ultralytics import YOLO
        os.environ.setdefault("YOLO_VERBOSE", "false")
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@register("model_disagreement_scan",
          "Run a pretrained COCO detector and compare its confident predictions "
          "against the stored labels, by IoU match. Surfaces systematic "
          "mislabels (class_swap) that geometry checks cannot see. Requires the "
          "`vision` extra; slowest detector by a wide margin.",
          reads_pixels=True, heavy=True, experimental=True,
          covers=("class_swap",))
def model_disagreement_scan(ds: Dataset, weights: str = "yolov8n.pt") -> list[Finding]:
    try:
        model = _load_model(weights)
    except Exception as exc:  # pragma: no cover - missing extra or weights
        return [Finding(
            type="class_swap", severity=MAJOR,
            title="model consistency check could not run",
            detail=f"Reference detector unavailable ({type(exc).__name__}: {exc}). "
                   "Install the `vision` extra to enable mislabel detection.",
            detector="model_disagreement_scan", items=[], evidence=[str(exc)],
            fix=None, verified=False)]

    # Our datasets keep COCO class *names*, so we can align the reference
    # detector's label space to ours by name rather than by index.
    ref_names = {i: n for i, n in model.names.items()}
    ours = {n: i for i, n in enumerate(ds.names)}

    # disagreement pattern -> list of evidence strings
    swaps: dict[tuple[str, str], list[str]] = defaultdict(list)
    swap_items: dict[tuple[str, str], set[str]] = defaultdict(set)

    cache = _load_cache()
    dirty = False
    targets = [s for s in ds.samples if s.image_path and s.label and s.label.boxes]
    for s in targets:
        try:
            digest = file_sha1(s.image_path)
        except OSError:
            continue
        if digest in cache:
            preds = [((p[0], p[1], p[2], p[3]), p[4], p[5]) for p in cache[digest]]
        else:
            try:
                res = model.predict(str(s.image_path), conf=CONF_THRESHOLD,
                                    verbose=False)[0]
            except Exception:
                continue
            preds = []
            w, h = res.orig_shape[1], res.orig_shape[0]
            for box in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                preds.append(((x1 / w, y1 / h, x2 / w, y2 / h),
                              ref_names.get(int(box.cls[0]), "?"),
                              float(box.conf[0])))
            cache[digest] = [[*b, n, c] for b, n, c in preds]
            dirty = True
        if not preds:
            continue

        for b in s.label.boxes:
            gt_name = ds.class_name(b.cls)
            best, best_iou = None, 0.0
            for pxyxy, pname, pconf in preds:
                v = _iou(b.xyxy, pxyxy)
                if v > best_iou:
                    best, best_iou = (pname, pconf), v
            if best is None or best_iou < IOU_MATCH:
                continue
            pname, pconf = best
            # Only a disagreement if the predicted class is one we also define.
            if pname == gt_name or pname not in ours:
                continue
            key = (gt_name, pname)
            swaps[key].append(
                f"{s.key()} line {b.line_no}: labelled '{gt_name}' but the "
                f"reference detector says '{pname}' at conf {pconf:.2f} "
                f"(IoU {best_iou:.2f})")
            swap_items[key].add(s.key())

    if dirty:
        _save_cache()

    out: list[Finding] = []
    for (gt_name, pred_name), rows in sorted(swaps.items(),
                                             key=lambda kv: -len(kv[1])):
        if len(rows) < MIN_GROUP:
            continue
        items = sorted(swap_items[(gt_name, pred_name)])
        out.append(Finding(
            type="class_swap", severity=MAJOR,
            title=f"{len(rows)} box(es) labelled '{gt_name}' look like '{pred_name}'",
            detail=f"A confident reference detector disagrees with the stored "
                   f"class on {len(rows)} well-localised boxes across "
                   f"{len(items)} file(s), always in the same direction "
                   f"('{gt_name}' -> '{pred_name}'). A one-directional pattern at "
                   f"this scale is an export or annotation-tool mapping error "
                   f"rather than annotator noise. Confirm against a handful of "
                   f"images before remapping.",
            detector="model_disagreement_scan", items=items,
            evidence=rows[:12],
            fix={"action": "review_class_remap", "targets": items,
                 "from": gt_name, "to": pred_name}))
    return out
