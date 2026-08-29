"""Apply an approved fix plan. Nothing here runs without an explicit yes.

A dataset audit that silently rewrites label files is a worse tool than no
audit, so this module is built around three rules:

  1. Nothing is touched until a human has seen the plan and confirmed it.
  2. Steps flagged `requires_human_review` are never applied automatically at
     all - a class remap in particular is a judgement about what the images
     actually show, and the tool has not seen the images.
  3. Every run writes a backup of each file it modifies before modifying it,
     and prints where the backup went.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .dataset import Dataset

# Actions this module knows how to carry out. Anything else is described to
# the user and left for them to do.
AUTOMATABLE = {
    "drop_degenerate_boxes",
    "drop_tiny_boxes",
    "dedupe_annotations",
    "clip_or_drop_boxes",
    "normalise_coordinates",
    "repair_or_drop_rows",
    "delete_orphan_labels",
    "remove_corrupt_images",
}


def summarise(plan: dict) -> str:
    L = [f"Fix plan for {plan['dataset']}",
         f"  verdict: {plan['verdict']}",
         f"  generated: {plan['generated']}", ""]
    auto = manual = 0
    for i, step in enumerate(plan.get("steps", []), 1):
        n = len(step.get("targets") or [])
        if step.get("requires_human_review") or step["action"] not in AUTOMATABLE:
            manual += 1
            mark = "MANUAL "
        else:
            auto += 1
            mark = "auto   "
        L.append(f"  {i:2d}. [{mark}] {step['action']:<26s} "
                 f"{n:4d} file(s)  ({step['type']})")
        L.append(f"        {step['why']}")
    L += ["", f"  {auto} step(s) can be applied automatically, "
              f"{manual} need you to decide."]
    return "\n".join(L)


def apply_plan(plan_path: str | Path, *, assume_yes: bool = False,
               backup_root: str | Path | None = None) -> dict:
    plan = json.loads(Path(plan_path).read_text())
    print(summarise(plan))

    steps = [s for s in plan.get("steps", [])
             if s["action"] in AUTOMATABLE and not s.get("requires_human_review")]
    if not steps:
        print("\nNothing can be applied automatically. No files were touched.")
        return {"applied": 0, "files_changed": 0}

    if not assume_yes:
        print("\nThis will modify label and image files in place.")
        try:
            answer = input("Type 'apply' to proceed, anything else to abort: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "apply":
            print("Aborted. Nothing was changed.")
            return {"applied": 0, "files_changed": 0, "aborted": True}

    ds = Dataset(plan["dataset"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = Path(backup_root or (ds.root.parent / f"{ds.root.name}.backup-{stamp}"))
    changed: set[str] = set()

    for step in steps:
        for key in step.get("targets") or []:
            sample = ds.get(key)
            if sample is None:
                continue
            if step["action"] == "remove_corrupt_images":
                if sample.image_path and sample.image_path.exists():
                    _backup(sample.image_path, ds.root, backup)
                    sample.image_path.unlink()
                    changed.add(key)
                    # Take the label with it. Leaving it behind just turns one
                    # corrupt image into one orphaned label on the next scan.
                    if sample.label_path and sample.label_path.exists():
                        _backup(sample.label_path, ds.root, backup)
                        sample.label_path.unlink()
                continue
            if step["action"] == "delete_orphan_labels":
                if sample.label_path and sample.label_path.exists():
                    _backup(sample.label_path, ds.root, backup)
                    sample.label_path.unlink()
                    changed.add(key)
                continue
            if sample.label_path and sample.label_path.exists():
                if _rewrite(sample, ds, step["action"], backup):
                    changed.add(key)

    print(f"\nApplied {len(steps)} step(s) across {len(changed)} file(s).")
    print(f"Originals saved to {backup}")
    return {"applied": len(steps), "files_changed": len(changed),
            "backup": str(backup)}


def _backup(path: Path, root: Path, backup: Path) -> None:
    rel = path.relative_to(root)
    dst = backup / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(path, dst)


def _rewrite(sample, ds: Dataset, action: str, backup: Path) -> bool:
    from .detectors.geometry import TINY_SIDE, EPS

    ds.ensure_image_meta(sample)
    rows_in = sample.label.boxes
    out: list[str] = []
    seen: set[tuple] = set()
    changed = False

    for b in rows_in:
        cls, xc, yc, w, h = b.cls, b.xc, b.yc, b.w, b.h

        if action == "normalise_coordinates" and sample.width and sample.height:
            if max(xc, yc, w, h) > 1.5:
                xc, w = xc / sample.width, w / sample.width
                yc, h = yc / sample.height, h / sample.height
                changed = True

        if action == "drop_degenerate_boxes" and (w <= EPS or h <= EPS):
            changed = True
            continue
        if action == "drop_tiny_boxes" and (w < TINY_SIDE or h < TINY_SIDE):
            changed = True
            continue
        if action == "clip_or_drop_boxes":
            x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
            x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
            nw, nh = x2 - x1, y2 - y1
            if nw <= EPS or nh <= EPS:
                changed = True
                continue
            if abs(nw - w) > 1e-9 or abs(nh - h) > 1e-9:
                changed = True
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, nw, nh
        if action == "dedupe_annotations":
            sig = (cls, round(xc, 6), round(yc, 6), round(w, 6), round(h, 6))
            if sig in seen:
                changed = True
                continue
            seen.add(sig)

        out.append(f"{cls} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")

    if action == "repair_or_drop_rows" and sample.label.parse_errors:
        changed = True   # unparseable rows are simply not re-emitted

    if not changed:
        return False
    _backup(sample.label_path, ds.root, backup)
    sample.label_path.write_text("\n".join(out) + ("\n" if out else ""))
    return True
