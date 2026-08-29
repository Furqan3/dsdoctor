"""File-level integrity: are the images, labels and data.yaml even coherent?

These are the checks that decide whether a training run starts at all, so
almost everything here is critical.
"""

from __future__ import annotations

from ..dataset import Dataset
from ..findings import Finding, CRITICAL, MAJOR, MINOR
from . import register


@register("structure_scan",
          "Pair images with label files and validate data.yaml. Detects "
          "missing_label_file, orphan_label_file, empty_label_file, "
          "malformed_label_row and yaml_inconsistency.",
          covers=("missing_label_file", "orphan_label_file", "empty_label_file",
                  "malformed_label_row", "yaml_inconsistency"))
def structure_scan(ds: Dataset) -> list[Finding]:
    out: list[Finding] = []
    missing, orphan, empty = [], [], []
    malformed: list[tuple[str, str]] = []

    for s in ds.samples:
        if s.image_path and not s.label_path:
            missing.append(s.key())
        elif s.label_path and not s.image_path:
            orphan.append(s.key())
        if s.label is not None:
            if s.label.n_boxes == 0 and not s.label.parse_errors:
                empty.append(s.key())
            for pe in s.label.parse_errors:
                malformed.append((s.key(), f"line {pe.line_no}: {pe.reason} -> {pe.raw!r}"))

    if missing:
        out.append(Finding(
            type="missing_label_file", severity=MAJOR,
            title=f"{len(missing)} image(s) have no label file",
            detail=("Ultralytics treats an image with no .txt as a pure background "
                    "image. If these are actually unlabelled foreground images the "
                    "model is being explicitly taught that the objects in them are "
                    "background, which suppresses recall for those classes."),
            detector="structure_scan", items=sorted(missing),
            evidence=[f"no labels/{k.split('/')[0]}/{k.split('/')[1]}.txt" for k in sorted(missing)[:10]],
            fix={"action": "create_empty_or_annotate", "targets": sorted(missing)}))

    if orphan:
        out.append(Finding(
            type="orphan_label_file", severity=MINOR,
            title=f"{len(orphan)} label file(s) have no image",
            detail="These labels are never read during training. Usually the "
                   "remains of a deleted or renamed image.",
            detector="structure_scan", items=sorted(orphan),
            evidence=[f"labels present, image absent: {k}" for k in sorted(orphan)[:10]],
            fix={"action": "delete_orphan_labels", "targets": sorted(orphan)}))

    if empty:
        out.append(Finding(
            type="empty_label_file", severity=MINOR,
            title=f"{len(empty)} label file(s) contain no boxes",
            detail="Intentional background images are a legitimate technique, but "
                   "an unintended empty file is a silently dropped annotation. "
                   "Confirm these are deliberate.",
            detector="structure_scan", items=sorted(empty),
            evidence=[f"0 rows in {k}.txt" for k in sorted(empty)[:10]],
            fix={"action": "review_background_images", "targets": sorted(empty)}))

    if malformed:
        keys = sorted({k for k, _ in malformed})
        out.append(Finding(
            type="malformed_label_row", severity=CRITICAL,
            title=f"{len(malformed)} label row(s) are not parseable",
            detail="A YOLO row must be `class_id xc yc w h`. Rows that are not "
                   "will raise during dataset scanning or be skipped silently "
                   "depending on the loader version.",
            detector="structure_scan", items=keys,
            evidence=[f"{k}: {msg}" for k, msg in malformed[:10]],
            fix={"action": "repair_or_drop_rows", "targets": keys}))

    if ds.yaml_error:
        out.append(Finding(
            type="yaml_inconsistency", severity=MAJOR,
            title="data.yaml is missing or inconsistent",
            detail=ds.yaml_error,
            detector="structure_scan", items=[],
            evidence=[ds.yaml_error],
            fix={"action": "fix_data_yaml", "targets": []}))

    return out


@register("image_integrity_scan",
          "Open every image header to find files that cannot be decoded. "
          "Detects corrupt_image. Reads pixels, so slower than structure_scan.",
          reads_pixels=True, covers=("corrupt_image",))
def image_integrity_scan(ds: Dataset) -> list[Finding]:
    bad: list[tuple[str, str]] = []
    for s in ds.samples:
        if s.image_path is None:
            continue
        ds.verify_decodable(s)
        if s.image_error:
            bad.append((s.key(), s.image_error))
    if not bad:
        return []
    return [Finding(
        type="corrupt_image", severity=CRITICAL,
        title=f"{len(bad)} image(s) cannot be decoded",
        detail="A corrupt image aborts the epoch in most training loops, or is "
               "skipped with a warning that is easy to miss in a long log.",
        detector="image_integrity_scan", items=[k for k, _ in bad],
        evidence=[f"{k}: {err}" for k, err in bad[:10]],
        fix={"action": "remove_corrupt_images", "targets": [k for k, _ in bad]})]
