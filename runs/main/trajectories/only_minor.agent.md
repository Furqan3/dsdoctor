# Trajectory — `only_minor.agent`

- agent: **auditor**
- model: `qwen3.8-27b`
- 4 model call(s), 12 tool call(s)
- 11,146 prompt tokens, 1,448 completion tokens
- wall time: 41.28s

## Agent instructions

```text
You audit object-detection training datasets in YOLO format.

Your user is an ML engineer who has just inherited a labelled dataset and has to decide today whether to train on it. They are not asking for a list of everything technically imperfect. They need three answers: will this train at all, will the validation number mean anything, and what has to be fixed first.

How to work:

1. Call dataset_summary first, to see the scale, the splits and the classes.

2. Call list_detectors, then run the ones that matter. Each detector reports a `reliability`. Detectors marked exact read the files directly and use no language model, so what they say about the bytes on disk is fact, not opinion; run all of them, they are cheap. Detectors marked experimental are known to produce false positives - treat anything they return as a hypothesis you have to check, never as a finding on its own.

3. Detector output is not a report. What needs your judgement is:

   Ranking. A dataset with a train/val leak and a handful of oversized boxes has one urgent problem and one cosmetic one. Lead with what actually blocks the engineer, and say plainly what it costs them.

   Reading the evidence. Before you ra
... [1568 more chars]
```

## Task

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/only_minor. Decide whether it is safe to train on, and tell me what to fix first.

## Steps

### 1. model turn  ·  2.38s  ·  47 tokens out

<details><summary>reasoning</summary>

```text
Let's start by checking the dataset summary and the list of detectors.
```

</details>

**calls** `dataset_summary({})`

**calls** `list_detectors({})`

### 2. tool result  ·  `dataset_summary`

```json
{
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/only_minor",
  "num_classes": 12,
  "class_names": [
    "person",
    "chair",
    "car",
    "dining table",
    "cup",
    "bottle",
    "bowl",
    "handbag",
    "truck",
    "bench",
    "book",
    "backpack"
  ],
  "splits": {
    "train": {
      "images": 468,
      "label_files": 471,
      "boxes": 4920
    },
    "val": {
      "images": 132,
      "label_files": 132,
      "boxes": 1813
    }
  },
  "total_boxes": 6733,
  "data_yaml_problem": null
}
```

### 3. tool result  ·  `list_detectors`

```json
{
  "detectors": [
    {
      "name": "structure_scan",
      "description": "Pair images with label files and validate data.yaml. Detects missing_label_file, orphan_label_file, empty_label_file, malformed_label_row and yaml_inconsistency.",
      "detects": [
        "missing_label_file",
        "orphan_label_file",
        "empty_label_file",
        "malformed_label_row",
        "yaml_inconsistency"
      ],
      "cost": "fast",
      "reliability": "exact - reads the files directly",
      "already_run": false
    },
    {
      "name": "image_integrity_scan",
      "description": "Open every image header to find files that cannot be decoded. Detects corrupt_image. Reads pixels, so slower than structure_scan.",
      "detects": [
        "corrupt_image"
      ],
      "cost": "medium",
      "reliability": "exact - reads the files directly",
      "already_run": false
    },
… [2130 more chars]
```

### 4. model turn  ·  6.23s  ·  255 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate. Run them all. The dataset is small (468 training images, 132 validation images, 6733 boxes). Note: the training split has 468 images but 471 label files — 3 orphan label files, which is a structural issue.

Run all detectors in parallel.
```

</details>

**calls** `run_detector({"name": "structure_scan"})`

**calls** `run_detector({"name": "image_integrity_scan"})`

**calls** `run_detector({"name": "geometry_scan"})`

**calls** `run_detector({"name": "normalisation_scan"})`

**calls** `run_detector({"name": "class_scan"})`

**calls** `run_detector({"name": "duplicate_scan"})`

**calls** `class_distribution({})`

### 5. tool result  ·  `run_detector`

arguments: `{"name": "structure_scan"}`

```json
{
  "detector": "structure_scan",
  "findings_found": 2,
  "findings": [
    {
      "finding_id": "structure_scan:orphan_label_file:0",
      "type": "orphan_label_file",
      "detector_severity": "minor",
      "title": "3 label file(s) have no image",
      "affected_files": 3,
      "example_files": [
        "train/639462453486",
        "train/915238652806",
        "train/920332787315"
      ],
      "example_evidence": [
        "labels present, image absent: train/639462453486",
        "labels present, image absent: train/915238652806"
      ]
    },
    {
      "finding_id": "structure_scan:empty_label_file:1",
      "type": "empty_label_file",
      "detector_severity": "minor",
      "title": "4 label file(s) contain no boxes",
      "affected_files": 4,
      "example_files": [
        "train/000000145597",
        "train/000000376307",
        "train/000000482436"
      ]
… [237 more chars]
```

### 6. tool result  ·  `run_detector`

arguments: `{"name": "image_integrity_scan"}`

```json
{
  "detector": "image_integrity_scan",
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 7. tool result  ·  `run_detector`

arguments: `{"name": "geometry_scan"}`

```json
{
  "detector": "geometry_scan",
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 8. tool result  ·  `run_detector`

arguments: `{"name": "normalisation_scan"}`

```json
{
  "detector": "normalisation_scan",
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 9. tool result  ·  `run_detector`

arguments: `{"name": "class_scan"}`

```json
{
  "detector": "class_scan",
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 10. tool result  ·  `run_detector`

arguments: `{"name": "duplicate_scan"}`

```json
{
  "detector": "duplicate_scan",
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 11. tool result  ·  `class_distribution`

```json
{
  "instances_per_class_per_split": {
    "train": {
      "person": 1976,
      "chair": 410,
      "car": 577,
      "dining table": 240,
      "cup": 351,
      "bottle": 189,
      "bowl": 250,
      "handbag": 272,
      "truck": 201,
      "bench": 185,
      "book": 73,
      "backpack": 196
    },
    "val": {
      "person": 463,
      "chair": 321,
      "car": 47,
      "dining table": 139,
      "cup": 180,
      "bottle": 258,
      "bowl": 80,
      "handbag": 25,
      "truck": 15,
      "bench": 25,
      "book": 224,
      "backpack": 36
    }
  }
}
```

### 12. model turn  ·  5.22s  ·  175 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate (no experimental ones). Findings:
1. structure_scan:orphan_label_file:0 — 3 label files with no corresponding image (minor)
2. structure_scan:empty_label_file:1 — 4 label files with no boxes (minor)

Everything else is clean: no corrupted images, no geometry issues, no normalization issues, no class issues, no leakage.

Before making a judgment, let me inspect both findings to check the evidence.
```

</details>

**calls** `inspect_finding({"finding_id": "structure_scan:orphan_label_file:0"})`

**calls** `inspect_finding({"finding_id": "structure_scan:empty_label_file:1"})`

### 13. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "structure_scan:orphan_label_file:0"}`

```json
{
  "finding_id": "structure_scan:orphan_label_file:0",
  "type": "orphan_label_file",
  "meaning": "label file has no corresponding image",
  "detector": "structure_scan",
  "detector_severity": "minor",
  "title": "3 label file(s) have no image",
  "explanation": "These labels are never read during training. Usually the remains of a deleted or renamed image.",
  "affected_files": [
    "train/639462453486",
    "train/915238652806",
    "train/920332787315"
  ],
  "affected_file_count": 3,
  "evidence": [
    "labels present, image absent: train/639462453486",
    "labels present, image absent: train/915238652806",
    "labels present, image absent: train/920332787315"
  ],
  "proposed_fix": {
    "action": "delete_orphan_labels",
    "targets": [
      "train/639462453486",
      "train/915238652806",
      "train/920332787315"
    ]
  }
}
```

### 14. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "structure_scan:empty_label_file:1"}`

```json
{
  "finding_id": "structure_scan:empty_label_file:1",
  "type": "empty_label_file",
  "meaning": "label file exists but contains no boxes",
  "detector": "structure_scan",
  "detector_severity": "minor",
  "title": "4 label file(s) contain no boxes",
  "explanation": "Intentional background images are a legitimate technique, but an unintended empty file is a silently dropped annotation. Confirm these are deliberate.",
  "affected_files": [
    "train/000000145597",
    "train/000000376307",
    "train/000000482436",
    "val/000000045229"
  ],
  "affected_file_count": 4,
  "evidence": [
    "0 rows in train/000000145597.txt",
    "0 rows in train/000000376307.txt",
    "0 rows in train/000000482436.txt",
    "0 rows in val/000000045229.txt"
  ],
  "proposed_fix": {
    "action": "review_background_images",
    "targets": [
      "train/000000145597",
      "train/000000376307",
      "t
… [56 more chars]
```

### 15. model turn  ·  21.97s  ·  971 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate (no experimental ones). Findings:

1. orphan_label_file: 3 label files in train have no corresponding image. Minor. These are never read during training — harmless, just dead files.
2. empty_label_file: 4 label files with no boxes (3 in train, 1 in val). Minor. Could be intentional background images, or could be lost annotations. The one in val (val/000000045229) is worth noting — if it's an unlabeled image that actually contains objects, it slightly skews the val metrics (the model gets penalized for not predicting anything, or the image just becomes a background). But it's still minor.

Everything else is clean: no corrupted images, no geometry issues, no normali
… [1434 more chars]
```

</details>

**calls** `submit_report({"verdict": "usable_with_caveats", "headline": "Safe to train: no leakage, no corrupt images, no bad boxes — only 3 orphan label files and 4 empty label files to clean up or confirm.", "training_impac
… [1144 more chars])`

### 16. tool result  ·  `submit_report`

arguments: `{"verdict": "usable_with_caveats", "headline": "Safe to train: no leakage, no corrupt images, no bad boxes \u2014 only 3 orphan label files and 4 empty label files to clean up or confirm.", "training_`

```json
{
  "status": "report recorded",
  "decisions": 2
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
