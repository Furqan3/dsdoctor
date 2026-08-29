# Trajectory — `subtle_leak.agent`

- agent: **auditor**
- model: `qwen3.8-27b`
- 5 model call(s), 13 tool call(s)
- 15,858 prompt tokens, 1,396 completion tokens
- wall time: 41.46s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/subtle_leak. Decide whether it is safe to train on, and tell me what to fix first.

## Steps

### 1. model turn  ·  2.37s  ·  47 tokens out

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/subtle_leak",
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
      "label_files": 468,
      "boxes": 4940
    },
    "val": {
      "images": 135,
      "label_files": 135,
      "boxes": 1871
    }
  },
  "total_boxes": 6811,
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

### 4. model turn  ·  6.32s  ·  259 tokens out

<details><summary>reasoning</summary>

```text
The dataset is COCO-style, 12 classes, 468 training images, 135 validation images, 6811 boxes. The case name is "subtle_leak" — so train/val leakage is likely. Let's run all the detectors. They're all accurate, and most are low-cost. I'll run them all.
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
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
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
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "duplicate_scan:train_val_leakage:0",
      "type": "train_val_leakage",
      "detector_severity": "critical",
      "title": "3 image pair(s) leak between train and val",
      "affected_files": 6,
      "example_files": [
        "train/000000221754",
        "train/000000277005",
        "train/000000336232"
      ],
      "example_evidence": [
        "val/218885329401 is byte-identical to train/000000221754 (sha1 e0369a05082d)",
        "val/706346898744 is byte-identical to train/000000336232 (sha1 844893998f27)"
      ]
    }
  ],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 11. tool result  ·  `class_distribution`

```json
{
  "instances_per_class_per_split": {
    "train": {
      "person": 1983,
      "chair": 411,
      "car": 577,
      "dining table": 242,
      "cup": 355,
      "bottle": 192,
      "bowl": 251,
      "handbag": 274,
      "truck": 201,
      "bench": 185,
      "book": 73,
      "backpack": 196
    },
    "val": {
      "person": 485,
      "chair": 321,
      "car": 65,
      "dining table": 139,
      "cup": 180,
      "bottle": 263,
      "bowl": 81,
      "handbag": 25,
      "truck": 20,
      "bench": 25,
      "book": 228,
      "backpack": 39
    }
  }
}
```

### 12. model turn  ·  2.78s  ·  67 tokens out

<details><summary>reasoning</summary>

```text
The only finding is train_val_leakage, and it's from an exact detector. Let's inspect the evidence to confirm.
```

</details>

**calls** `inspect_finding({"finding_id": "duplicate_scan:train_val_leakage:0"})`

### 13. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "duplicate_scan:train_val_leakage:0"}`

```json
{
  "finding_id": "duplicate_scan:train_val_leakage:0",
  "type": "train_val_leakage",
  "meaning": "the same image content appears in both train and val",
  "detector": "duplicate_scan",
  "detector_severity": "critical",
  "title": "3 image pair(s) leak between train and val",
  "explanation": "The validation split contains content the model trained on. Every metric computed against it is optimistic by an unknown margin, so it cannot be used to compare checkpoints or to decide the model is ready to ship. Re-split before doing anything else - fixing this after training means retraining.",
  "affected_files": [
    "train/000000221754",
    "train/000000277005",
    "train/000000336232",
    "val/218885329401",
    "val/706346898744",
    "val/916278631571"
  ],
  "affected_file_count": 6,
  "evidence": [
    "val/218885329401 is byte-identical to train/000000221754 (sha1 e0369a05082d)",
… [338 more chars]
```

### 14. model turn  ·  5.46s  ·  227 tokens out

<details><summary>reasoning</summary>

```text
The leak is confirmed: 3 pairs of byte-identical images (sha1 hashes) exist between train and val. This is an exact detector, and the evidence is decisive.

Let me quickly verify one of the label files to confirm the leak is real (i.e., that the val image actually has annotations, and that the content is identical). In fact, the sha1 evidence is decisive — byte-identical. The detector is exact. No need to suppress.

Let me quickly check the label files of the leaked pairs to see whether the val side has annotations (if the val label is empty, the leak is less serious, but it's still a leak). I'll read a few.
```

</details>

**calls** `read_label_file({"file_key": "val/218885329401"})`

**calls** `read_label_file({"file_key": "train/000000221754"})`

### 15. tool result  ·  `read_label_file`

arguments: `{"file_key": "val/218885329401"}`

```json
{
  "file": "val/218885329401",
  "image_size": [
    640,
    427
  ],
  "image_error": null,
  "num_boxes": 10,
  "rows": [
    {
      "line": 1,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.265867,
      "yc": 0.631803,
      "w": 0.054546,
      "h": 0.04267
    },
    {
      "line": 2,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.149898,
      "yc": 0.620527,
      "w": 0.033704,
      "h": 0.028642
    },
    {
      "line": 3,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.123844,
      "yc": 0.623173,
      "w": 0.025156,
      "h": 0.024146
    },
    {
      "line": 4,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.391016,
      "yc": 0.645773,
      "w": 0.019844,
      "h": 0.06157
    },
    {
      "line": 5,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.803844,
      "yc": 0.65555,
… [895 more chars]
```

### 16. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000221754"}`

```json
{
  "file": "train/000000221754",
  "image_size": [
    640,
    427
  ],
  "image_error": null,
  "num_boxes": 10,
  "rows": [
    {
      "line": 1,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.265867,
      "yc": 0.631803,
      "w": 0.054546,
      "h": 0.04267
    },
    {
      "line": 2,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.149898,
      "yc": 0.620527,
      "w": 0.033704,
      "h": 0.028642
    },
    {
      "line": 3,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.123844,
      "yc": 0.623173,
      "w": 0.025156,
      "h": 0.024146
    },
    {
      "line": 4,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.391016,
      "yc": 0.645773,
      "w": 0.019844,
      "h": 0.06157
    },
    {
      "line": 5,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.803844,
      "yc": 0.65555,
… [897 more chars]
```

### 17. model turn  ·  19.1s  ·  796 tokens out

<details><summary>reasoning</summary>

```text
The leak is confirmed: the val image is byte-for-byte identical to the train image, and the label files are also identical (same 10 boxes, same coordinates). This is a genuine train/val leak.

All other detectors (structure, image integrity, geometry, normalization, class) returned zero findings. The class distribution is somewhat imbalanced (person is dominant, book is rare in train), but class_scan did not flag extreme_class_imbalance, so it's not a finding.

Therefore, there is exactly one finding: duplicate_scan:train_val_leakage:0, critical.

Verdict: blocked (a critical finding is being reported).

Wait — let me think about the verdict. "blocked = training will crash, or metrics become
… [680 more chars]
```

</details>

**calls** `submit_report({"verdict": "blocked", "headline": "3 val images are byte-identical to 3 train images (labels included) — your validation metrics are untrustworthy; remove the 3 val pairs and re-split before training
… [1120 more chars])`

### 18. tool result  ·  `submit_report`

arguments: `{"verdict": "blocked", "headline": "3 val images are byte-identical to 3 train images (labels included) \u2014 your validation metrics are untrustworthy; remove the 3 val pairs and re-split before tra`

```json
{
  "status": "report recorded",
  "decisions": 1
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
