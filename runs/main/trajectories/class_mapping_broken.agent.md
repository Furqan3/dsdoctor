# Trajectory — `class_mapping_broken.agent`

- agent: **auditor**
- model: `qwen3.8-27b`
- 5 model call(s), 19 tool call(s)
- 21,003 prompt tokens, 2,345 completion tokens
- wall time: 66.77s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/class_mapping_broken. Decide whether it is safe to train on, and tell me what to fix first.

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/class_mapping_broken",
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
      "boxes": 4937
    },
    "val": {
      "images": 132,
      "label_files": 132,
      "boxes": 1823
    }
  },
  "total_boxes": 6760,
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

### 4. model turn  ·  5.7s  ·  230 tokens out

<details><summary>reasoning</summary>

```text
The case name is "class_mapping_broken" — a strong hint that the class mapping is broken. Let's run all the detectors. They're all accurate, and most of them are low-cost. I'll run them all.
```

</details>

**calls** `run_detector({"name": "structure_scan"})`

**calls** `run_detector({"name": "geometry_scan"})`

**calls** `run_detector({"name": "class_scan"})`

**calls** `run_detector({"name": "normalisation_scan"})`

**calls** `run_detector({"name": "duplicate_scan"})`

**calls** `run_detector({"name": "image_integrity_scan"})`

**calls** `class_distribution({})`

### 5. tool result  ·  `run_detector`

arguments: `{"name": "structure_scan"}`

```json
{
  "detector": "structure_scan",
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "structure_scan:malformed_label_row:0",
      "type": "malformed_label_row",
      "detector_severity": "critical",
      "title": "3 label row(s) are not parseable",
      "affected_files": 3,
      "example_files": [
        "train/000000085376",
        "train/000000336053",
        "train/000000468332"
      ],
      "example_evidence": [
        "train/000000085376: line 4: expected 5 fields, found 6 -> '2 0.35076000 0.51731150 0.14235400 0.03737500 0.87'",
        "train/000000336053: line 8: expected 5 fields, found 6 -> '0 0.84375000 0.49211550 0.31250000 0.76352100 0.87'"
      ]
    }
  ],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 6. tool result  ·  `run_detector`

arguments: `{"name": "geometry_scan"}`

```json
{
  "detector": "geometry_scan",
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 7. tool result  ·  `run_detector`

arguments: `{"name": "class_scan"}`

```json
{
  "detector": "class_scan",
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "class_scan:class_id_out_of_range:0",
      "type": "class_id_out_of_range",
      "detector_severity": "critical",
      "title": "4 box(es) use a class id that does not exist",
      "affected_files": 4,
      "example_files": [
        "train/000000436883",
        "train/000000492362",
        "val/000000297084"
      ],
      "example_evidence": [
        "train/000000436883 line 4: class id 13 but data.yaml defines 12 classes (0..11)",
        "train/000000492362 line 5: class id 15 but data.yaml defines 12 classes (0..11)"
      ]
    }
  ],
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

arguments: `{"name": "duplicate_scan"}`

```json
{
  "detector": "duplicate_scan",
  "findings_found": 0,
  "findings": [],
  "hint": "Use inspect_finding to see the full evidence before deciding whether a finding is real."
}
```

### 10. tool result  ·  `run_detector`

arguments: `{"name": "image_integrity_scan"}`

```json
{
  "detector": "image_integrity_scan",
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
      "person": 1980,
      "chair": 411,
      "car": 576,
      "dining table": 242,
      "cup": 355,
      "bottle": 192,
      "bowl": 250,
      "handbag": 274,
      "truck": 201,
      "bench": 185,
      "book": 73,
      "backpack": 196,
      "<id 13 not in data.yaml>": 1,
      "<id 15 not in data.yaml>": 1
    },
    "val": {
      "person": 463,
      "chair": 320,
      "car": 47,
      "dining table": 138,
      "cup": 180,
      "bottle": 263,
      "bowl": 81,
      "handbag": 25,
      "truck": 15,
      "bench": 25,
      "book": 228,
      "backpack": 36,
      "<id 13 not in data.yaml>": 1,
      "<id 15 not in data.yaml>": 1
    }
  }
}
```

### 12. model turn  ·  4.15s  ·  119 tokens out

<details><summary>reasoning</summary>

```text
Let's inspect the two findings. The case name is "class_mapping_broken" — this is a strong hint that the class ID out-of-range issue is the core problem. Let's inspect both findings.
```

</details>

**calls** `inspect_finding({"finding_id": "class_scan:class_id_out_of_range:0"})`

**calls** `inspect_finding({"finding_id": "structure_scan:malformed_label_row:0"})`

### 13. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "class_scan:class_id_out_of_range:0"}`

```json
{
  "finding_id": "class_scan:class_id_out_of_range:0",
  "type": "class_id_out_of_range",
  "meaning": "class id is >= the number of classes in data.yaml",
  "detector": "class_scan",
  "detector_severity": "critical",
  "title": "4 box(es) use a class id that does not exist",
  "explanation": "An id at or above nc indexes past the model's classification head. Ultralytics raises on this during dataset verification; a hand-rolled loader will index out of bounds mid-epoch.",
  "affected_files": [
    "train/000000436883",
    "train/000000492362",
    "val/000000297084",
    "val/000000536947"
  ],
  "affected_file_count": 4,
  "evidence": [
    "train/000000436883 line 4: class id 13 but data.yaml defines 12 classes (0..11)",
    "train/000000492362 line 5: class id 15 but data.yaml defines 12 classes (0..11)",
    "val/000000297084 line 4: class id 13 but data.yaml defines 12 classes (0
… [293 more chars]
```

### 14. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "structure_scan:malformed_label_row:0"}`

```json
{
  "finding_id": "structure_scan:malformed_label_row:0",
  "type": "malformed_label_row",
  "meaning": "label row does not have 5 whitespace-separated fields",
  "detector": "structure_scan",
  "detector_severity": "critical",
  "title": "3 label row(s) are not parseable",
  "explanation": "A YOLO row must be `class_id xc yc w h`. Rows that are not will raise during dataset scanning or be skipped silently depending on the loader version.",
  "affected_files": [
    "train/000000085376",
    "train/000000336053",
    "train/000000468332"
  ],
  "affected_file_count": 3,
  "evidence": [
    "train/000000085376: line 4: expected 5 fields, found 6 -> '2 0.35076000 0.51731150 0.14235400 0.03737500 0.87'",
    "train/000000336053: line 8: expected 5 fields, found 6 -> '0 0.84375000 0.49211550 0.31250000 0.76352100 0.87'",
    "train/000000468332: line 5: expected 5 fields, found 6 -> '0 0.867
… [219 more chars]
```

### 15. model turn  ·  10.18s  ·  415 tokens out

<details><summary>reasoning</summary>

```text
Let's check the actual label files and verify the evidence. In particular, the malformed rows (6 fields — the 6th field is 0.87, which looks like a confidence score, suggesting these labels were exported from a detection tool with a confidence column) and the out-of-range class IDs (13 and 15 — the case name is "class_mapping_broken", so it's highly likely that the class mapping is off. For example, the labels were generated with a 16-class or 17-class mapping, or the IDs were shifted).

Let's read the 4 affected files and check the rows.
```

</details>

**calls** `read_label_file({"file_key": "train/000000085376"})`

**calls** `read_label_file({"file_key": "train/000000336053"})`

**calls** `read_label_file({"file_key": "train/000000468332"})`

**calls** `read_label_file({"file_key": "train/000000436883"})`

**calls** `read_label_file({"file_key": "train/000000492362"})`

**calls** `read_label_file({"file_key": "val/000000297084"})`

**calls** `read_label_file({"file_key": "val/000000536947"})`

### 16. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000085376"}`

```json
{
  "file": "train/000000085376",
  "image_size": [
    480,
    640
  ],
  "image_error": null,
  "num_boxes": 9,
  "rows": [
    {
      "line": 1,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.450656,
      "yc": 0.560547,
      "w": 0.208938,
      "h": 0.084468
    },
    {
      "line": 2,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.241021,
      "yc": 0.544164,
      "w": 0.323042,
      "h": 0.08536
    },
    {
      "line": 3,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.0973855,
      "yc": 0.650188,
      "w": 0.182771,
      "h": 0.141562
    },
    {
      "line": 5,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.505812,
      "yc": 0.51482,
      "w": 0.152292,
      "h": 0.017422
    },
    {
      "line": 6,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.088406,
      "yc": 0.495711,
      "
… [868 more chars]
```

### 17. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000336053"}`

```json
{
  "file": "train/000000336053",
  "image_size": [
    640,
    480
  ],
  "image_error": null,
  "num_boxes": 14,
  "rows": [
    {
      "line": 1,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.746359,
      "yc": 0.106781,
      "w": 0.076156,
      "h": 0.175772
    },
    {
      "line": 2,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.6680775,
      "yc": 0.108104,
      "w": 0.065875,
      "h": 0.173416
    },
    {
      "line": 3,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.2412495,
      "yc": 0.263062,
      "w": 0.338125,
      "h": 0.265208
    },
    {
      "line": 4,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.9619375,
      "yc": 0.086708,
      "w": 0.076125,
      "h": 0.059042
    },
    {
      "line": 5,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.79618,
      "yc": 0.0
… [1713 more chars]
```

### 18. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000468332"}`

```json
{
  "file": "train/000000468332",
  "image_size": [
    640,
    424
  ],
  "image_error": null,
  "num_boxes": 12,
  "rows": [
    {
      "line": 1,
      "class_id": 3,
      "class_name": "dining table",
      "xc": 0.375359,
      "yc": 0.95217,
      "w": 0.74753,
      "h": 0.09566
    },
    {
      "line": 2,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.370047,
      "yc": 0.0895635,
      "w": 0.054282,
      "h": 0.025259
    },
    {
      "line": 3,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.919828,
      "yc": 0.3573,
      "w": 0.160344,
      "h": 0.310118
    },
    {
      "line": 4,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.490547,
      "yc": 0.642689,
      "w": 0.418344,
      "h": 0.561792
    },
    {
      "line": 6,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.758258,
      "yc": 0
… [1392 more chars]
```

### 19. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000436883"}`

```json
{
  "file": "train/000000436883",
  "image_size": [
    640,
    421
  ],
  "image_error": null,
  "num_boxes": 9,
  "rows": [
    {
      "line": 1,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.947281,
      "yc": 0.614228,
      "w": 0.105438,
      "h": 0.493538
    },
    {
      "line": 2,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.910016,
      "yc": 0.4619595,
      "w": 0.0305,
      "h": 0.128005
    },
    {
      "line": 3,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.883445,
      "yc": 0.463064,
      "w": 0.023046,
      "h": 0.13183
    },
    {
      "line": 4,
      "class_id": 13,
      "class_name": "<id 13 not in data.yaml>",
      "xc": 0.886164,
      "yc": 0.394133,
      "w": 0.015922,
      "h": 0.03715
    },
    {
      "line": 5,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.0206955,
… [756 more chars]
```

### 20. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000492362"}`

```json
{
  "file": "train/000000492362",
  "image_size": [
    427,
    640
  ],
  "image_error": null,
  "num_boxes": 7,
  "rows": [
    {
      "line": 1,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.427061,
      "yc": 0.516766,
      "w": 0.395432,
      "h": 0.707062
    },
    {
      "line": 2,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.730105,
      "yc": 0.474734,
      "w": 0.200586,
      "h": 0.431686
    },
    {
      "line": 3,
      "class_id": 11,
      "class_name": "backpack",
      "xc": 0.396276,
      "yc": 0.343117,
      "w": 0.128056,
      "h": 0.200296
    },
    {
      "line": 4,
      "class_id": 7,
      "class_name": "handbag",
      "xc": 0.79418,
      "yc": 0.411438,
      "w": 0.048736,
      "h": 0.037218
    },
    {
      "line": 5,
      "class_id": 15,
      "class_name": "<id 15 not in data.yaml>",
      "xc": 0
… [435 more chars]
```

### 21. tool result  ·  `read_label_file`

arguments: `{"file_key": "val/000000297084"}`

```json
{
  "file": "val/000000297084",
  "image_size": [
    612,
    612
  ],
  "image_error": null,
  "num_boxes": 10,
  "rows": [
    {
      "line": 1,
      "class_id": 5,
      "class_name": "bottle",
      "xc": 0.416136,
      "yc": 0.8316095,
      "w": 0.105376,
      "h": 0.313971
    },
    {
      "line": 2,
      "class_id": 5,
      "class_name": "bottle",
      "xc": 0.577418,
      "yc": 0.850539,
      "w": 0.096764,
      "h": 0.277418
    },
    {
      "line": 3,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.14598,
      "yc": 0.5492475,
      "w": 0.185294,
      "h": 0.233889
    },
    {
      "line": 4,
      "class_id": 13,
      "class_name": "<id 13 not in data.yaml>",
      "xc": 0.496634,
      "yc": 0.820972,
      "w": 0.97977,
      "h": 0.321356
    },
    {
      "line": 5,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.5584
… [917 more chars]
```

### 22. tool result  ·  `read_label_file`

arguments: `{"file_key": "val/000000536947"}`

```json
{
  "file": "val/000000536947",
  "image_size": [
    640,
    480
  ],
  "image_error": null,
  "num_boxes": 5,
  "rows": [
    {
      "line": 1,
      "class_id": 15,
      "class_name": "<id 15 not in data.yaml>",
      "xc": 0.17493,
      "yc": 0.587948,
      "w": 0.09939,
      "h": 0.209562
    },
    {
      "line": 2,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.023773,
      "yc": 0.59851,
      "w": 0.042672,
      "h": 0.139354
    },
    {
      "line": 3,
      "class_id": 8,
      "class_name": "truck",
      "xc": 0.915914,
      "yc": 0.485833,
      "w": 0.168172,
      "h": 0.126
    },
    {
      "line": 4,
      "class_id": 5,
      "class_name": "bottle",
      "xc": 0.442969,
      "yc": 0.933729,
      "w": 0.076468,
      "h": 0.041876
    },
    {
      "line": 5,
      "class_id": 5,
      "class_name": "bottle",
      "xc": 0.287367,
… [91 more chars]
```

### 23. model turn  ·  38.98s  ·  1534 tokens out

<details><summary>reasoning</summary>

```text
All the evidence is in. Let me summarize:

**Finding 1: class_scan:class_id_out_of_range:0 (critical, exact)**
- 4 boxes use class IDs 13 and 15, but data.yaml only defines 12 classes (0–11).
- 2 in train (000000436883 line 4: id 13; 000000492362 line 5: id 15), 2 in val (000000297084 line 4: id 13; 000000536947 line 1: id 15).
- This is a classic "class mapping is broken" case (the directory name is literally class_mapping_broken). The labels were probably generated with a different class mapping (e.g., a 16-class or 14-class mapping), and the IDs were never remapped to the 12 classes in data.yaml.
- Impact: Ultralytics' dataset verification will raise an exception (class ID out of range) —
… [2392 more chars]
```

</details>

**calls** `submit_report({"verdict": "blocked", "headline": "Do not train yet: 4 boxes use class ids 13/15 that don't exist in data.yaml (2 of them in val), and 3 train rows have a stray 6th field — both are trivial to fix, b
… [1898 more chars])`

### 24. tool result  ·  `submit_report`

arguments: `{"verdict": "blocked", "headline": "Do not train yet: 4 boxes use class ids 13/15 that don't exist in data.yaml (2 of them in val), and 3 train rows have a stray 6th field \u2014 both are trivial to f`

```json
{
  "status": "report recorded",
  "decisions": 2
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
