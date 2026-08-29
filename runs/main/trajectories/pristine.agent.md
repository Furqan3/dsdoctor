# Trajectory — `pristine.agent`

- agent: **auditor**
- model: `qwen3.8-27b`
- 3 model call(s), 10 tool call(s)
- 6,992 prompt tokens, 771 completion tokens
- wall time: 25.47s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/pristine. Decide whether it is safe to train on, and tell me what to fix first.

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/pristine",
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
      "images": 132,
      "label_files": 132,
      "boxes": 1823
    }
  },
  "total_boxes": 6763,
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

### 4. model turn  ·  5.79s  ·  234 tokens out

<details><summary>reasoning</summary>

```text
The dataset is 600 images, 12 classes, 6763 boxes. All detectors are exact. Let's run them all — they're all cheap/medium cost, and all exact. I'll run them in parallel.
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
      "person": 463,
      "chair": 321,
      "car": 47,
      "dining table": 139,
      "cup": 180,
      "bottle": 263,
      "bowl": 81,
      "handbag": 25,
      "truck": 15,
      "bench": 25,
      "book": 228,
      "backpack": 36
    }
  }
}
```

### 12. model turn  ·  11.92s  ·  490 tokens out

<details><summary>reasoning</summary>

```text
All 7 detectors (all exact) returned zero findings. The dataset is "pristine" as the case name suggests. Class distribution is reasonable — no extreme imbalance (person is dominant at 1983/463, but that's typical for COCO-style data, and class_scan didn't flag extreme_class_imbalance).

No findings to rank or suppress. Submit the report with an empty decisions array, verdict "usable_with_caveats" — no, actually, if there are no findings at all, is it "usable_with_caveats" or something better? The verdict options are: blocked, fix_before_training, usable_with_caveats. If there are no findings, the dataset is clean. "usable_with_caveats" is the best available option — the only caveat being the
… [114 more chars]
```

</details>

**calls** `submit_report({"verdict": "usable_with_caveats", "headline": "Dataset is clean: all exact checks (structure, image integrity, box geometry, normalisation, class IDs, train/val leakage) passed with zero findings — s
… [514 more chars])`

### 13. tool result  ·  `submit_report`

arguments: `{"verdict": "usable_with_caveats", "headline": "Dataset is clean: all exact checks (structure, image integrity, box geometry, normalisation, class IDs, train/val leakage) passed with zero findings \u2`

```json
{
  "status": "report recorded",
  "decisions": 0
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
