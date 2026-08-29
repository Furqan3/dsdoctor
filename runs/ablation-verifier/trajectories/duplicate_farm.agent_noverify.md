# Trajectory — `duplicate_farm.agent_noverify`

- agent: **auditor**
- model: `qwen3.8-27b`
- 5 model call(s), 14 tool call(s)
- 18,078 prompt tokens, 3,906 completion tokens
- wall time: 95.1s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/duplicate_farm. Decide whether it is safe to train on, and tell me what to fix first.

## Steps

### 1. model turn  ·  1.21s  ·  47 tokens out

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/duplicate_farm",
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
      "images": 472,
      "label_files": 472,
      "boxes": 4977
    },
    "val": {
      "images": 132,
      "label_files": 132,
      "boxes": 1823
    }
  },
  "total_boxes": 6800,
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

### 4. model turn  ·  5.39s  ·  216 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate. Run them all. The case name is "duplicate_farm" — probably a train/val leak. Let's run all the detectors.
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
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "geometry_scan:duplicate_annotation:0",
      "type": "duplicate_annotation",
      "detector_severity": "major",
      "title": "4 box(es) are listed more than once in the same file",
      "affected_files": 4,
      "example_files": [
        "train/000000016451",
        "train/000000136715",
        "train/000000230993"
      ],
      "example_evidence": [
        "train/000000016451 line 9 repeats line 8: '11 0.87822700 0.78724750 0.13799000 0.10070300'",
        "train/000000136715 line 16 repeats line 6: '0 0.62028100 0.19757600 0.04471800 0.04908200'"
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
      "finding_id": "duplicate_scan:near_duplicate_image:0",
      "type": "near_duplicate_image",
      "detector_severity": "major",
      "title": "4 near-duplicate image pair(s) inside a split",
      "affected_files": 8,
      "example_files": [
        "train/000000064084",
        "train/000000077396",
        "train/000000227399"
      ],
      "example_evidence": [
        "train/000000064084 ~ train/404490299022 (perceptual distance 0/64)",
        "train/000000077396 ~ train/553349435394 (perceptual distance 0/64)"
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
      "person": 2001,
      "chair": 413,
      "car": 578,
      "dining table": 246,
      "cup": 356,
      "bottle": 195,
      "bowl": 252,
      "handbag": 278,
      "truck": 201,
      "bench": 185,
      "book": 74,
      "backpack": 198
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

### 12. model turn  ·  4.17s  ·  178 tokens out

<details><summary>reasoning</summary>

```text
Let's check the two findings. Both are from exact detectors, so they can't be suppressed, but I should inspect the evidence to rank them properly.

1. `geometry_scan:duplicate_annotation:0` — 4 boxes are listed multiple times within the same file, 4 files affected, all within train.
2. `duplicate_scan:near_duplicate_image:0` — 4 pairs of near-duplicate images within a split (train), 8 files affected.

Let's inspect both.
```

</details>

**calls** `inspect_finding({"finding_id": "geometry_scan:duplicate_annotation:0"})`

**calls** `inspect_finding({"finding_id": "duplicate_scan:near_duplicate_image:0"})`

### 13. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "geometry_scan:duplicate_annotation:0"}`

```json
{
  "finding_id": "geometry_scan:duplicate_annotation:0",
  "type": "duplicate_annotation",
  "meaning": "the same box is listed more than once in one label file",
  "detector": "geometry_scan",
  "detector_severity": "major",
  "title": "4 box(es) are listed more than once in the same file",
  "explanation": "Duplicated boxes double-count in the loss and break the one-target-per-object assumption used by NMS-free matchers.",
  "affected_files": [
    "train/000000016451",
    "train/000000136715",
    "train/000000230993",
    "train/000000313588"
  ],
  "affected_file_count": 4,
  "evidence": [
    "train/000000016451 line 9 repeats line 8: '11 0.87822700 0.78724750 0.13799000 0.10070300'",
    "train/000000136715 line 16 repeats line 6: '0 0.62028100 0.19757600 0.04471800 0.04908200'",
    "train/000000230993 line 6 repeats line 1: '0 0.68262500 0.71503750 0.21059400 0.51909300'",
… [293 more chars]
```

### 14. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "duplicate_scan:near_duplicate_image:0"}`

```json
{
  "finding_id": "duplicate_scan:near_duplicate_image:0",
  "type": "near_duplicate_image",
  "meaning": "perceptually near-identical images inflate the dataset",
  "detector": "duplicate_scan",
  "detector_severity": "major",
  "title": "4 near-duplicate image pair(s) inside a split",
  "explanation": "Duplicates inside train over-weight whatever they contain and inflate the dataset size the team reports. Duplicates inside val make one scene count several times toward the score.",
  "affected_files": [
    "train/000000064084",
    "train/000000077396",
    "train/000000227399",
    "train/000000435208",
    "train/354608865506",
    "train/404490299022",
    "train/553349435394",
    "train/931528508977"
  ],
  "affected_file_count": 8,
  "evidence": [
    "train/000000064084 ~ train/404490299022 (perceptual distance 0/64)",
    "train/000000077396 ~ train/553349435394 (perceptual dis
… [473 more chars]
```

### 15. model turn  ·  5.87s  ·  215 tokens out

<details><summary>reasoning</summary>

```text
Let me check the evidence. Both findings are from exact detectors, so they can't be suppressed. Let me verify by cross-checking against the actual label files.

1. geometry_scan:duplicate_annotation:0 — 4 boxes are duplicated within the same file, all in train. Let me read one of the label files to confirm.

2. duplicate_scan:near_duplicate_image:0 — 4 pairs of near-duplicate images within train, perceptual distance 0/64 (i.e., identical perceptual hash). All in train, no train/val leakage.

Let me read the label files to verify the duplicate annotations.
```

</details>

**calls** `read_label_file({"file_key": "train/000000016451"})`

**calls** `read_label_file({"file_key": "train/000000136715"})`

### 16. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000016451"}`

```json
{
  "file": "train/000000016451",
  "image_size": [
    612,
    612
  ],
  "image_error": null,
  "num_boxes": 9,
  "rows": [
    {
      "line": 1,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.313954,
      "yc": 0.354918,
      "w": 0.056176,
      "h": 0.078366
    },
    {
      "line": 2,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.857565,
      "yc": 0.275711,
      "w": 0.02585,
      "h": 0.03701
    },
    {
      "line": 3,
      "class_id": 7,
      "class_name": "handbag",
      "xc": 0.875866,
      "yc": 0.6892235,
      "w": 0.109216,
      "h": 0.101683
    },
    {
      "line": 4,
      "class_id": 7,
      "class_name": "handbag",
      "xc": 0.9371,
      "yc": 0.687492,
      "w": 0.1258,
      "h": 0.142142
    },
    {
      "line": 5,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.414436,
      "yc": 0.3700
… [749 more chars]
```

### 17. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000136715"}`

```json
{
  "file": "train/000000136715",
  "image_size": [
    640,
    425
  ],
  "image_error": null,
  "num_boxes": 16,
  "rows": [
    {
      "line": 1,
      "class_id": 2,
      "class_name": "car",
      "xc": 0.896563,
      "yc": 0.424247,
      "w": 0.2065,
      "h": 0.633106
    },
    {
      "line": 2,
      "class_id": 8,
      "class_name": "truck",
      "xc": 0.896195,
      "yc": 0.435894,
      "w": 0.202516,
      "h": 0.65353
    },
    {
      "line": 3,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.4081405,
      "yc": 0.5280935,
      "w": 0.392469,
      "h": 0.826965
    },
    {
      "line": 4,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.067906,
      "yc": 0.5775175,
      "w": 0.132812,
      "h": 0.822471
    },
    {
      "line": 5,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.2667805,
      "yc": 0.2
… [1909 more chars]
```

### 18. model turn  ·  73.09s  ·  3250 tokens out

<details><summary>reasoning</summary>

```text
All 7 detectors have been run (all exact). Findings:

1. `geometry_scan:duplicate_annotation:0` — 4 boxes are duplicated within the same file, all in train. Verified by cross-checking against the raw label files: line 9 of train/000000016451 is a byte-for-byte identical repeat of line 8, and line 16 of train/000000136715 is a byte-for-byte identical repeat of line 6. Exact detector, evidence confirmed. Severity: major (per the detector). This is a real defect — duplicated boxes double-count in the loss. But it's only 4 boxes out of 6800, all in train, not val. So it's a real defect but small in scale.

2. `duplicate_scan:near_duplicate_image:0` — 4 pairs of near-duplicate images within train
… [9914 more chars]
```

</details>

**calls** `submit_report({"verdict": "fix_before_training", "headline": "Dataset is structurally clean and will train, but dedupe 4 label files with repeated boxes and 4 pairs of identical train images first; val is clean so
… [1511 more chars])`

### 19. tool result  ·  `submit_report`

arguments: `{"verdict": "fix_before_training", "headline": "Dataset is structurally clean and will train, but dedupe 4 label files with repeated boxes and 4 pairs of identical train images first; val is clean so `

```json
{
  "status": "report recorded",
  "decisions": 2
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
