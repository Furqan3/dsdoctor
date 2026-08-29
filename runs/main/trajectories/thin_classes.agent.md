# Trajectory — `thin_classes.agent`

- agent: **auditor**
- model: `qwen3.8-27b`
- 4 model call(s), 12 tool call(s)
- 10,725 prompt tokens, 1,999 completion tokens
- wall time: 52.98s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/thin_classes. Decide whether it is safe to train on, and tell me what to fix first.

## Steps

### 1. model turn  ·  2.36s  ·  47 tokens out

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/thin_classes",
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
      "boxes": 4691
    },
    "val": {
      "images": 132,
      "label_files": 132,
      "boxes": 1742
    }
  },
  "total_boxes": 6433,
  "data_yaml_problem": "data.yaml declares nc=15 but lists 12 names"
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
All detectors are accurate. Run them all. The dataset is small (600 images, 6433 boxes). Let's run all the detectors.
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
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "structure_scan:yaml_inconsistency:0",
      "type": "yaml_inconsistency",
      "detector_severity": "major",
      "title": "data.yaml is missing or inconsistent",
      "affected_files": 0,
      "example_files": [],
      "example_evidence": [
        "data.yaml declares nc=15 but lists 12 names"
      ]
    }
  ],
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
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "class_scan:extreme_class_imbalance:0",
      "type": "extreme_class_imbalance",
      "detector_severity": "major",
      "title": "1 class(es) have too few instances to train or validate",
      "affected_files": 0,
      "example_files": [],
      "example_evidence": [
        "'bowl' (id 6): only 2 instances in train; 992x rarer than the most common class; only 0 instances in val, so its per-class mAP is not meaningful"
      ]
    }
  ],
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
      "bowl": 2,
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
      "bowl": 0,
      "handbag": 25,
      "truck": 15,
      "bench": 25,
      "book": 228,
      "backpack": 36
    }
  }
}
```

### 12. model turn  ·  7.62s  ·  289 tokens out

<details><summary>reasoning</summary>

```text
Let's check the two findings. Both are from exact detectors, so they can't be suppressed. Let's inspect the evidence for both.

1. yaml_inconsistency: data.yaml declares nc=15 but only lists 12 names. This is a real problem — if nc=15 but there are only 12 names, the model will be trained with 15 classes, and classes 12–14 will be empty. In practice, this could cause problems: the model will allocate 15 output channels, and the 3 extra classes will never be learned. It won't crash, but it's a real defect. Severity: major (the detector says major).

2. extreme_class_imbalance: 'bowl' has only 2 instances in train, and 0 in val. This is a real problem — the per-class mAP for bowl in val is mea
… [104 more chars]
```

</details>

**calls** `inspect_finding({"finding_id": "structure_scan:yaml_inconsistency:0"})`

**calls** `inspect_finding({"finding_id": "class_scan:extreme_class_imbalance:0"})`

### 13. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "structure_scan:yaml_inconsistency:0"}`

```json
{
  "finding_id": "structure_scan:yaml_inconsistency:0",
  "type": "yaml_inconsistency",
  "meaning": "data.yaml disagrees with the labels on disk",
  "detector": "structure_scan",
  "detector_severity": "major",
  "title": "data.yaml is missing or inconsistent",
  "explanation": "data.yaml declares nc=15 but lists 12 names",
  "affected_files": [],
  "affected_file_count": 0,
  "evidence": [
    "data.yaml declares nc=15 but lists 12 names"
  ],
  "proposed_fix": {
    "action": "fix_data_yaml",
    "targets": []
  }
}
```

### 14. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "class_scan:extreme_class_imbalance:0"}`

```json
{
  "finding_id": "class_scan:extreme_class_imbalance:0",
  "type": "extreme_class_imbalance",
  "meaning": "a class has too few instances to learn or to validate",
  "detector": "class_scan",
  "detector_severity": "major",
  "title": "1 class(es) have too few instances to train or validate",
  "explanation": "Under-represented classes do not simply score badly, they make the headline mAP misleading: a class with two validation instances moves the average by whole points on a single detection. Decide up front whether to merge, drop or collect more of these.",
  "affected_files": [],
  "affected_file_count": 0,
  "evidence": [
    "'bowl' (id 6): only 2 instances in train; 992x rarer than the most common class; only 0 instances in val, so its per-class mAP is not meaningful"
  ],
  "proposed_fix": {
    "action": "rebalance_or_merge_classes",
    "targets": [],
    "class_ids": [
      6
… [12 more chars]
```

### 15. model turn  ·  32.14s  ·  1447 tokens out

<details><summary>reasoning</summary>

```text
Let me evaluate the two findings:

1. **yaml_inconsistency**: data.yaml declares nc=15 but only lists 12 names. This is a real defect (exact detector, cannot be suppressed). Impact: if the training framework reads nc=15, the model will allocate 15 output channels, but only 12 classes are actually labeled. The extra 3 channels will never be trained, and the mAP calculation will include 3 empty classes, diluting the headline mAP. In some frameworks, a mismatch between nc and the number of names can cause a crash or a misalignment of class names. This is a real defect that should be fixed before training — it's a one-line fix (set nc=12). Severity: major (not critical — it won't necessarily cra
… [2882 more chars]
```

</details>

**calls** `submit_report({"verdict": "fix_before_training", "headline": "Fix the data.yaml class-count mismatch (nc=15 vs 12 names) before training, and decide what to do with 'bowl' (2 train instances, 0 in val); everything
… [1532 more chars])`

### 16. tool result  ·  `submit_report`

arguments: `{"verdict": "fix_before_training", "headline": "Fix the data.yaml class-count mismatch (nc=15 vs 12 names) before training, and decide what to do with 'bowl' (2 train instances, 0 in val); everything `

```json
{
  "status": "report recorded",
  "decisions": 2
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
