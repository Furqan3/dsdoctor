# Trajectory — `leak_and_dupes.agent`

- agent: **auditor**
- model: `qwen3.8-27b`
- 4 model call(s), 13 tool call(s)
- 12,602 prompt tokens, 1,698 completion tokens
- wall time: 46.79s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/leak_and_dupes. Decide whether it is safe to train on, and tell me what to fix first.

## Steps

### 1. model turn  ·  1.2s  ·  47 tokens out

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/leak_and_dupes",
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
      "images": 470,
      "label_files": 470,
      "boxes": 4966
    },
    "val": {
      "images": 136,
      "label_files": 136,
      "boxes": 1849
    }
  },
  "total_boxes": 6815,
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

### 4. model turn  ·  5.38s  ·  216 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate. Run them all. The case name is "leak_and_dupes", so duplicate_scan is important. Let's run all the detectors.
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
      "title": "3 box(es) are listed more than once in the same file",
      "affected_files": 3,
      "example_files": [
        "train/000000231822",
        "train/000000255917",
        "train/000000462756"
      ],
      "example_evidence": [
        "train/000000231822 line 6 repeats line 4: '4 0.45267000 0.19101100 0.13306000 0.36856000'",
        "train/000000255917 line 15 repeats line 7: '2 0.30855500 0.77034000 0.06717200 0.03032800'"
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
  "findings_found": 2,
  "findings": [
    {
      "finding_id": "duplicate_scan:train_val_leakage:0",
      "type": "train_val_leakage",
      "detector_severity": "critical",
      "title": "4 image pair(s) leak between train and val",
      "affected_files": 8,
      "example_files": [
        "train/000000056127",
        "train/000000080932",
        "train/000000166166"
      ],
      "example_evidence": [
        "val/214827540762 is byte-identical to train/000000476787 (sha1 37fead158d85)",
        "val/382780712741 is byte-identical to train/000000056127 (sha1 9d075c4e25e8)"
      ]
    },
    {
      "finding_id": "duplicate_scan:near_duplicate_image:1",
      "type": "near_duplicate_image",
      "detector_severity": "major",
      "title": "2 near-duplicate image pair(s) inside a split",
      "affected_files": 4,
      "example_files": [
… [398 more chars]
```

### 11. tool result  ·  `class_distribution`

```json
{
  "instances_per_class_per_split": {
    "train": {
      "person": 1988,
      "chair": 411,
      "car": 581,
      "dining table": 243,
      "cup": 363,
      "bottle": 198,
      "bowl": 251,
      "handbag": 274,
      "truck": 203,
      "bench": 185,
      "book": 73,
      "backpack": 196
    },
    "val": {
      "person": 465,
      "chair": 325,
      "car": 47,
      "dining table": 141,
      "cup": 187,
      "bottle": 269,
      "bowl": 82,
      "handbag": 25,
      "truck": 15,
      "bench": 27,
      "book": 230,
      "backpack": 36
    }
  }
}
```

### 12. model turn  ·  6.52s  ·  222 tokens out

<details><summary>reasoning</summary>

```text
Let's check the findings. There are 3 findings:

1. `geometry_scan:duplicate_annotation:0` — 3 boxes are duplicated within the same file (train). Major.
2. `duplicate_scan:train_val_leakage:0` — 4 image pairs are leaking between train and val (byte-identical). Critical.
3. `duplicate_scan:near_duplicate_image:1` — 2 near-duplicate pairs within a split. Major.

Let's inspect each finding to check the evidence.
```

</details>

**calls** `inspect_finding({"finding_id": "duplicate_scan:train_val_leakage:0"})`

**calls** `inspect_finding({"finding_id": "duplicate_scan:near_duplicate_image:1"})`

**calls** `inspect_finding({"finding_id": "geometry_scan:duplicate_annotation:0"})`

### 13. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "duplicate_scan:train_val_leakage:0"}`

```json
{
  "finding_id": "duplicate_scan:train_val_leakage:0",
  "type": "train_val_leakage",
  "meaning": "the same image content appears in both train and val",
  "detector": "duplicate_scan",
  "detector_severity": "critical",
  "title": "4 image pair(s) leak between train and val",
  "explanation": "The validation split contains content the model trained on. Every metric computed against it is optimistic by an unknown margin, so it cannot be used to compare checkpoints or to decide the model is ready to ship. Re-split before doing anything else - fixing this after training means retraining.",
  "affected_files": [
    "train/000000056127",
    "train/000000080932",
    "train/000000166166",
    "train/000000476787",
    "val/214827540762",
    "val/382780712741",
    "val/438274936455",
    "val/889941193761"
  ],
  "affected_file_count": 8,
  "evidence": [
    "val/214827540762 is byte-ide
… [498 more chars]
```

### 14. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "duplicate_scan:near_duplicate_image:1"}`

```json
{
  "finding_id": "duplicate_scan:near_duplicate_image:1",
  "type": "near_duplicate_image",
  "meaning": "perceptually near-identical images inflate the dataset",
  "detector": "duplicate_scan",
  "detector_severity": "major",
  "title": "2 near-duplicate image pair(s) inside a split",
  "explanation": "Duplicates inside train over-weight whatever they contain and inflate the dataset size the team reports. Duplicates inside val make one scene count several times toward the score.",
  "affected_files": [
    "train/000000150638",
    "train/000000295420",
    "train/303522373741",
    "train/912024475486"
  ],
  "affected_file_count": 4,
  "evidence": [
    "train/000000150638 ~ train/912024475486 (perceptual distance 1/64)",
    "train/000000295420 ~ train/303522373741 (perceptual distance 0/64)"
  ],
  "proposed_fix": {
    "action": "deduplicate_images",
    "targets": [
      "train/
… [109 more chars]
```

### 15. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "geometry_scan:duplicate_annotation:0"}`

```json
{
  "finding_id": "geometry_scan:duplicate_annotation:0",
  "type": "duplicate_annotation",
  "meaning": "the same box is listed more than once in one label file",
  "detector": "geometry_scan",
  "detector_severity": "major",
  "title": "3 box(es) are listed more than once in the same file",
  "explanation": "Duplicated boxes double-count in the loss and break the one-target-per-object assumption used by NMS-free matchers.",
  "affected_files": [
    "train/000000231822",
    "train/000000255917",
    "train/000000462756"
  ],
  "affected_file_count": 3,
  "evidence": [
    "train/000000231822 line 6 repeats line 4: '4 0.45267000 0.19101100 0.13306000 0.36856000'",
    "train/000000255917 line 15 repeats line 7: '2 0.30855500 0.77034000 0.06717200 0.03032800'",
    "train/000000462756 line 3 repeats line 2: '2 0.08343750 0.46062200 0.04906300 0.04899000'"
  ],
  "proposed_fix": {
    "a
… [141 more chars]
```

### 16. model turn  ·  28.29s  ·  1213 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate (no experimental ones). Findings:

1. train_val_leakage: 4 pairs of byte-identical images between train and val — critical. This is the headline.
2. near_duplicate_image: 2 pairs within train — major.
3. duplicate_annotation: 3 boxes duplicated within 3 train label files — major.

Everything else is clean: no structural issues, no corrupted images, no geometry issues (out-of-bounds, degenerate, tiny), no normalization issues, no class issues.

Class distribution: val has some sparse classes (truck 15, handbag 25, backpack 36), but class_scan didn't flag extreme_class_imbalance, so it's within tolerance.

Verdict: critical finding reported → "blocked".

Order of dec
… [1043 more chars]
```

</details>

**calls** `submit_report({"verdict": "blocked", "headline": "4 val images are byte-identical to train images, so the validation metric is untrustworthy — re-split before training, or you will retrain after the fact.", "traini
… [2150 more chars])`

### 17. tool result  ·  `submit_report`

arguments: `{"verdict": "blocked", "headline": "4 val images are byte-identical to train images, so the validation metric is untrustworthy \u2014 re-split before training, or you will retrain after the fact.", "t`

```json
{
  "status": "report recorded",
  "decisions": 3
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
