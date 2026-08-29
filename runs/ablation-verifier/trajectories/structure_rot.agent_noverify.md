# Trajectory — `structure_rot.agent_noverify`

- agent: **auditor**
- model: `qwen3.8-27b`
- 4 model call(s), 13 tool call(s)
- 11,765 prompt tokens, 2,419 completion tokens
- wall time: 60.67s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/structure_rot. Decide whether it is safe to train on, and tell me what to fix first.

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/structure_rot",
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
      "boxes": 4889
    },
    "val": {
      "images": 132,
      "label_files": 130,
      "boxes": 1807
    }
  },
  "total_boxes": 6696,
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

### 4. model turn  ·  6.08s  ·  248 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate. Run them all. The dataset has 12 classes, 468 training images, 132 validation images (but only 130 label files — 2 missing label files in validation!). 6696 boxes in total.

Let's run all the detectors.
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
  "findings_found": 3,
  "findings": [
    {
      "finding_id": "structure_scan:missing_label_file:0",
      "type": "missing_label_file",
      "detector_severity": "major",
      "title": "5 image(s) have no label file",
      "affected_files": 5,
      "example_files": [
        "train/000000006040",
        "train/000000226147",
        "train/000000322610"
      ],
      "example_evidence": [
        "no labels/train/000000006040.txt",
        "no labels/train/000000226147.txt"
      ]
    },
    {
      "finding_id": "structure_scan:orphan_label_file:1",
      "type": "orphan_label_file",
      "detector_severity": "minor",
      "title": "3 label file(s) have no image",
      "affected_files": 3,
      "example_files": [
        "train/398764888884",
        "train/531232685887",
        "train/654553082395"
      ],
      "example_evidence": [
… [702 more chars]
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
      "person": 1963,
      "chair": 407,
      "car": 572,
      "dining table": 240,
      "cup": 354,
      "bottle": 192,
      "bowl": 250,
      "handbag": 264,
      "truck": 198,
      "bench": 184,
      "book": 70,
      "backpack": 195
    },
    "val": {
      "person": 454,
      "chair": 319,
      "car": 47,
      "dining table": 139,
      "cup": 180,
      "bottle": 261,
      "bowl": 81,
      "handbag": 25,
      "truck": 15,
      "bench": 23,
      "book": 227,
      "backpack": 36
    }
  }
}
```

### 12. model turn  ·  20.97s  ·  944 tokens out

<details><summary>reasoning</summary>

```text
Let me analyze the results.

Dataset: 12 classes (COCO-style), 468 training images, 132 validation images, 4889 training boxes, 1807 validation boxes.

Findings:
1. `structure_scan:missing_label_file:0` — 5 images in the training set have no label file (major). Note: the summary says train has 468 images and 468 label files, val has 132 images and 130 label files. Hmm, interesting — the summary says train has 468 label files, but the detector says 5 training images have no label file. Wait, the summary says train: images 468, label_files 468. But the detector found 5 missing label files in train. Hmm, maybe the summary is counting label files in a different way, or maybe the 5 missing ones a
… [2436 more chars]
```

</details>

**calls** `inspect_finding({"finding_id": "structure_scan:missing_label_file:0"})`

**calls** `inspect_finding({"finding_id": "structure_scan:orphan_label_file:1"})`

**calls** `inspect_finding({"finding_id": "structure_scan:empty_label_file:2"})`

### 13. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "structure_scan:missing_label_file:0"}`

```json
{
  "finding_id": "structure_scan:missing_label_file:0",
  "type": "missing_label_file",
  "meaning": "image has no corresponding label file",
  "detector": "structure_scan",
  "detector_severity": "major",
  "title": "5 image(s) have no label file",
  "explanation": "Ultralytics treats an image with no .txt as a pure background image. If these are actually unlabelled foreground images the model is being explicitly taught that the objects in them are background, which suppresses recall for those classes.",
  "affected_files": [
    "train/000000006040",
    "train/000000226147",
    "train/000000322610",
    "val/000000223182",
    "val/000000549220"
  ],
  "affected_file_count": 5,
  "evidence": [
    "no labels/train/000000006040.txt",
    "no labels/train/000000226147.txt",
    "no labels/train/000000322610.txt",
    "no labels/val/000000223182.txt",
    "no labels/val/000000549220.tx
… [233 more chars]
```

### 14. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "structure_scan:orphan_label_file:1"}`

```json
{
  "finding_id": "structure_scan:orphan_label_file:1",
  "type": "orphan_label_file",
  "meaning": "label file has no corresponding image",
  "detector": "structure_scan",
  "detector_severity": "minor",
  "title": "3 label file(s) have no image",
  "explanation": "These labels are never read during training. Usually the remains of a deleted or renamed image.",
  "affected_files": [
    "train/398764888884",
    "train/531232685887",
    "train/654553082395"
  ],
  "affected_file_count": 3,
  "evidence": [
    "labels present, image absent: train/398764888884",
    "labels present, image absent: train/531232685887",
    "labels present, image absent: train/654553082395"
  ],
  "proposed_fix": {
    "action": "delete_orphan_labels",
    "targets": [
      "train/398764888884",
      "train/531232685887",
      "train/654553082395"
    ]
  }
}
```

### 15. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "structure_scan:empty_label_file:2"}`

```json
{
  "finding_id": "structure_scan:empty_label_file:2",
  "type": "empty_label_file",
  "meaning": "label file exists but contains no boxes",
  "detector": "structure_scan",
  "detector_severity": "minor",
  "title": "3 label file(s) contain no boxes",
  "explanation": "Intentional background images are a legitimate technique, but an unintended empty file is a silently dropped annotation. Confirm these are deliberate.",
  "affected_files": [
    "train/000000211120",
    "train/000000242934",
    "train/000000368335"
  ],
  "affected_file_count": 3,
  "evidence": [
    "0 rows in train/000000211120.txt",
    "0 rows in train/000000242934.txt",
    "0 rows in train/000000368335.txt"
  ],
  "proposed_fix": {
    "action": "review_background_images",
    "targets": [
      "train/000000211120",
      "train/000000242934",
      "train/000000368335"
    ]
  }
}
```

### 16. model turn  ·  27.0s  ·  1180 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate (no experimental ones). Findings:

1. missing_label_file: 5 images (3 train, 2 val) have no label file. Major. The 2 val images are important — val images without labels are treated as background. If they actually contain objects, the val metrics get corrupted (the model is evaluated on images where objects are treated as background → mAP is unfairly penalized, or the val set is inconsistent). This is the most urgent issue.

2. orphan_label_file: 3 label files in train with no image. Minor — never read, harmless.

3. empty_label_file: 3 label files in train with no boxes. Minor — could be intentional background images.

Everything else is clean: no corrupted images
… [717 more chars]
```

</details>

**calls** `submit_report({"verdict": "fix_before_training", "headline": "Trainable, but 5 images lack label files — 2 of them in val, which will corrupt your validation mAP; create the missing .txt files (empty if truly backg
… [2155 more chars])`

### 17. tool result  ·  `submit_report`

arguments: `{"verdict": "fix_before_training", "headline": "Trainable, but 5 images lack label files \u2014 2 of them in val, which will corrupt your validation mAP; create the missing .txt files (empty if truly `

```json
{
  "status": "report recorded",
  "decisions": 3
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
