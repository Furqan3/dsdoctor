# Trajectory — `export_bug.agent`

- agent: **auditor**
- model: `qwen3.8-27b`
- 7 model call(s), 18 tool call(s)
- 44,288 prompt tokens, 7,976 completion tokens
- wall time: 201.27s

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

> Audit the dataset at /home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/export_bug. Decide whether it is safe to train on, and tell me what to fix first.

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
  "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/export_bug",
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
      "boxes": 4938
    },
    "val": {
      "images": 132,
      "label_files": 132,
      "boxes": 1822
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

### 4. model turn  ·  5.59s  ·  226 tokens out

<details><summary>reasoning</summary>

```text
All detectors are accurate. Run them all. The case name is "export_bug" — probably a normalization issue (coordinates not normalized, e.g., pixel coordinates from an export). Let's run all the detectors.
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
      "finding_id": "structure_scan:malformed_label_row:0",
      "type": "malformed_label_row",
      "detector_severity": "critical",
      "title": "3 label row(s) are not parseable",
      "affected_files": 3,
      "example_files": [
        "train/000000242946",
        "train/000000356531",
        "val/000000172595"
      ],
      "example_evidence": [
        "train/000000242946: line 17: expected 5 fields, found 6 -> '6 0.43035900 0.54203100 0.07406200 0.05743800 0.87'",
        "train/000000356531: line 1: expected 5 fields, found 6 -> '5 0.67551600 0.52602100 0.02968800 0.15954200 0.87'"
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
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "geometry_scan:out_of_bounds:0",
      "type": "out_of_bounds",
      "detector_severity": "critical",
      "title": "51 box(es) fall outside the normalised [0,1] range",
      "affected_files": 4,
      "example_files": [
        "train/000000001000",
        "train/000000067616",
        "train/000000355325"
      ],
      "example_evidence": [
        "train/000000001000 line 1: xyxy=(20.5402,226.7098,73.8298,348.2400) -> '7 47.1850 287.4749 53.2896 121.5302'",
        "train/000000001000 line 2: xyxy=(196.9107,224.3500,267.1705,341.9001) -> '7 232.0406 283.1251 70.2598 117.5501'"
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
  "findings_found": 1,
  "findings": [
    {
      "finding_id": "normalisation_scan:denormalised_coords:0",
      "type": "denormalised_coords",
      "detector_severity": "critical",
      "title": "4 file(s) hold pixel coordinates instead of normalised fractions",
      "affected_files": 4,
      "example_files": [
        "train/000000001000",
        "train/000000067616",
        "train/000000355325"
      ],
      "example_evidence": [
        "train/000000001000: 16/16 rows exceed 1.0; max x=572.335 y=344.265 vs image 640x480 -> '7 47.1850 287.4749 53.2896 121.5302'",
        "train/000000067616: 25/25 rows exceed 1.0; max x=626.535 y=416.46 vs image 640x480 -> '2 464.2598 392.2102 145.9008 104.5598'"
      ]
    }
  ],
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
      "bottle": 191,
      "bowl": 250,
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
      "bottle": 262,
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

### 12. model turn  ·  11.78s  ·  454 tokens out

<details><summary>reasoning</summary>

```text
Let's check the findings. There are 3 findings:

1. `structure_scan:malformed_label_row:0` — 3 label rows are unparseable (6 fields instead of 5). Critical.
2. `geometry_scan:out_of_bounds:0` — 51 boxes outside the [0,1] range. Critical.
3. `normalisation_scan:denormalised_coords:0` — 4 files contain pixel coordinates. Critical.

The out_of_bounds and denormalised_coords findings seem to be related — the same 4 files (train/000000001000, train/000000067616, train/000000355325, and one more). The out-of-bounds boxes are probably caused by the denormalised coordinates. Let's inspect each finding to check the evidence.

Also, note the class distribution: val has 228 books, but train only has 73
… [515 more chars]
```

</details>

**calls** `inspect_finding({"finding_id": "structure_scan:malformed_label_row:0"})`

**calls** `inspect_finding({"finding_id": "geometry_scan:out_of_bounds:0"})`

**calls** `inspect_finding({"finding_id": "normalisation_scan:denormalised_coords:0"})`

### 13. tool result  ·  `inspect_finding`

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
    "train/000000242946",
    "train/000000356531",
    "val/000000172595"
  ],
  "affected_file_count": 3,
  "evidence": [
    "train/000000242946: line 17: expected 5 fields, found 6 -> '6 0.43035900 0.54203100 0.07406200 0.05743800 0.87'",
    "train/000000356531: line 1: expected 5 fields, found 6 -> '5 0.67551600 0.52602100 0.02968800 0.15954200 0.87'",
    "val/000000172595: line 5: expected 5 fields, found 6 -> '5 0.948617
… [214 more chars]
```

### 14. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "geometry_scan:out_of_bounds:0"}`

```json
{
  "finding_id": "geometry_scan:out_of_bounds:0",
  "type": "out_of_bounds",
  "meaning": "box coordinates fall outside the normalised [0,1] range",
  "detector": "geometry_scan",
  "detector_severity": "critical",
  "title": "51 box(es) fall outside the normalised [0,1] range",
  "explanation": "Coordinates outside [0,1] are clipped or rejected depending on the loader. Where they are clipped the box silently changes shape, so the model is trained against a target the annotator never drew.",
  "affected_files": [
    "train/000000001000",
    "train/000000067616",
    "train/000000355325",
    "train/000000511076"
  ],
  "affected_file_count": 4,
  "evidence": [
    "train/000000001000 line 1: xyxy=(20.5402,226.7098,73.8298,348.2400) -> '7 47.1850 287.4749 53.2896 121.5302'",
    "train/000000001000 line 2: xyxy=(196.9107,224.3500,267.1705,341.9001) -> '7 232.0406 283.1251 70.2598 117.5
… [1399 more chars]
```

### 15. tool result  ·  `inspect_finding`

arguments: `{"finding_id": "normalisation_scan:denormalised_coords:0"}`

```json
{
  "finding_id": "normalisation_scan:denormalised_coords:0",
  "type": "denormalised_coords",
  "meaning": "coordinates look like raw pixels, not normalised fractions",
  "detector": "normalisation_scan",
  "detector_severity": "critical",
  "title": "4 file(s) hold pixel coordinates instead of normalised fractions",
  "explanation": "The values are bounded by the image dimensions rather than by 1.0, so this is an export that skipped the divide-by-width/height step. Training on it produces boxes that collapse to the top-left corner.",
  "affected_files": [
    "train/000000001000",
    "train/000000067616",
    "train/000000355325",
    "train/000000511076"
  ],
  "affected_file_count": 4,
  "evidence": [
    "train/000000001000: 16/16 rows exceed 1.0; max x=572.335 y=344.265 vs image 640x480 -> '7 47.1850 287.4749 53.2896 121.5302'",
    "train/000000067616: 25/25 rows exceed 1.0; max
… [539 more chars]
```

### 16. model turn  ·  7.12s  ·  199 tokens out

<details><summary>reasoning</summary>

```text
Let me check the label files to verify the evidence. In particular, I'll check the malformed rows (6 fields — the 6th field looks like a confidence score of 0.87, which suggests an export bug where the confidence was appended) and the denormalized files.

Let me read the 3 malformed files and 1 denormalized file.
```

</details>

**calls** `read_label_file({"file_key": "train/000000242946"})`

**calls** `read_label_file({"file_key": "val/000000172595"})`

**calls** `read_label_file({"file_key": "train/000000001000"})`

### 17. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000242946"}`

```json
{
  "file": "train/000000242946",
  "image_size": [
    640,
    480
  ],
  "image_error": null,
  "num_boxes": 28,
  "rows": [
    {
      "line": 1,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.643906,
      "yc": 0.16274,
      "w": 0.019374,
      "h": 0.052104
    },
    {
      "line": 2,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.790063,
      "yc": 0.132177,
      "w": 0.021312,
      "h": 0.020438
    },
    {
      "line": 3,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.010961,
      "yc": 0.326958,
      "w": 0.015172,
      "h": 0.2
    },
    {
      "line": 4,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.4424135,
      "yc": 0.182021,
      "w": 0.237641,
      "h": 0.364042
    },
    {
      "line": 5,
      "class_id": 0,
      "class_name": "person",
      "xc": 0.817625,
      "yc": 0.249521
… [4014 more chars]
```

### 18. tool result  ·  `read_label_file`

arguments: `{"file_key": "val/000000172595"}`

```json
{
  "file": "val/000000172595",
  "image_size": [
    640,
    360
  ],
  "image_error": null,
  "num_boxes": 6,
  "rows": [
    {
      "line": 1,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.1994215,
      "yc": 0.547417,
      "w": 0.333375,
      "h": 0.905166
    },
    {
      "line": 2,
      "class_id": 1,
      "class_name": "chair",
      "xc": 0.034078,
      "yc": 0.0690835,
      "w": 0.068156,
      "h": 0.134611
    },
    {
      "line": 3,
      "class_id": 10,
      "class_name": "book",
      "xc": 0.901656,
      "yc": 0.367667,
      "w": 0.088656,
      "h": 0.082222
    },
    {
      "line": 4,
      "class_id": 7,
      "class_name": "handbag",
      "xc": 0.424672,
      "yc": 0.657528,
      "w": 0.165,
      "h": 0.281056
    },
    {
      "line": 6,
      "class_id": 5,
      "class_name": "bottle",
      "xc": 0.333422,
      "yc": 0.0465
… [389 more chars]
```

### 19. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000001000"}`

```json
{
  "file": "train/000000001000",
  "image_size": [
    640,
    480
  ],
  "image_error": null,
  "num_boxes": 16,
  "rows": [
    {
      "line": 1,
      "class_id": 7,
      "class_name": "handbag",
      "xc": 47.185,
      "yc": 287.4749,
      "w": 53.2896,
      "h": 121.5302
    },
    {
      "line": 2,
      "class_id": 7,
      "class_name": "handbag",
      "xc": 232.0406,
      "yc": 283.1251,
      "w": 70.2598,
      "h": 117.5501
    },
    {
      "line": 3,
      "class_id": 0,
      "class_name": "person",
      "xc": 156.775,
      "yc": 266.3352,
      "w": 83.2294,
      "h": 228.4099
    },
    {
      "line": 4,
      "class_id": 0,
      "class_name": "person",
      "xc": 424.4947,
      "yc": 143.1802,
      "w": 37.1302,
      "h": 45.5203
    },
    {
      "line": 5,
      "class_id": 0,
      "class_name": "person",
      "xc": 309.7904,
      "yc": 253.80
… [1892 more chars]
```

### 20. model turn  ·  94.07s  ·  4096 tokens out

<details><summary>reasoning</summary>

```text
All the evidence is in. Let me organize:

**Findings (all from exact detectors, none suppressible):**

1. `structure_scan:malformed_label_row:0` — 3 rows with 6 fields instead of 5. The 6th field is `0.87` — looks like a confidence score that got appended during export (e.g., from a COCO/labelme export that includes a score). Files: train/000000242946 (line 17), train/000000356531 (line 1), val/000000172595 (line 5). One of them is in **val** — this is important. Depending on the loader, this will either crash the dataset scan or silently drop the row. The val row means the val set is corrupted: the bottle box on line 5 of val/000000172595 is lost (or the loader crashes). This is a real defe
… [14713 more chars]
```

</details>

> ⚠️ **step 4: turn hit the output token limit while reasoning; retrying with thinking disabled and a forced tool choice**

### 21. model turn  ·  10.38s  ·  108 tokens out

**calls** `read_label_file({"file_key": "train/000000355325"})`

**calls** `read_label_file({"file_key": "train/000000511076"})`

### 22. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000355325"}`

```json
{
  "file": "train/000000355325",
  "image_size": [
    478,
    640
  ],
  "image_error": null,
  "num_boxes": 5,
  "rows": [
    {
      "line": 1,
      "class_id": 3,
      "class_name": "dining table",
      "xc": 239.0,
      "yc": 299.0998,
      "w": 478.0,
      "h": 286.8397
    },
    {
      "line": 2,
      "class_id": 4,
      "class_name": "cup",
      "xc": 376.805,
      "yc": 186.9648,
      "w": 166.8296,
      "h": 178.3302
    },
    {
      "line": 3,
      "class_id": 6,
      "class_name": "bowl",
      "xc": 110.9603,
      "yc": 174.0998,
      "w": 150.3401,
      "h": 104.5197
    },
    {
      "line": 4,
      "class_id": 3,
      "class_name": "dining table",
      "xc": 239.0,
      "yc": 479.6397,
      "w": 478.0,
      "h": 303.4598
    },
    {
      "line": 5,
      "class_id": 0,
      "class_name": "person",
      "xc": 415.2099,
      "yc": 73.1395
… [76 more chars]
```

### 23. tool result  ·  `read_label_file`

arguments: `{"file_key": "train/000000511076"}`

```json
{
  "file": "train/000000511076",
  "image_size": [
    640,
    427
  ],
  "image_error": null,
  "num_boxes": 5,
  "rows": [
    {
      "line": 1,
      "class_id": 2,
      "class_name": "car",
      "xc": 200.6598,
      "yc": 133.8051,
      "w": 41.3606,
      "h": 35.33
    },
    {
      "line": 2,
      "class_id": 0,
      "class_name": "person",
      "xc": 447.5402,
      "yc": 224.5751,
      "w": 190.0998,
      "h": 303.7695
    },
    {
      "line": 3,
      "class_id": 9,
      "class_name": "bench",
      "xc": 544.0653,
      "yc": 333.99,
      "w": 184.2291,
      "h": 163.12
    },
    {
      "line": 4,
      "class_id": 0,
      "class_name": "person",
      "xc": 144.9747,
      "yc": 138.5598,
      "w": 10.7494,
      "h": 32.5203
    },
    {
      "line": 5,
      "class_id": 7,
      "class_name": "handbag",
      "xc": 523.415,
      "yc": 242.9397,
… [66 more chars]
```

### 24. model turn  ·  64.43s  ·  2846 tokens out

<details><summary>reasoning</summary>

```text
All confirmed. Let me summarize:

1. **normalisation_scan:denormalised_coords:0** — 4 files (all in train) contain raw pixel coordinates. 100% of rows in each file exceed 1.0, and the max values match the image dimensions (640x480, 640x480, 478x640, 640x427). This is a classic export bug: the divide-by-width/height step was skipped. This is the root cause of the out_of_bounds finding.

2. **geometry_scan:out_of_bounds:0** — 51 boxes out of range, all in the same 4 files. This is a *consequence* of the denormalization, not an independent defect. Both are exact detectors, so neither can be suppressed. But I should rank the denormalization as the root cause and note that the out_of_bounds is a
… [5477 more chars]
```

</details>

**calls** `submit_report({"verdict": "blocked", "headline": "Don't train yet: 3 label rows carry a stray 6th field (one in val) that will crash dataset verification, and 4 train files were exported in pixel coordinates — fix
… [2748 more chars])`

### 25. tool result  ·  `submit_report`

arguments: `{"verdict": "blocked", "headline": "Don't train yet: 3 label rows carry a stray 6th field (one in val) that will crash dataset verification, and 4 train files were exported in pixel coordinates \u2014`

```json
{
  "status": "report recorded",
  "decisions": 3
}
```

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
