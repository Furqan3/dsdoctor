# Trajectory — `export_bug.baseline`

- agent: **baseline-single-prompt**
- model: `qwen3.8-27b`
- 1 model call(s), 0 tool call(s)
- 36,044 prompt tokens, 2,006 completion tokens
- wall time: 79.75s

## Agent instructions

```text
You are reviewing an object-detection dataset in YOLO format to decide whether it is safe to train on. Below is a summary of the dataset, the list of files in each split, and the raw label rows for a large sample of those files.

Report every defect you can find. These are the defect types, use these exact type ids:

  out_of_bounds - box coordinates fall outside the normalised [0,1] range
  denormalised_coords - coordinates look like raw pixels, not normalised fractions
  degenerate_box - box has zero or negative width/height
  class_id_out_of_range - class id is >= the number of classes in data.yaml
  corrupt_image - image file cannot be decoded
  train_val_leakage - the same image content appears in both train and val
  malformed_label_row - label row does not have 5 whitespace-separated fields
  class_swap - a group of boxes carries a systematically wrong class id
  duplicate_annotation - the same box is listed more than once in one label file
  missing_label_file - image has no corresponding label file
  near_duplicate_image - perceptually near-identical images inflate the dataset
  tiny_box - box is small enough to be dropped or destabilise training
  extreme_class_imbalance 
... [744 more chars]
```

## Task

> ## Dataset summary
> ```json
> {
>   "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/export_bug",
>   "nc": 12,
>   "names": [
>     "person",
>     "chair",
>     "car",
>     "dining table",
>     "cup",
>     "bottle",
>     "bowl",
>     "handbag",
>     "truck",
>     "bench",
>     "book",
>     "backpack"
>   ],
>   "splits": {
>     "train": {
>       "images": 468,
>       "label_files": 468,
>       "boxes": 4938
>     },
>     "val": {
>       "images": 132,
>       "label_files": 132,
>       "boxes": 1822
>     }
>   },
>   "total_boxes": 6760,
>   "class_counts": {
>     "person": 2446,
>     "chair": 732,
>     "car": 624,
> … [623 more chars]

## Steps

> ⚠️ **baseline context: 52/600 label files, 25891 chars of label rows**

### 1. model turn  ·  79.75s  ·  2006 tokens out

```json
{
  "verdict": "blocked",
  "headline": "Dataset contains critical coordinate normalization errors, malformed label rows, and potential class imbalance issues that prevent safe training.",
  "findings": [
    {
      "type": "denormalised_coords",
      "severity": "critical",
      "files": [
        "train/000000355325",
        "train/000000511076"
      ],
      "rationale": "Label files contain raw pixel coordinates (e.g., 239.0, 478.0) instead of normalized [0,1] fractions. This will cause the model to learn incorrect spatial relationships or fail to converge."
    },
    {
      "type": "malformed_label_row",
      "severity": "critical",
      "files": [
        "val/000000172595"
      ],
      "rationale": "The last row in val/000000172595 has 6 fields ('5 0.94861700 0.32
… [4179 more chars]

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
