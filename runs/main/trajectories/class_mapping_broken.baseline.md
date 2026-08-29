# Trajectory — `class_mapping_broken.baseline`

- agent: **baseline-single-prompt**
- model: `qwen3.8-27b`
- 1 model call(s), 0 tool call(s)
- 36,159 prompt tokens, 266 completion tokens
- wall time: 39.45s

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
>   "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/class_mapping_broken",
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
>       "boxes": 4937
>     },
>     "val": {
>       "images": 132,
>       "label_files": 132,
>       "boxes": 1823
>     }
>   },
>   "total_boxes": 6760,
>   "class_counts": {
>     "person": 2443,
>     "chair": 731,
>     "ca
> … [623 more chars]

## Steps

> ⚠️ **baseline context: 52/600 label files, 25972 chars of label rows**

### 1. model turn  ·  39.45s  ·  266 tokens out

{
  "verdict": "fix_before_training",
  "headline": "The dataset contains class IDs (13 and 15) that are not defined in the data.yaml file, which will cause training to fail or produce incorrect results.",
  "findings": [
    {
      "type": "class_id_out_of_range",
      "severity": "critical",
      "files": [
        "<dataset>"
      ],
      "rationale": "The dataset summary reports 2 instances of class ID 13 and 2 instances of class ID 15. The data.yaml defines only 12 classes (IDs 0-11). These out-of-range IDs will cause errors in most YOLO training frameworks or be silently ignored, leading to data loss."
    },
    {
      "type": "yaml_inconsistency",
      "severity": "major",
      "files": [
        "<dataset>"
      ],
      "rationale": "The class counts in the summary inclu
… [162 more chars]

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
