# Trajectory — `subtle_leak.baseline`

- agent: **baseline-single-prompt**
- model: `qwen3.8-27b`
- 1 model call(s), 0 tool call(s)
- 36,172 prompt tokens, 1,262 completion tokens
- wall time: 62.68s

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
>   "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/subtle_leak",
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
>       "boxes": 4940
>     },
>     "val": {
>       "images": 135,
>       "label_files": 135,
>       "boxes": 1871
>     }
>   },
>   "total_boxes": 6811,
>   "class_counts": {
>     "person": 2468,
>     "chair": 732,
>     "car": 642,
> … [623 more chars]

## Steps

> ⚠️ **baseline context: 49/603 label files, 25956 chars of label rows**

### 1. model turn  ·  62.68s  ·  1262 tokens out

{
  "verdict": "fix_before_training",
  "headline": "Dataset contains critical train/val leakage via duplicate image content and near-duplicates, along with minor coordinate anomalies.",
  "findings": [
    {
      "type": "train_val_leakage",
      "severity": "critical",
      "files": [
        "train/000000363840",
        "val/000000363840"
      ],
      "rationale": "The file ID '000000363840' appears in both the train and val splits. In YOLO datasets, identical filenames across splits typically indicate the same image content is used for both training and validation, leading to data leakage and inflated performance metrics."
    },
    {
      "type": "near_duplicate_image",
      "severity": "major",
      "files": [
        "train/000000363840",
        "val/000000363840"
      ]
… [2163 more chars]

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
