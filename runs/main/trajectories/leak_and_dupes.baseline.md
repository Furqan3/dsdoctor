# Trajectory — `leak_and_dupes.baseline`

- agent: **baseline-single-prompt**
- model: `qwen3.8-27b`
- 1 model call(s), 0 tool call(s)
- 36,223 prompt tokens, 299 completion tokens
- wall time: 40.33s

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
>   "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/leak_and_dupes",
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
>       "images": 470,
>       "label_files": 470,
>       "boxes": 4966
>     },
>     "val": {
>       "images": 136,
>       "label_files": 136,
>       "boxes": 1849
>     }
>   },
>   "total_boxes": 6815,
>   "class_counts": {
>     "person": 2453,
>     "chair": 736,
>     "car": 62
> … [623 more chars]

## Steps

> ⚠️ **baseline context: 52/606 label files, 25972 chars of label rows**

### 1. model turn  ·  40.32s  ·  299 tokens out

{
  "verdict": "fix_before_training",
  "headline": "Dataset contains critical train/val leakage and near-duplicate images that invalidate evaluation metrics.",
  "findings": [
    {
      "type": "train_val_leakage",
      "severity": "critical",
      "files": [
        "train/000000303566",
        "val/000000303566"
      ],
      "rationale": "The image ID 000000303566 appears in both the train and val file lists. This means the model is being evaluated on data it has already seen during training, leading to artificially high performance metrics that do not reflect real-world generalization."
    },
    {
      "type": "near_duplicate_image",
      "severity": "major",
      "files": [
        "train/000000303566",
        "val/000000303566"
      ],
      "rationale": "The presence o
… [213 more chars]

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
