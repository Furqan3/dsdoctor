# Trajectory — `only_minor.baseline`

- agent: **baseline-single-prompt**
- model: `qwen3.8-27b`
- 1 model call(s), 0 tool call(s)
- 36,171 prompt tokens, 627 completion tokens
- wall time: 47.91s

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
>   "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/only_minor",
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
>       "label_files": 471,
>       "boxes": 4920
>     },
>     "val": {
>       "images": 132,
>       "label_files": 132,
>       "boxes": 1813
>     }
>   },
>   "total_boxes": 6733,
>   "class_counts": {
>     "person": 2439,
>     "chair": 731,
>     "car": 624,
> … [623 more chars]

## Steps

> ⚠️ **baseline context: 49/603 label files, 25967 chars of label rows**

### 1. model turn  ·  47.9s  ·  627 tokens out

{
  "verdict": "fix_before_training",
  "headline": "Dataset contains orphan label files, an empty label file, and numerous tiny boxes that require cleanup before training.",
  "findings": [
    {
      "type": "orphan_label_file",
      "severity": "major",
      "files": [
        "train/639462453486",
        "train/915238652806",
        "train/920332787315"
      ],
      "rationale": "The train split has 471 label files but only 468 images. The file listing shows 3 label files (639462453486, 915238652806, 920332787315) that do not have corresponding entries in the train image list, indicating orphaned labels."
    },
    {
      "type": "empty_label_file",
      "severity": "minor",
      "files": [
        "train/000000145597"
      ],
      "rationale": "The label file for image 00
… [862 more chars]

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
