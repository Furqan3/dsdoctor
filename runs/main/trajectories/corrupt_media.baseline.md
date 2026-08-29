# Trajectory — `corrupt_media.baseline`

- agent: **baseline-single-prompt**
- model: `qwen3.8-27b`
- 1 model call(s), 0 tool call(s)
- 36,141 prompt tokens, 965 completion tokens
- wall time: 55.7s

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
>   "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/corrupt_media",
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
>       "label_files": 466,
>       "boxes": 4914
>     },
>     "val": {
>       "images": 132,
>       "label_files": 132,
>       "boxes": 1823
>     }
>   },
>   "total_boxes": 6737,
>   "class_counts": {
>     "person": 2433,
>     "chair": 726,
>     "car": 622
> … [623 more chars]

## Steps

> ⚠️ **baseline context: 53/600 label files, 25993 chars of label rows**

### 1. model turn  ·  55.7s  ·  965 tokens out

{
  "verdict": "fix_before_training",
  "headline": "Dataset contains 2 missing label files and numerous out-of-bounds boxes that extend beyond the image boundaries.",
  "findings": [
    {
      "type": "missing_label_file",
      "severity": "major",
      "files": [
        "train/000000001000",
        "train/000000001268"
      ],
      "rationale": "The summary indicates 468 train images but only 466 label files. The file listing confirms that '000000001000' and '000000001268' are present in the train image list but do not have corresponding entries in the raw label rows sample (and are the first two in the list, suggesting they are the missing ones). Training on images without labels can cause errors or undefined behavior depending on the framework."
    },
    {
      "type": "out_
… [1525 more chars]

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
