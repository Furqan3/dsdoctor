# Trajectory — `everything.baseline`

- agent: **baseline-single-prompt**
- model: `qwen3.8-27b`
- 1 model call(s), 0 tool call(s)
- 36,284 prompt tokens, 621 completion tokens
- wall time: 47.89s

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
>   "root": "/home/unk/projects/micro1-hackathon/dataset-doctor/data/cases/everything",
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
>       "images": 471,
>       "label_files": 470,
>       "boxes": 4924
>     },
>     "val": {
>       "images": 135,
>       "label_files": 135,
>       "boxes": 1852
>     }
>   },
>   "total_boxes": 6776,
>   "class_counts": {
>     "person": 2434,
>     "chair": 734,
>     "car": 635,
> … [623 more chars]

## Steps

> ⚠️ **baseline context: 52/608 label files, 25967 chars of label rows**

### 1. model turn  ·  47.89s  ·  621 tokens out

```json
{
  "verdict": "blocked",
  "headline": "Dataset is blocked due to critical YAML inconsistencies, out-of-range class IDs, and a malformed label row that will cause training crashes or invalid model outputs.",
  "findings": [
    {
      "type": "yaml_inconsistency",
      "severity": "critical",
      "files": [
        "<dataset>"
      ],
      "rationale": "data.yaml declares nc=15 but only lists 12 class names. This mismatch will cause the training script to either crash or misinterpret class indices, as the number of classes defined in the config does not match the number of names provided."
    },
    {
      "type": "class_id_out_of_range",
      "severity": "critical",
      "files": [
        "<dataset>"
      ],
      "rationale": "The class counts report instances for cl
… [1327 more chars]

## Human checkpoint

This trajectory ends at a written report and a fix plan. Nothing in the dataset has been modified. Applying the plan requires `dsdoctor apply`, which refuses to proceed without an explicit confirmation typed by a person, and which never applies steps marked `requires_human_review`.
