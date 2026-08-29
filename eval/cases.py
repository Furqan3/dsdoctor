"""The evaluation cases.

Twelve datasets derived from the same clean corpus, each with a known set of
injected defects. They are meant to cover the realistic ways a detection
dataset goes wrong, not to be uniformly hard:

  * `pristine` injects nothing at all and exists to measure what each arm
    invents; `only_minor` injects nothing severe and measures whether an arm
    can say so instead of over-warning;
  * `subtle_leak` is the challenging case - a dataset that looks entirely
    healthy except for three leaked images, which is the defect with the worst
    consequence and the fewest symptoms;
  * `everything` is the stress case, with every injectable defect type at once.

`class_swap` is deliberately absent. It was injected in earlier revisions and
removed once measurement showed no arm could detect it at usable precision -
see README.md, Improvement changelog, iteration 4. A defect that nothing can find
measures nothing except the size of the corpus.

Every case uses its own seed, so which files get corrupted differs between
them and no arm can succeed by memorising file names.
"""

CASES: list[dict] = [
    {
        "name": "pristine",
        "seed": 101,
        "why": "No injected defects at all. Measures what each arm invents.",
        "recipe": {},
    },
    {
        "name": "subtle_leak",
        "seed": 102,
        "why": "CHALLENGE CASE. Three leaked train images in val and nothing "
               "else. Every other signal looks healthy, so an arm that only "
               "inspects label text sees a clean dataset.",
        "recipe": {"train_val_leakage": 3},
    },
    {
        "name": "export_bug",
        "seed": 103,
        "why": "An export that skipped normalisation and left a confidence "
               "column on some rows.",
        "recipe": {"denormalised_coords": 4, "malformed_label_row": 3},
    },
    {
        "name": "geometry_mess",
        "seed": 104,
        "why": "A bad augmentation pass wrote back boxes that leave the frame, "
               "collapse to zero, or shrink below the dataloader's threshold.",
        "recipe": {"out_of_bounds": 5, "degenerate_box": 3, "tiny_box": 4},
    },
    {
        "name": "structure_rot",
        "seed": 105,
        "why": "Files moved and renamed by hand: labels without images, images "
               "without labels, files emptied by a failed re-export.",
        "recipe": {"missing_label_file": 5, "empty_label_file": 3,
                   "orphan_label_file": 3},
    },
    {
        "name": "corrupt_media",
        "seed": 106,
        "why": "An interrupted download left truncated JPEGs behind.",
        "recipe": {"corrupt_image": 4, "missing_label_file": 2},
    },
    {
        "name": "class_mapping_broken",
        "seed": 107,
        "why": "A wrong class-id map in the export script: ids past the end of "
               "data.yaml, and rows carrying a stray confidence column.",
        "recipe": {"class_id_out_of_range": 4, "malformed_label_row": 3},
    },
    {
        "name": "thin_classes",
        "seed": 108,
        "why": "A class stripped almost to nothing, and a data.yaml whose nc no "
               "longer matches its names.",
        "recipe": {"extreme_class_imbalance": 1, "yaml_inconsistency": 1},
    },
    {
        "name": "duplicate_farm",
        "seed": 109,
        "why": "Re-encoded copies of training images plus rows pasted twice "
               "inside single label files.",
        "recipe": {"near_duplicate_image": 4, "duplicate_annotation": 4},
    },
    {
        "name": "only_minor",
        "seed": 110,
        "why": "CALIBRATION CASE. Nothing here blocks training: some deliberate "
               "background images and a few stale label files. An arm that "
               "calls this 'blocked' is crying wolf, which is how a reviewer "
               "learns to ignore the tool.",
        "recipe": {"empty_label_file": 4, "orphan_label_file": 3},
    },
    {
        "name": "leak_and_dupes",
        "seed": 111,
        "why": "The defects that quietly inflate a validation score together: "
               "leakage into val, re-encoded copies inside train, and rows "
               "counted twice.",
        "recipe": {"train_val_leakage": 4, "near_duplicate_image": 2,
                   "duplicate_annotation": 3},
    },
    {
        "name": "everything",
        "seed": 112,
        "why": "STRESS CASE. Every defect type at once - tests whether triage "
               "still surfaces the run-blocking issues first.",
        "recipe": {"out_of_bounds": 3, "degenerate_box": 2, "tiny_box": 3,
                   "duplicate_annotation": 3, "denormalised_coords": 2,
                   "class_id_out_of_range": 2, "missing_label_file": 3,
                   "empty_label_file": 2, "orphan_label_file": 2,
                   "malformed_label_row": 2, "corrupt_image": 2,
                   "yaml_inconsistency": 1, "train_val_leakage": 3,
                   "near_duplicate_image": 3},
    },
]


def by_name(name: str) -> dict:
    for c in CASES:
        if c["name"] == name:
            return c
    raise KeyError(f"unknown case {name!r}; have {[c['name'] for c in CASES]}")
