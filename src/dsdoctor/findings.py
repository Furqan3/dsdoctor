"""The finding vocabulary shared by the detectors, the agent and the scorer.

Keeping the defect-type ids in one place is what makes the evaluation
objective: the injector labels ground truth with these ids, the detectors
report with these ids, and the scorer matches on ``(type, file)`` pairs
instead of on prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

CRITICAL = "critical"   # training will crash or silently learn the wrong thing
MAJOR = "major"         # measurably hurts mAP or makes validation misleading
MINOR = "minor"         # worth knowing, safe to defer

SEVERITY_ORDER = {CRITICAL: 0, MAJOR: 1, MINOR: 2}


# type id -> (default severity, one-line meaning)
DEFECT_TYPES: dict[str, tuple[str, str]] = {
    "out_of_bounds":        (CRITICAL, "box coordinates fall outside the normalised [0,1] range"),
    "denormalised_coords":  (CRITICAL, "coordinates look like raw pixels, not normalised fractions"),
    "degenerate_box":       (CRITICAL, "box has zero or negative width/height"),
    "class_id_out_of_range":(CRITICAL, "class id is >= the number of classes in data.yaml"),
    "corrupt_image":        (CRITICAL, "image file cannot be decoded"),
    "train_val_leakage":    (CRITICAL, "the same image content appears in both train and val"),
    "malformed_label_row":  (CRITICAL, "label row does not have 5 whitespace-separated fields"),
    "class_swap":           (MAJOR,    "a group of boxes carries a systematically wrong class id"),
    "duplicate_annotation": (MAJOR,    "the same box is listed more than once in one label file"),
    "missing_label_file":   (MAJOR,    "image has no corresponding label file"),
    "near_duplicate_image": (MAJOR,    "perceptually near-identical images inflate the dataset"),
    "tiny_box":             (MAJOR,    "box is small enough to be dropped or destabilise training"),
    "extreme_class_imbalance": (MAJOR, "a class has too few instances to learn or to validate"),
    "empty_label_file":     (MINOR,    "label file exists but contains no boxes"),
    "orphan_label_file":    (MINOR,    "label file has no corresponding image"),
    "yaml_inconsistency":   (MAJOR,    "data.yaml disagrees with the labels on disk"),

    # --- polygon and keypoint geometry ----------------------------------
    # These can only fire on a segmentation or pose dataset. On a detection
    # dataset they are structurally inert, which is why they sit in `core`
    # rather than an opt-in group: they cannot alter the published numbers,
    # and `test_measured_configuration.py` demonstrates that rather than
    # assuming it.
    "polygon_too_few_points": (CRITICAL, "a segmentation polygon has fewer than three points, so it encloses nothing"),
    "polygon_zero_area":    (CRITICAL, "a segmentation polygon has zero area; its points are collinear or coincident"),
    "polygon_self_intersecting": (CRITICAL, "a segmentation polygon crosses itself, so its interior is undefined"),
    "polygon_unverified":   (MINOR,    "a polygon has too many vertices to check for self-intersection"),
    "keypoint_visibility_invalid": (MAJOR, "a keypoint visibility flag is not 0, 1 or 2"),
    "keypoint_outside_box": (MINOR,   "a visible keypoint lies outside the box it belongs to"),

    # --- trainability against a specific training configuration ---------
    # These need a parameter the dataset does not carry (the input resolution
    # you intend to train at), so they live in an opt-in group and take it
    # from --imgsz.
    "undetectable_at_imgsz": (MAJOR, "boxes are smaller than the model's finest feature stride at the chosen input size"),
    "over_max_detections":  (MAJOR,  "images carry more objects than the default max_det, so evaluation silently truncates them"),

    # --- annotation provenance smells -----------------------------------
    "template_annotation":  (MAJOR,  "an identical box is repeated verbatim across many different images"),
    "whole_frame_box":      (MINOR,  "a box spans essentially the entire image"),

    # --- split integrity (group: "split") -------------------------------
    "class_absent_from_val": (MAJOR,   "class has training instances but none in val, so its AP is undefined"),
    "split_ratio_extreme":  (MAJOR,    "the train/val split is far outside a usable range"),

    # --- capture metadata (group: "metadata") ---------------------------
    "exif_orientation":     (CRITICAL, "image carries a non-trivial EXIF orientation tag, so labels and pixels may disagree"),

    # --- governance and privacy (group: "privacy") ----------------------
    # These do not affect whether the model trains. They affect whether the
    # dataset may lawfully be trained on or published at all, which is a
    # question no amount of mAP answers - hence a separate category.
    "gps_metadata":         (MAJOR,    "images carry EXIF GPS coordinates that disclose where they were taken"),
    "missing_license":      (MAJOR,    "no licence or attribution file accompanies the dataset"),
    "representation_skew":  (MINOR,    "a class is concentrated in one capture slice, so the split may not measure generalisation"),
}

# Findings answer one of two different questions, and conflating them makes
# the verdict incoherent: "will this train" and "may I train on this at all"
# have different audiences and different remedies. The trainability verdict is
# computed from TRAINABILITY findings only; governance findings are reported
# alongside it and never silently change it.
TRAINABILITY = "trainability"
GOVERNANCE = "governance"

GOVERNANCE_TYPES = {"gps_metadata", "missing_license", "representation_skew"}


def category_for(defect_type: str) -> str:
    return GOVERNANCE if defect_type in GOVERNANCE_TYPES else TRAINABILITY


@dataclass
class Finding:
    """One defect claim, always carrying the evidence that supports it."""

    type: str
    title: str
    detail: str
    detector: str
    items: list[str] = field(default_factory=list)      # sample keys "split/stem"
    evidence: list[str] = field(default_factory=list)   # raw rows / numbers
    severity: str = ""          # filled from DEFECT_TYPES when not given
    fix: dict | None = None
    verified: bool | None = None
    verifier_note: str = ""
    category: str = ""          # trainability | governance; derived from type
    # "split/stem" -> the label rows this finding is actually about. Optional:
    # many findings are about a file as a whole. Where it is known, the visual
    # report can outline the offending box instead of every box in the image,
    # which is the difference between evidence and decoration.
    locations: dict[str, list[int]] | None = None

    def __post_init__(self) -> None:
        # A detector may override the severity, but when it does not, the
        # shared table decides - so the same defect never changes rank
        # between runs or between the code paths that construct it.
        if not self.severity:
            self.severity = DEFECT_TYPES.get(self.type, (MAJOR, ""))[0]
        if not self.category:
            self.category = category_for(self.type)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        head = f"[{self.severity}] {self.type}: {self.title}"
        if self.items:
            shown = ", ".join(self.items[:4])
            more = f" (+{self.n_items - 4} more)" if self.n_items > 4 else ""
            head += f"\n    files: {shown}{more}"
        return head


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings,
                  key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), -f.n_items, f.type))
