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
}


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

    def __post_init__(self) -> None:
        # A detector may override the severity, but when it does not, the
        # shared table decides - so the same defect never changes rank
        # between runs or between the code paths that construct it.
        if not self.severity:
            self.severity = DEFECT_TYPES.get(self.type, (MAJOR, ""))[0]

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
