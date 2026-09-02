"""Governance and privacy: may this dataset be trained on, and published?

Every other detector in this package answers "will this train correctly". These
answer a question that no amount of mAP will ever raise, and that tends to be
asked for the first time by a lawyer, after the model has shipped:

  - Do these images disclose where they were taken?
  - Is there any statement of what may lawfully be done with them?
  - Is a class present only in one narrow slice of capture conditions, so that
    a good validation score describes that slice rather than the world?

Findings from this module are categorised as ``governance`` rather than
``trainability`` and are deliberately kept out of the trainability verdict.
"Do not train on this yet" and "do not publish this without checking the
licence" are different sentences to different people, and merging them makes
both easier to ignore.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from PIL import Image, UnidentifiedImageError

from ..dataset import Dataset
from ..findings import Finding, MAJOR, MINOR
from . import register

GPS_IFD_TAG = 34853          # EXIF GPSInfo pointer
GPS_LATITUDE = 2
GPS_LONGITUDE = 4

LICENCE_FILENAMES = (
    "license", "licence", "license.txt", "licence.txt", "license.md",
    "licence.md", "copying", "notice", "notice.txt", "attribution",
    "attribution.txt", "attribution.md", "readme.md", "readme.txt",
    "datasheet.md", "provenance.md", "sources.csv", "sources.txt",
)

# Representation skew: only worth a sentence when a class is both common
# enough to matter and concentrated far above the dataset's own base rate.
SKEW_MIN_INSTANCES = 20
SKEW_CLASS_SHARE = 0.90
SKEW_BASE_RATE_HEADROOM = 0.60


def _has_gps(path) -> bool:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return False
            if GPS_IFD_TAG not in exif:
                return False
            try:
                gps = exif.get_ifd(GPS_IFD_TAG)
            except (AttributeError, OSError, ValueError):
                # An older Pillow, or a malformed IFD. The pointer alone is
                # not proof of coordinates, so do not claim one.
                return False
            # The pointer can exist with an empty payload; only actual
            # latitude/longitude values disclose a location.
            return bool(gps) and (GPS_LATITUDE in gps or GPS_LONGITUDE in gps)
    except (UnidentifiedImageError, OSError, ValueError, AttributeError):
        return False


@register("privacy_scan",
          "Look for capture metadata and missing paperwork that create legal "
          "or privacy exposure rather than training failures: EXIF GPS "
          "coordinates and absent licence/attribution files. Detects "
          "gps_metadata and missing_license. Reads image headers.",
          reads_pixels=True,
          covers=("gps_metadata", "missing_license"), group="privacy")
def privacy_scan(ds: Dataset) -> list[Finding]:
    out: list[Finding] = []

    located = [s.key() for s in ds.samples
               if s.image_path is not None and _has_gps(s.image_path)]
    if located:
        out.append(Finding(
            type="gps_metadata", severity=MAJOR,
            title=f"{len(located)} image(s) carry EXIF GPS coordinates",
            detail="These files record where the photograph was taken, to a "
                   "precision of a few metres. That travels with the dataset: "
                   "into every copy, every re-share and every public release, "
                   "long after anyone remembers it is there. If the images "
                   "show homes, vehicles, workplaces or people, the "
                   "coordinates turn a picture into a location record about "
                   "identifiable individuals, which is personal data under "
                   "GDPR and comparable regimes. Training does not read EXIF, "
                   "so stripping it costs nothing in model quality.",
            detector="privacy_scan", items=sorted(located),
            evidence=[f"{k}: EXIF GPSInfo present with coordinates"
                      for k in sorted(located)[:10]],
            fix={"action": "strip_exif_metadata", "targets": sorted(located)}))

    present = {p.name.lower() for p in ds.root.iterdir() if p.is_file()}
    if not (present & set(LICENCE_FILENAMES)):
        out.append(Finding(
            type="missing_license", severity=MAJOR,
            title="no licence or attribution file accompanies this dataset",
            detail="Nothing at the dataset root states what may be done with "
                   "these images. For an inherited or vendor-supplied dataset "
                   "that is the default state, not a reassuring one: absent "
                   "an express grant, the rights sit with whoever took the "
                   "photographs, and 'we found no restriction' is not a "
                   "finding of permission. Establish the provenance and terms "
                   "before a model trained on this reaches production, while "
                   "the person who can answer the question is still reachable.",
            detector="privacy_scan", items=[],
            evidence=[f"searched {ds.root} for: "
                      + ", ".join(LICENCE_FILENAMES[:8]) + ", ...",
                      f"files at root: "
                      + (", ".join(sorted(present)[:10]) or "(none)")],
            fix={"action": "document_provenance", "targets": []}))

    return out


@register("representation_scan",
          "Measure whether each class is spread across capture conditions or "
          "concentrated in one slice, using image resolution as a proxy for "
          "capture source. Detects representation_skew. Reads image headers.",
          reads_pixels=True, covers=("representation_skew",), group="privacy")
def representation_scan(ds: Dataset) -> list[Finding]:
    # Resolution is a coarse proxy for "which camera/pipeline produced this",
    # and it is the only one available without reading pixels. It is a proxy,
    # and the finding says so: this reports a distribution, and asks a human
    # whether it is the distribution they intended.
    per_class: dict[int, Counter] = defaultdict(Counter)
    overall: Counter = Counter()

    for s in ds.samples:
        if not s.label or s.image_path is None:
            continue
        ds.ensure_image_meta(s)
        if not s.width or not s.height:
            continue
        bucket = f"{s.width}x{s.height}"
        overall[bucket] += 1
        for b in s.label.boxes:
            per_class[b.cls][bucket] += 1

    n_images = sum(overall.values())
    if n_images < SKEW_MIN_INSTANCES or len(overall) < 2:
        return []   # one resolution everywhere: nothing to compare against

    skewed: list[str] = []
    for cls, counts in sorted(per_class.items()):
        total = sum(counts.values())
        if total < SKEW_MIN_INSTANCES:
            continue
        bucket, n = counts.most_common(1)[0]
        share = n / total
        base = overall[bucket] / n_images
        if share >= SKEW_CLASS_SHARE and base <= SKEW_BASE_RATE_HEADROOM:
            skewed.append(
                f"'{ds.class_name(cls)}' (id {cls}): {share:.0%} of "
                f"{total} instances come from {bucket} images, which are only "
                f"{base:.0%} of the dataset")

    if not skewed:
        return []

    return [Finding(
        type="representation_skew", severity=MINOR,
        title=f"{len(skewed)} class(es) are concentrated in one capture slice",
        detail="Each of these classes was almost entirely photographed under "
               "one set of conditions. A model can then score well on it by "
               "learning the conditions rather than the object, and a random "
               "train/val split cannot detect that, because both sides inherit "
               "the same concentration. This is measured on image resolution "
               "as a stand-in for capture source, so treat it as a question "
               "rather than a defect: if that concentration is a property of "
               "the collection rather than of the world you will deploy into, "
               "the validation score is measuring the collection.",
        detector="representation_scan", items=[],
        evidence=skewed[:15],
        fix={"action": "review_capture_diversity", "targets": []})]
