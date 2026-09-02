"""Propose a split that cannot leak.

`duplicate_scan` finds train/val leakage; until now the report handed the fix
back to the engineer, because re-splitting a dataset is a judgement call. Most
of it is not. The part that is mechanical - and the part people get wrong - is
that **deduplicating is not the fix**. Deleting the val copy of a leaked image
leaves its near-duplicates behind, and a random re-split of the survivors
re-creates the leak from a different pair.

The fix is to split *groups*, not images. Build the near-duplicate graph over
the whole dataset, take its connected components, and assign whole components
to one side or the other. Then no two images related by any chain of
near-duplication can end up on opposite sides, so the resulting split is
leak-free by construction rather than by inspection - and `dsdoctor` verifies
that afterwards by re-running the leakage detector on the proposal.

Within that constraint the assignment is greedy on the thing that actually
breaks otherwise: per-class coverage in val. A leak-free split that leaves
four classes unvalidated has traded one measurement failure for another.

Nothing here modifies the source dataset. The proposal is a JSON manifest, and
materialising it writes a new directory of symlinks.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import Dataset, split_role
from .detectors.duplicates import _candidate_pairs, _hamming, _hash_all, NEAR_DUP_DISTANCE
from .formats import link_or_copy

DEFAULT_VAL_FRACTION = 0.2


class _DisjointSet:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def duplicate_groups(ds: Dataset) -> list[list[str]]:
    """Connected components of the near-duplicate graph, over every split."""
    table = _hash_all(ds)
    items = list(table.items())
    ds_set = _DisjointSet([k for k, _ in items])

    hashes = [(k, v[1]) for k, v in items]
    for i, j in _candidate_pairs(hashes):
        ka, (_sa, ha, sha_a) = items[i]
        kb, (_sb, hb, sha_b) = items[j]
        if (sha_a and sha_a == sha_b) or _hamming(ha, hb) <= NEAR_DUP_DISTANCE:
            ds_set.union(ka, kb)

    groups: dict[str, list[str]] = defaultdict(list)
    for k, _ in items:
        groups[ds_set.find(k)].append(k)
    # Images with no readable hash are still part of the dataset and must be
    # placed somewhere; each becomes its own group.
    for s in ds.samples:
        if s.key() not in table:
            groups[s.key()].append(s.key())
    return [sorted(v) for v in groups.values()]


def _class_counts(ds: Dataset, keys: list[str]) -> Counter:
    c: Counter = Counter()
    for k in keys:
        s = ds.get(k)
        if s and s.label:
            for b in s.label.boxes:
                c[b.cls] += 1
    return c


def propose(ds: Dataset, *, val_fraction: float = DEFAULT_VAL_FRACTION,
            seed: int = 0) -> dict:
    """A leak-free, class-aware train/val assignment."""
    groups = duplicate_groups(ds)
    total = sum(len(g) for g in groups)
    target_val = int(round(total * val_fraction))

    per_group_classes = {i: _class_counts(ds, g) for i, g in enumerate(groups)}
    all_classes = set()
    for c in per_group_classes.values():
        all_classes |= set(c)

    # Deterministic ordering: biggest groups first, ties broken by content, so
    # two runs on the same dataset produce the same split without a seed
    # mattering - a split you cannot reproduce is not a split you can debug.
    order = sorted(range(len(groups)),
                   key=lambda i: (-len(groups[i]), groups[i][0]))

    val_keys: list[str] = []
    train_keys: list[str] = []
    val_classes: Counter = Counter()
    n_val = 0

    for i in order:
        g = groups[i]
        classes = per_group_classes[i]
        missing = set(classes) - set(val_classes)
        room = n_val + len(g) <= target_val
        # Take a group into val if there is room, or if it carries a class val
        # does not have yet - an unvalidatable class is worse than a slightly
        # oversized val split.
        if room or (missing and n_val < target_val * 1.5):
            val_keys += g
            val_classes.update(classes)
            n_val += len(g)
        else:
            train_keys += g

    uncovered = sorted(all_classes - set(val_classes))
    return {
        "schema": "dsdoctor/split-proposal/1",
        "dataset": str(ds.root),
        "val_fraction_requested": val_fraction,
        "val_fraction_achieved": round(len(val_keys) / max(total, 1), 4),
        "groups": len(groups),
        "largest_group": max((len(g) for g in groups), default=0),
        "classes_missing_from_val": [ds.class_name(c) for c in uncovered],
        "assignment": {"train": sorted(train_keys), "val": sorted(val_keys)},
    }


def materialise(ds: Dataset, proposal: dict, out: Path) -> dict:
    """Write the proposal as a new dataset of symlinks. Source is untouched."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if ds.yaml_path and ds.yaml_path.is_file():
        (out / "data.yaml").write_text(ds.yaml_path.read_text())

    written = {"train": 0, "val": 0}
    for split, keys in proposal["assignment"].items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for k in keys:
            s = ds.get(k)
            if s is None:
                continue
            if s.image_path and s.image_path.is_file():
                link_or_copy(s.image_path, out / "images" / split / s.image_path.name)
            if s.label_path and s.label_path.is_file():
                link_or_copy(s.label_path, out / "labels" / split / s.label_path.name)
            written[split] = written.get(split, 0) + 1
    return written


def verify(out: Path) -> dict:
    """Re-run the leakage detector on a materialised proposal.

    A guard that is never exercised is worth nothing, so the re-split checks
    its own output rather than asserting the construction is correct.
    """
    from .detectors.duplicates import duplicate_scan
    ds = Dataset(out)
    found = duplicate_scan(ds)
    leaks = [f for f in found if f.type == "train_val_leakage"]
    return {"leak_free": not leaks,
            "leaked_pairs": sum(f.n_items for f in leaks),
            "images": len(ds.samples)}


def write(ds: Dataset, proposal: dict, path: Path) -> Path:
    Path(path).write_text(json.dumps(proposal, indent=2))
    return Path(path)
