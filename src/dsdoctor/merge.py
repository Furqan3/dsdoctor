"""Merge datasets, and report what the merge is about to get wrong.

The README's opening names three ways an engineer inherits a dataset, and one
of them is "a merge of three internal collections". Merging is where the class
mapping defects this tool detects downstream are actually *created*: two
collections each label `0` as the thing they care most about, and concatenating
their label files teaches the model that a forklift and a pedestrian are the
same object. Nothing about the result looks wrong afterwards - `class_scan`
sees ids inside range, `structure_scan` sees matched files - which is exactly
why it is worth catching at the moment it happens.

So this does the merge by *name*, never by id, and reports every place where
the sources disagree about what a name means or what a thing is called.

Three conflicts matter, in descending order of how quietly they corrupt:

  id_collision      the same class id means different things in two sources.
                    Handled automatically, by rebuilding ids from names - but
                    reported, because it tells you the sources were never
                    designed to be combined.
  name_variant      two names that differ only in case, spacing or plurality.
                    `car` and `Car` are almost certainly one class, and this
                    is the one case the tool refuses to decide: merging them
                    silently would invent a relabelling nobody approved, and
                    keeping them apart splits a class in two. Reported, and
                    left as distinct classes until a human says otherwise.
  cross_source_duplicate
                    the same image content present in more than one source.
                    After a merge these become duplicates, and if the sources
                    are split differently they become train/val leakage.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from .dataset import Dataset, split_role
from .formats import link_or_copy


def normalise_name(name: str) -> str:
    """Fold the differences that are almost never meaningful.

    Case, separators and a crude plural. Separators are removed rather than
    collapsed to a space, because the common real disagreement between two
    teams is `fork lift` against `forklift` rather than anything subtler.

    This only ever *reports* a suspected pair - nothing is merged on the
    strength of it - so a heuristic that over-groups slightly is the right
    trade. Under-grouping means the conflict is never raised at all.
    """
    s = re.sub(r"[\s_\-]+", "", str(name).strip().lower())
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]          # enough to notice car/cars
    return s


def taxonomy(datasets: list[Dataset]) -> dict:
    """Build the merged class list and describe every disagreement."""
    names: list[str] = []
    seen: set[str] = set()
    for ds in datasets:
        for n in ds.names:
            if n not in seen:
                seen.add(n)
                names.append(n)
    names.sort()
    index = {n: i for i, n in enumerate(names)}

    # Same id, different meaning.
    by_id: dict[int, set[str]] = defaultdict(set)
    for ds in datasets:
        for i, n in enumerate(ds.names):
            by_id[i].add(n)
    id_collisions = [{"id": i, "names": sorted(v)}
                     for i, v in sorted(by_id.items()) if len(v) > 1]

    # Different spelling, probably one thing.
    by_norm: dict[str, set[str]] = defaultdict(set)
    for n in names:
        by_norm[normalise_name(n)].add(n)
    name_variants = [{"normalised": k, "names": sorted(v)}
                     for k, v in sorted(by_norm.items()) if len(v) > 1]

    return {"names": names, "index": index,
            "id_collisions": id_collisions, "name_variants": name_variants}


def cross_source_duplicates(datasets: list[Dataset]) -> list[dict]:
    """Images whose content appears in more than one source."""
    from .detectors.duplicates import _hash_all, _candidate_pairs, _hamming, \
        NEAR_DUP_DISTANCE

    entries: list[tuple[int, str, int, str]] = []
    for src, ds in enumerate(datasets):
        for key, (_s, h, sha) in _hash_all(ds).items():
            entries.append((src, key, h, sha))

    pairs = _candidate_pairs([(f"{s}:{k}", h) for s, k, h, _sha in entries])
    out = []
    for i, j in sorted(pairs):
        si, ki, hi, shai = entries[i]
        sj, kj, hj, shaj = entries[j]
        if si == sj:
            continue                       # within one source: not our problem
        if (shai and shai == shaj) or _hamming(hi, hj) <= NEAR_DUP_DISTANCE:
            out.append({"a": {"source": si, "key": ki},
                        "b": {"source": sj, "key": kj},
                        "identical": bool(shai and shai == shaj)})
    return out


def plan(roots: list[str | Path]) -> dict:
    """Everything the merge would do, and everything it would get wrong."""
    datasets = [Dataset(r) for r in roots]
    tax = taxonomy(datasets)
    dupes = cross_source_duplicates(datasets)

    remaps = []
    for src, ds in enumerate(datasets):
        moved = {i: tax["index"][n] for i, n in enumerate(ds.names)
                 if tax["index"][n] != i}
        remaps.append({"source": src, "root": str(ds.root),
                       "classes": len(ds.names),
                       "images": sum(1 for s in ds.samples if s.image_path),
                       "remapped_ids": moved})

    # A stem present in two sources would overwrite itself on the way out.
    stems: dict[str, list[int]] = defaultdict(list)
    for src, ds in enumerate(datasets):
        for s in ds.samples:
            stems[s.stem].append(src)
    collisions = sorted(k for k, v in stems.items() if len(set(v)) > 1)

    return {
        "schema": "dsdoctor/merge-plan/1",
        "sources": [str(Path(r).resolve()) for r in roots],
        "merged_classes": tax["names"],
        "sources_detail": remaps,
        "conflicts": {
            "id_collisions": tax["id_collisions"],
            "name_variants": tax["name_variants"],
            "cross_source_duplicates": dupes,
            "filename_collisions": collisions,
        },
        "needs_a_decision": bool(tax["name_variants"]),
    }


def materialise(roots: list[str | Path], plan_doc: dict, out: Path) -> dict:
    """Write the merged dataset. Sources are never modified.

    Label files are rewritten with remapped class ids; images are symlinked.
    Stems are prefixed with their source index wherever two sources use the
    same one, so nothing silently overwrites anything.
    """
    out = Path(out)
    datasets = [Dataset(r) for r in roots]
    names = plan_doc["merged_classes"]
    index = {n: i for i, n in enumerate(names)}
    colliding = set(plan_doc["conflicts"]["filename_collisions"])

    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"names": names, "nc": len(names),
         "train": "images/train", "val": "images/val"}, sort_keys=False))

    written = defaultdict(int)
    for src, ds in enumerate(datasets):
        remap = {i: index[n] for i, n in enumerate(ds.names)}
        for s in ds.samples:
            split = s.split
            stem = f"s{src}_{s.stem}" if s.stem in colliding else s.stem
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            if s.image_path and s.image_path.is_file():
                link_or_copy(s.image_path,
                             out / "images" / split /
                             f"{stem}{s.image_path.suffix}")
            if s.label is not None:
                rows = []
                for b in s.label.boxes:
                    cls = remap.get(b.cls, b.cls)
                    parts = b.raw.split()
                    parts[0] = str(cls)         # keep every other field verbatim
                    rows.append(" ".join(parts))
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(rows) + ("\n" if rows else ""))
            written[split] += 1
    return dict(written)


def write(plan_doc: dict, path: Path) -> Path:
    Path(path).write_text(json.dumps(plan_doc, indent=2, default=str))
    return Path(path)


def summarise(plan_doc: dict) -> str:
    c = plan_doc["conflicts"]
    L = [f"merging {len(plan_doc['sources'])} dataset(s) -> "
         f"{len(plan_doc['merged_classes'])} class(es)"]
    for d in plan_doc["sources_detail"]:
        L.append(f"  {Path(d['root']).name}: {d['images']} images, "
                 f"{d['classes']} classes"
                 + (f", {len(d['remapped_ids'])} id(s) remapped"
                    if d["remapped_ids"] else ""))

    if c["id_collisions"]:
        L += ["", f"  {len(c['id_collisions'])} class id(s) mean different "
                  f"things in different sources:"]
        for row in c["id_collisions"][:8]:
            L.append(f"    id {row['id']}: {', '.join(row['names'])}")
        L.append("    -> resolved by rebuilding ids from names. Concatenating "
                 "these label files without remapping would have merged "
                 "unrelated classes.")

    if c["name_variants"]:
        L += ["", f"  {len(c['name_variants'])} name(s) look like the same "
                  f"class spelled differently:"]
        for row in c["name_variants"][:8]:
            L.append(f"    {', '.join(row['names'])}")
        L.append("    -> kept as SEPARATE classes. Merging them is a "
                 "relabelling decision, and this tool does not make it for "
                 "you. Edit the sources to agree, then merge again.")

    if c["cross_source_duplicates"]:
        L += ["", f"  {len(c['cross_source_duplicates'])} image(s) appear in "
                  f"more than one source:"]
        for row in c["cross_source_duplicates"][:6]:
            kind = "identical" if row["identical"] else "near-duplicate"
            L.append(f"    {kind}: source {row['a']['source']} "
                     f"{row['a']['key']} ~ source {row['b']['source']} "
                     f"{row['b']['key']}")
        L.append("    -> after merging these are duplicates, and if the "
                 "sources split them differently they are train/val leakage. "
                 "Run `dsdoctor resplit` on the result.")

    if c["filename_collisions"]:
        L += ["", f"  {len(c['filename_collisions'])} filename(s) are used by "
                  f"more than one source; they will be prefixed on write."]

    if not any(c[k] for k in c):
        L += ["", "  no conflicts."]
    return "\n".join(L)
