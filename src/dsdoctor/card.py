"""The dataset health card: an audit that can travel with the data.

A report is something you read once. A *card* is something you hand over. The
problem this addresses is the one in the README's opening line - an engineer
receiving a dataset they did not create - viewed from the other end: the
sender knows things about the delivery that the receiver has no way to
recover, and nothing in the ML toolchain carries them across the gap.

So this module emits two artefacts, meant to sit in the dataset directory and
be shipped with it:

  ``health.json``      machine-readable: composition, findings, verdict, and a
                       content fingerprint of the exact bytes described.
  ``DATASET_CARD.md``  the same thing for a person.

The fingerprint is what makes the card more than a claim. It is a digest over
every file's path and contents, so ``dsdoctor verify-card`` can answer the
question a receiver actually has - *does this card describe the data in front
of me, or a different version of it?* - without trusting anyone's changelog. A
card whose fingerprint does not match is not a warning about staleness; it is
positive evidence that the dataset changed after it was described.

The verdict here is rule-based, not the agent's. `dsdoctor audit` produces a
triaged judgement with a model in the loop; a card has to be reproducible by
anyone, offline, byte-for-byte, so it applies a fixed severity rule and says
so. The two are complementary and the card records which one it is.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .dataset import Dataset, file_sha1
from .findings import (Finding, CRITICAL, MAJOR, GOVERNANCE, TRAINABILITY,
                       sort_findings)

CARD_SCHEMA = "dsdoctor/health-card/1"

# Files that describe the dataset rather than constituting it. Excluding them
# is what lets a card live *inside* the directory it describes without the act
# of writing it invalidating its own fingerprint.
CARD_FILENAMES = {"health.json", "DATASET_CARD.md", "dataset_manifest.tsv"}


# --------------------------------------------------------------- fingerprint

def iter_content_files(root: Path) -> list[Path]:
    """Every file that is part of the dataset proper, in a stable order."""
    out = [p for p in sorted(root.rglob("*"))
           if p.is_file()
           and p.name not in CARD_FILENAMES
           and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    return out


def manifest(root: Path, workers: int = 8) -> list[tuple[str, int, str]]:
    """(relative path, size, sha1) for every content file.

    Hashing is the expensive part of a card on a large dataset and it is pure
    I/O, so it is threaded. Order is by path, not by completion, because the
    digest built from this must not depend on scheduling.
    """
    files = iter_content_files(root)

    def entry(p: Path) -> tuple[str, int, str]:
        return (p.relative_to(root).as_posix(), p.stat().st_size, file_sha1(p))

    from .progress import track

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(track(pool.map(entry, files), "fingerprinting",
                          total=len(files)))


def digest(rows: list[tuple[str, int, str]]) -> str:
    """One hash standing for the whole dataset.

    Built over ``path\\0sha1`` lines so that renaming a file changes the
    digest even when its contents do not - a dataset whose splits were
    reshuffled is a different dataset, and this is exactly the case the naive
    "hash of hashes" misses.
    """
    import hashlib
    h = hashlib.sha256()
    for rel, _size, sha in rows:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def write_manifest(rows: list[tuple[str, int, str]], path: Path) -> Path:
    path.write_text("".join(f"{rel}\t{size}\t{sha}\n" for rel, size, sha in rows))
    return path


def read_manifest(path: Path) -> list[tuple[str, int, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rel, size, sha = line.split("\t")
        rows.append((rel, int(size), sha))
    return rows


# ------------------------------------------------------------------- verdict

def deterministic_verdict(findings: list[Finding]) -> str:
    """The rule the card applies, stated in code so it can be argued with.

    Governance findings are excluded on purpose: whether you are permitted to
    use a dataset is not a statement about whether it will train.
    """
    trainability = [f for f in findings if f.category == TRAINABILITY]
    if any(f.severity == CRITICAL for f in trainability):
        return "blocked"
    if any(f.severity == MAJOR for f in trainability):
        return "fix_before_training"
    return "usable_with_caveats"


# ---------------------------------------------------------------- the card

def build(ds: Dataset, findings: list[Finding], *,
          groups: list[str] | None = None, detectors_run: list[str] | None = None,
          rows: list[tuple[str, int, str]] | None = None,
          source_root: Path | None = None,
          max_items: int = 200) -> dict:
    """Assemble health.json.

    `source_root` matters whenever the dataset was read through a converted
    view. A COCO or VOC delivery is audited via YOLO label files written to a
    cache directory, and fingerprinting *those* would describe an artefact
    this tool generated rather than the bytes the engineer was sent - so
    `verify-card` against the actual delivery could never match, and the
    vendor hand-off the card exists for would not work at all.
    """
    root = Path(source_root) if source_root else ds.root
    rows = manifest(root) if rows is None else rows
    s = ds.summary()
    findings = sort_findings(findings)

    def as_row(f: Finding) -> dict:
        return {
            "type": f.type,
            "severity": f.severity,
            "category": f.category,
            "detector": f.detector,
            "title": f.title,
            "affected_files": f.n_items,
            "items": f.items[:max_items],
            "items_truncated": max(0, f.n_items - max_items),
            "fix_action": (f.fix or {}).get("action"),
        }

    trainability = [f for f in findings if f.category == TRAINABILITY]
    governance = [f for f in findings if f.category == GOVERNANCE]

    return {
        "schema": CARD_SCHEMA,
        "dsdoctor_version": __version__,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": root.name,
            "path": str(root),
            "read_through": (str(ds.root) if root != ds.root else None),
            "task": ds.task,
            "classes": s["names"],
            "num_classes": s["nc"],
            "splits": s["splits"],
            "total_boxes": s["total_boxes"],
            "class_counts": s["class_counts"],
            "data_yaml_problem": s["yaml_error"],
        },
        "fingerprint": {
            "algorithm": "sha256 over path+sha1 of every content file",
            "digest": digest(rows),
            "files": len(rows),
            "bytes": sum(size for _, size, _ in rows),
        },
        "checks": {
            "groups": ["core", *(groups or [])],
            "detectors_run": detectors_run or [],
            "verdict_source": "rule-based (severity), not the auditing agent",
        },
        "verdict": deterministic_verdict(findings),
        "summary": {
            "critical": sum(1 for f in trainability if f.severity == CRITICAL),
            "major": sum(1 for f in trainability if f.severity == MAJOR),
            "minor": sum(1 for f in trainability
                         if f.severity not in (CRITICAL, MAJOR)),
            "governance": len(governance),
        },
        "findings": [as_row(f) for f in trainability],
        "governance_findings": [as_row(f) for f in governance],
    }


VERDICT_SENTENCE = {
    "blocked": "**Do not train on this yet.** At least one defect here will "
               "stop the run or make its results meaningless.",
    "fix_before_training": "**Fix these before training.** The run will "
                           "complete, but the model or the metrics will be "
                           "worse than they need to be.",
    "usable_with_caveats": "**Usable.** No critical or major trainability "
                           "defect was found by the checks that were run.",
}


def render_markdown(health: dict) -> str:
    d, fp, sm = health["dataset"], health["fingerprint"], health["summary"]
    L: list[str] = []
    L.append(f"# Dataset card: `{d['name']}`")
    L.append("")
    L.append(VERDICT_SENTENCE.get(health["verdict"], health["verdict"]))
    L.append("")
    L.append(f"`{sm['critical']}` critical · `{sm['major']}` major · "
             f"`{sm['minor']}` minor · `{sm['governance']}` governance")
    L.append("")

    L.append("## What this describes")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| Fingerprint | `{fp['digest'][:16]}…` |")
    L.append(f"| Files | {fp['files']:,} ({fp['bytes'] / 1e6:,.1f} MB) |")
    L.append(f"| Classes | {d['num_classes']} |")
    L.append(f"| Boxes | {d['total_boxes']:,} |")
    for name, v in d["splits"].items():
        L.append(f"| Split `{name}` | {v['images']:,} images, {v['boxes']:,} boxes |")
    L.append(f"| Checks run | {', '.join(health['checks']['groups'])} |")
    if d.get("read_through"):
        L.append(f"| Read through | a converted YOLO view; the fingerprint "
                 f"covers the original delivery |")
    L.append(f"| Produced by | dsdoctor {health['dsdoctor_version']}, "
             f"{health['generated']} |")
    L.append("")
    L.append("Verify that this card describes the data you actually received:")
    L.append("")
    L.append("```bash")
    L.append("dsdoctor verify-card /path/to/dataset")
    L.append("```")
    L.append("")
    L.append("A fingerprint mismatch is not a stale-file warning. It is "
             "evidence that the dataset changed after this card was written.")
    L.append("")

    L.append("## Trainability findings")
    L.append("")
    if not health["findings"]:
        L.append("None. Every check that was run passed.")
        L.append("")
    for f in health["findings"]:
        L.append(f"- **[{f['severity']}] {f['type']}** — {f['title']}  ")
        L.append(f"  {f['affected_files']} file(s) affected · detector "
                 f"`{f['detector']}`"
                 + (f" · fix `{f['fix_action']}`" if f["fix_action"] else ""))
    L.append("")

    L.append("## Governance and privacy")
    L.append("")
    L.append("These do not affect whether the dataset trains. They affect "
             "whether it may lawfully be trained on or published.")
    L.append("")
    if not health["governance_findings"]:
        L.append("Nothing flagged by the checks that were run. Note that the "
                 "`privacy` check group is opt-in: if it does not appear in "
                 "*Checks run* above, these questions were not asked.")
        L.append("")
    for f in health["governance_findings"]:
        L.append(f"- **[{f['severity']}] {f['type']}** — {f['title']}  ")
        L.append(f"  {f['affected_files']} file(s) affected · detector "
                 f"`{f['detector']}`")
    L.append("")

    L.append("## Class distribution")
    L.append("")
    counts = d["class_counts"]
    if counts:
        L.append("| class | instances |")
        L.append("|---|---:|")
        for k, v in list(counts.items())[:40]:
            L.append(f"| `{k}` | {v:,} |")
        if len(counts) > 40:
            L.append(f"| … {len(counts) - 40} more | |")
    L.append("")

    L.append("## How to read this")
    L.append("")
    L.append("Every finding above was produced by a deterministic check that "
             "read the files directly. No language model generated any claim "
             "on this page, and the verdict is a fixed severity rule "
             "(`critical` → blocked, `major` → fix first), not a judgement. "
             "For a triaged, ranked and explained version, run "
             "`dsdoctor audit`.")
    L.append("")
    L.append("Absence of a finding is only evidence about the checks that "
             "were run, which are listed above.")
    L.append("")
    return "\n".join(L)


def write(ds: Dataset, findings: list[Finding], out_dir: Path, *,
          groups: list[str] | None = None,
          detectors_run: list[str] | None = None,
          source_root: Path | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest(Path(source_root) if source_root else ds.root)
    health = build(ds, findings, groups=groups, detectors_run=detectors_run,
                   rows=rows, source_root=source_root)
    hp = out_dir / "health.json"
    hp.write_text(json.dumps(health, indent=2))
    mp = write_manifest(rows, out_dir / "dataset_manifest.tsv")
    cp = out_dir / "DATASET_CARD.md"
    cp.write_text(render_markdown(health))
    return {"health": hp, "card": cp, "manifest": mp}


# ------------------------------------------------------------ verification

def verify(ds: Dataset, health: dict,
           manifest_path: Path | None = None) -> dict:
    """Does this card describe the dataset in front of us?"""
    rows = manifest(ds.root)
    current = digest(rows)
    claimed = health.get("fingerprint", {}).get("digest", "")
    result = {
        "match": current == claimed,
        "claimed_digest": claimed,
        "current_digest": current,
        "claimed_files": health.get("fingerprint", {}).get("files"),
        "current_files": len(rows),
        "added": [], "removed": [], "modified": [],
    }
    if result["match"] or manifest_path is None or not Path(manifest_path).is_file():
        return result

    # With the manifest we can say *what* changed, which is the difference
    # between "do not trust this" and "here is the diff to review".
    before = {rel: sha for rel, _s, sha in read_manifest(Path(manifest_path))}
    after = {rel: sha for rel, _s, sha in rows}
    result["added"] = sorted(set(after) - set(before))
    result["removed"] = sorted(set(before) - set(after))
    result["modified"] = sorted(r for r in set(before) & set(after)
                                if before[r] != after[r])
    return result
