"""Inject known defects into a clean corpus and record exactly what was done.

Every injector returns the set of ``(defect_type, sample_key)`` facts it
created. That set is the ground truth the scorer matches against, which is why
the evaluation needs no LLM judge and no human grading: either the report named
the right defect on the right file or it did not.

Two subtleties this module takes care to be honest about:

*collateral* - some injections legitimately trip a second detector. Writing
pixel coordinates into a label file also puts the box outside [0,1], and
out_of_bounds firing on it is a correct observation, not a false positive. Each
injector declares the collateral facts it expects, and the scorer excludes them
from both recall and the false-positive count.

*symmetry* - a defect that relates two files (leakage, near-duplicates) records
both keys, because that is what a correct report names.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

Fact = tuple[str, str]      # (defect_type, "split/stem") or (type, "<dataset>")
DATASET_LEVEL = "<dataset>"

TRAIN = "train"
VAL = "val"


@dataclass
class Ctx:
    """A working copy of the corpus that injectors mutate in place."""

    root: Path
    rng: random.Random
    names: list[str] = field(default_factory=list)
    _touched: set[str] = field(default_factory=set)

    # -------------------------------------------------------------- helpers

    def label_path(self, key: str) -> Path:
        split, stem = key.split("/", 1)
        return self.root / "labels" / split / f"{stem}.txt"

    def image_path(self, key: str) -> Path:
        split, stem = key.split("/", 1)
        return self.root / "images" / split / f"{stem}.jpg"

    def keys(self, split: str | None = None) -> list[str]:
        out = []
        for sp in ([split] if split else ("train", "val")):
            d = self.root / "labels" / sp
            if d.is_dir():
                out += [f"{sp}/{p.stem}" for p in sorted(d.iterdir())
                        if p.suffix == ".txt"]
        return out

    def read(self, key: str) -> list[list[str]]:
        return [ln.split() for ln in
                self.label_path(key).read_text().splitlines() if ln.strip()]

    def write(self, key: str, rows: list[list[str]]) -> None:
        self.label_path(key).write_text(
            "\n".join(" ".join(r) for r in rows) + ("\n" if rows else ""))

    def new_stem(self) -> str:
        """A filename that does not give the game away.

        Naming a leaked copy `<stem>_v0` or a duplicate `<stem>_dup1` makes the
        defect detectable from the file listing alone, which is not how real
        leakage arrives and would let an arm score without ever comparing two
        images. Injected files therefore get ordinary COCO-style ids.
        """
        existing = {k.split("/", 1)[1] for k in self.keys()}
        while True:
            stem = f"{self.rng.randrange(10 ** 11, 10 ** 12):012d}"
            if stem not in existing:
                return stem

    def pick(self, n: int, split: str | None = None,
             need_boxes: int = 1) -> list[str]:
        """Choose n untouched files, so injections never overlap by accident."""
        pool = [k for k in self.keys(split)
                if k not in self._touched and len(self.read(k)) >= need_boxes]
        self.rng.shuffle(pool)
        chosen = pool[:n]
        self._touched.update(chosen)
        return chosen


INJECTORS: dict[str, callable] = {}


def injector(name: str):
    def deco(fn):
        INJECTORS[name] = fn
        return fn
    return deco


# ------------------------------------------------------------ geometry bugs

@injector("out_of_bounds")
def inj_out_of_bounds(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        rows = ctx.read(key)
        i = ctx.rng.randrange(len(rows))
        # Widen the box well past the frame, the way a bad affine augmentation
        # writes it back out.
        rows[i][3] = f"{float(rows[i][3]) + 0.9:.8f}"
        rows[i][4] = f"{float(rows[i][4]) + 0.9:.8f}"
        ctx.write(key, rows)
        facts.append(("out_of_bounds", key))
    return facts, []


@injector("degenerate_box")
def inj_degenerate(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        rows = ctx.read(key)
        i = ctx.rng.randrange(len(rows))
        rows[i][3] = "0.00000000"
        ctx.write(key, rows)
        facts.append(("degenerate_box", key))
    return facts, []


@injector("tiny_box")
def inj_tiny(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        rows = ctx.read(key)
        i = ctx.rng.randrange(len(rows))
        rows[i][3] = "0.00050000"
        rows[i][4] = "0.00050000"
        ctx.write(key, rows)
        facts.append(("tiny_box", key))
    return facts, []


@injector("duplicate_annotation")
def inj_duplicate_anno(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        rows = ctx.read(key)
        rows.append(list(rows[ctx.rng.randrange(len(rows))]))
        ctx.write(key, rows)
        facts.append(("duplicate_annotation", key))
    return facts, []


@injector("denormalised_coords")
def inj_denormalised(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """The classic export bug: coordinates left in pixels."""
    facts, collateral = [], []
    for key in ctx.pick(n, need_boxes=2):
        with Image.open(ctx.image_path(key)) as im:
            w, h = im.size
        rows = ctx.read(key)
        for r in rows:
            r[1] = f"{float(r[1]) * w:.4f}"
            r[2] = f"{float(r[2]) * h:.4f}"
            r[3] = f"{float(r[3]) * w:.4f}"
            r[4] = f"{float(r[4]) * h:.4f}"
        ctx.write(key, rows)
        facts.append(("denormalised_coords", key))
        # Pixel coordinates are by definition outside [0,1]; a report that
        # also says so is right, so this must not count against it.
        collateral.append(("out_of_bounds", key))
    return facts, collateral


# ------------------------------------------------------------- class errors

@injector("class_id_out_of_range")
def inj_bad_class_id(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts, collateral = [], []
    nc = len(ctx.names)
    for key in ctx.pick(n):
        rows = ctx.read(key)
        i = ctx.rng.randrange(len(rows))
        rows[i][0] = str(nc + ctx.rng.randrange(1, 4))
        ctx.write(key, rows)
        facts.append(("class_id_out_of_range", key))
    return facts, collateral


@injector("class_swap")
def inj_class_swap(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """Relabel one class as another across several files, in one direction.

    This imitates a broken class-id mapping in an export script, which is the
    realistic way a whole batch of labels goes wrong at once.
    """
    facts, collateral = [], []
    if len(ctx.names) < 2:
        return facts, collateral
    # Prefer a visually distinguishable pair so the reference detector has a
    # fair chance of noticing.
    pairs = [("car", "truck"), ("cup", "bowl"), ("chair", "bench"),
             ("bottle", "cup"), ("backpack", "handbag")]
    src = dst = None
    for a, b in pairs:
        if a in ctx.names and b in ctx.names:
            src, dst = ctx.names.index(a), ctx.names.index(b)
            break
    if src is None:
        src, dst = 0, 1

    pool = [k for k in ctx.keys() if k not in ctx._touched
            and any(int(float(r[0])) == src for r in ctx.read(k))]
    ctx.rng.shuffle(pool)
    for key in pool[:n]:
        rows = ctx.read(key)
        changed = False
        for r in rows:
            if int(float(r[0])) == src:
                r[0] = str(dst)
                changed = True
        if not changed:
            continue
        ctx.write(key, rows)
        ctx._touched.add(key)
        facts.append(("class_swap", key))
    return facts, collateral


@injector("extreme_class_imbalance")
def inj_imbalance(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """Strip a class down until it can no longer be trained or validated."""
    facts, collateral = [], []
    counts: dict[int, int] = {}
    for k in ctx.keys():
        for r in ctx.read(k):
            c = int(float(r[0]))
            counts[c] = counts.get(c, 0) + 1
    # Target a mid-frequency class: gutting the rarest is less interesting and
    # gutting the most common changes the dataset's character.
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) < 3:
        return facts, collateral
    target = ranked[len(ranked) // 2][0]

    kept = 0
    for k in ctx.keys():
        rows = ctx.read(k)
        out = []
        for r in rows:
            if int(float(r[0])) == target:
                if kept < 2:
                    kept += 1
                    out.append(r)
                continue
            out.append(r)
        if len(out) != len(rows):
            ctx.write(k, out)
            if not out:
                # An emptied file is a real, separate observation.
                collateral.append(("empty_label_file", k))
    facts.append(("extreme_class_imbalance", DATASET_LEVEL))
    return facts, collateral


# ---------------------------------------------------------- file structure

@injector("missing_label_file")
def inj_missing_label(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        ctx.label_path(key).unlink()
        facts.append(("missing_label_file", key))
    return facts, []


@injector("empty_label_file")
def inj_empty_label(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        ctx.label_path(key).write_text("")
        facts.append(("empty_label_file", key))
    return facts, []


@injector("orphan_label_file")
def inj_orphan(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for _ in range(n):
        stem = ctx.new_stem()
        key = f"{TRAIN}/{stem}"
        (ctx.root / "labels" / TRAIN / f"{stem}.txt").write_text(
            "0 0.50000000 0.50000000 0.20000000 0.20000000\n")
        facts.append(("orphan_label_file", key))
    return facts, []


@injector("malformed_label_row")
def inj_malformed(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        rows = ctx.read(key)
        i = ctx.rng.randrange(len(rows))
        # A trailing confidence column: what you get when a prediction dump is
        # mistaken for a label file.
        rows[i] = rows[i] + ["0.87"]
        ctx.write(key, rows)
        facts.append(("malformed_label_row", key))
    return facts, []


@injector("corrupt_image")
def inj_corrupt_image(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    facts = []
    for key in ctx.pick(n):
        p = ctx.image_path(key)
        data = p.read_bytes()
        p.write_bytes(data[:len(data) // 3])   # truncated mid-scan
        facts.append(("corrupt_image", key))
    return facts, []


@injector("yaml_inconsistency")
def inj_yaml(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    p = ctx.root / "data.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["nc"] = len(cfg["names"]) + 3
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return [("yaml_inconsistency", DATASET_LEVEL)], []


# -------------------------------------------------------------- duplicates

@injector("train_val_leakage")
def inj_leakage(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """Copy train images into val under new names - the split bug that makes
    a model look ready to ship when it is not."""
    facts = []
    for key in ctx.pick(n, split=TRAIN):
        stem = ctx.new_stem()
        shutil.copy2(ctx.image_path(key),
                     ctx.root / "images" / VAL / f"{stem}.jpg")
        shutil.copy2(ctx.label_path(key),
                     ctx.root / "labels" / VAL / f"{stem}.txt")
        vkey = f"{VAL}/{stem}"
        ctx._touched.add(vkey)
        facts += [("train_val_leakage", key), ("train_val_leakage", vkey)]
    return facts, []


@injector("near_duplicate_image")
def inj_near_duplicate(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """Re-encode a train image at slightly lower quality and add it again."""
    facts = []
    for key in ctx.pick(n, split=TRAIN):
        stem = ctx.new_stem()
        with Image.open(ctx.image_path(key)) as im:
            im.convert("RGB").save(
                ctx.root / "images" / TRAIN / f"{stem}.jpg", quality=82)
        shutil.copy2(ctx.label_path(key),
                     ctx.root / "labels" / TRAIN / f"{stem}.txt")
        dkey = f"{TRAIN}/{stem}"
        ctx._touched.add(dkey)
        facts += [("near_duplicate_image", key), ("near_duplicate_image", dkey)]
    return facts, []


# ------------------------------------------------------------------ driver

# --------------------------------------------- defects the later groups catch
#
# These exist so the opt-in check groups can be scored the same way the core
# ones are, against ground truth rather than against a claim. Each mirrors how
# the defect actually arrives in a delivered dataset.

@injector("exif_orientation")
def inj_exif_orientation(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """A phone photo whose rotation lives in metadata rather than in pixels."""
    facts = []
    for key in ctx.pick(n):
        path = ctx.image_path(key)
        with Image.open(path) as im:
            im = im.convert("RGB")
            exif = Image.Exif()
            exif[274] = ctx.rng.choice([3, 6, 8])
            im.save(path, "JPEG", exif=exif, quality=95)
        facts.append(("exif_orientation", key))
    return facts, []


@injector("gps_metadata")
def inj_gps_metadata(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """Coordinates left in the file by the camera that took it."""
    facts = []
    for key in ctx.pick(n):
        path = ctx.image_path(key)
        with Image.open(path) as im:
            im = im.convert("RGB")
            exif = Image.Exif()
            gps = exif.get_ifd(0x8825)
            gps.update({1: "N", 2: (float(ctx.rng.randrange(0, 90)), 30.0, 0.0),
                        3: "W", 4: (float(ctx.rng.randrange(0, 180)), 7.0, 0.0)})
            im.save(path, "JPEG", exif=exif, quality=95)
        facts.append(("gps_metadata", key))
    return facts, []


@injector("class_absent_from_val")
def inj_class_absent_from_val(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """A class stripped out of val only - it trains, it cannot be scored.

    Reported at dataset level because the defect is a property of the split,
    not of any one file.
    """
    counts: dict[int, int] = {}
    for key in ctx.keys(VAL):
        for row in ctx.read(key):
            counts[int(row[0])] = counts.get(int(row[0]), 0) + 1
    # Prefer thin classes: removing a common one would also trip the imbalance
    # detector and muddy which check found what.
    candidates = [c for c, k in sorted(counts.items(), key=lambda kv: kv[1])
                  if k > 0][:max(n, 1)]
    facts = []
    for cls in candidates[:n]:
        for key in ctx.keys(VAL):
            rows = ctx.read(key)
            kept = [r for r in rows if int(r[0]) != cls]
            if len(kept) != len(rows):
                ctx.write(key, kept)
        facts.append(("class_absent_from_val", DATASET_LEVEL))
    return facts, []


@injector("undetectable_at_imgsz")
def inj_undetectable(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """Annotations of objects far too small to survive the network's stride.

    Deliberately larger than the `tiny_box` threshold, so the two checks are
    telling the engineer different things: `tiny_box` is "the dataloader will
    drop this", this is "the architecture cannot represent it at your imgsz".
    """
    facts = []
    for key in ctx.pick(n):
        rows = ctx.read(key)
        i = ctx.rng.randrange(len(rows))
        rows[i][3] = f"{0.008:.8f}"      # 5.1px at 640: above tiny, below stride
        rows[i][4] = f"{0.008:.8f}"
        ctx.write(key, rows)
        facts.append(("undetectable_at_imgsz", key))
    return facts, []


@injector("template_annotation")
def inj_template(ctx: Ctx, n: int) -> tuple[list[Fact], list[Fact]]:
    """A pre-annotation pass whose suggested box was never adjusted."""
    facts = []
    keys = ctx.pick(max(n, 6))
    for key in keys:
        rows = ctx.read(key)
        rows.append(["0", "0.50000000", "0.50000000", "0.20000000", "0.20000000"])
        ctx.write(key, rows)
        facts.append(("template_annotation", key))
    return facts, []


def build_case(base: Path, out: Path, recipe: dict[str, int], seed: int) -> dict:
    """Copy the clean corpus, apply a recipe, and return the ground truth."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)

    cfg = yaml.safe_load((out / "data.yaml").read_text())
    ctx = Ctx(root=out, rng=random.Random(seed), names=list(cfg["names"]))

    facts: list[Fact] = []
    collateral: list[Fact] = []
    applied: dict[str, int] = {}
    for dtype, count in recipe.items():
        if dtype not in INJECTORS:
            raise KeyError(f"unknown defect type {dtype!r}")
        f, c = INJECTORS[dtype](ctx, count)
        facts += f
        collateral += c
        applied[dtype] = len(f)

    return {
        "seed": seed,
        "recipe": recipe,
        "applied": applied,
        "ground_truth": sorted(set(facts)),
        "collateral": sorted(set(collateral) - set(facts)),
    }
