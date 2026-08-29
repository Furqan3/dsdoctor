"""A tiny synthetic dataset, built in a temp dir.

The evaluation corpus needs a network round trip and 600 images; the unit
tests need neither. Everything here is generated, so the tests run offline in
about a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NAMES = ["person", "car", "cup"]


def write_sample(root: Path, split: str, stem: str, rows: list[str],
                 size=(64, 48), seed: int = 0) -> None:
    """Write one image/label pair.

    The pixels are seeded noise rather than a flat colour on purpose: a
    difference hash of a flat image is all zeros, so every flat image looks
    like a perfect duplicate of every other one and the duplicate detectors
    fire on a dataset that is actually fine.
    """
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(root / "images" / split / f"{stem}.jpg", quality=95)
    (root / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(rows) + ("\n" if rows else ""))


@pytest.fixture
def clean_root(tmp_path: Path) -> Path:
    """A small dataset that every detector should be silent about."""
    root = tmp_path / "ds"
    # Enough instances per class per split to clear the class detector's
    # train (>=10) and val (>=3) thresholds with room to spare.
    for i in range(14):
        rows = [f"{c} {0.2 + 0.15 * c:.4f} {0.3 + 0.12 * c:.4f} 0.10 0.10"
                for c in range(len(NAMES))]
        rows.append(f"{i % 3} 0.5 0.5 0.20 0.20")
        write_sample(root, "train", f"t{i:03d}", rows, seed=i)
    for i in range(6):
        rows = [f"{c} {0.25 + 0.14 * c:.4f} {0.35 + 0.11 * c:.4f} 0.09 0.09"
                for c in range(len(NAMES))]
        write_sample(root, "val", f"v{i:03d}", rows, seed=100 + i)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": NAMES, "nc": len(NAMES)}, sort_keys=False))
    return root
