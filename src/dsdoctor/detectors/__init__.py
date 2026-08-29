"""Deterministic detectors: every defect claim in this system starts here.

A detector is a pure function ``Dataset -> list[Finding]``. No detector calls
an LLM. That split is the central design decision of the project: the language
model decides *which* checks matter, how to group them and what to tell the
user, but it never invents a fact about the data. Anything the report asserts
can be traced back to a row a detector actually read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..dataset import Dataset
from ..findings import Finding


@dataclass
class Detector:
    name: str
    description: str
    fn: Callable[[Dataset], list[Finding]]
    reads_pixels: bool = False
    heavy: bool = False           # needs the optional `vision` extra
    experimental: bool = False    # measured as net-harmful; opt in explicitly
    covers: tuple[str, ...] = ()  # defect types this detector can emit


REGISTRY: dict[str, Detector] = {}


def register(name: str, description: str, *, reads_pixels: bool = False,
             heavy: bool = False, experimental: bool = False,
             covers: tuple[str, ...] = ()):
    def deco(fn):
        REGISTRY[name] = Detector(name=name, description=description, fn=fn,
                                  reads_pixels=reads_pixels, heavy=heavy,
                                  experimental=experimental, covers=covers)
        return fn
    return deco


def available(include_experimental: bool = False) -> list[Detector]:
    """Detectors the audit should offer.

    Experimental detectors are excluded by default. That is a measured
    decision, not caution: see README.md, Improvement changelog, iteration 4.
    """
    return [d for d in REGISTRY.values()
            if include_experimental or not d.experimental]


def run(name: str, ds: Dataset) -> list[Finding]:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name].fn(ds)


# Importing the modules is what populates REGISTRY.
from . import structure, geometry, classes, duplicates  # noqa: E402,F401

try:  # optional, pulls in torch via the `vision` extra
    from . import consistency  # noqa: F401
except ImportError:  # pragma: no cover - depends on install extras
    pass
