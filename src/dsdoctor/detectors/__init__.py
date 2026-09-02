"""Deterministic detectors: every defect claim in this system starts here.

A detector is a pure function ``Dataset -> list[Finding]``. No detector calls
an LLM. That split is the central design decision of the project: the language
model decides *which* checks matter, how to group them and what to tell the
user, but it never invents a fact about the data. Anything the report asserts
can be traced back to a row a detector actually read.

Detectors belong to a **group**. Only ``core`` runs by default, and that is a
measurement decision rather than a stylistic one: the results table in the
README was produced with exactly the core set, so anything added later is
opt-in until it has been evaluated on the same twelve cases. ``--checks`` on
the command line widens the set; ``EXTRA_GROUPS`` lists what is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..dataset import Dataset
from ..findings import Finding

CORE = "core"

# Opt-in groups, with the one-line rationale the CLI prints for each.
EXTRA_GROUPS: dict[str, str] = {
    "split": "train/val split integrity: classes that cannot be validated, "
             "unusable split ratios",
    "metadata": "capture metadata that silently contradicts the labels "
                "(EXIF orientation)",
    "privacy": "governance and privacy: EXIF GPS, missing licence, "
               "representation skew",
    "training": "fit against the training configuration you intend to use: "
                "objects too small to detect at --imgsz, images over max_det",
    "annotations": "annotation smells: one box repeated verbatim across many "
                   "images, whole-frame placeholder boxes",
}
ALL_GROUPS = [CORE, *EXTRA_GROUPS]


@dataclass
class Detector:
    name: str
    description: str
    fn: Callable[[Dataset], list[Finding]]
    reads_pixels: bool = False
    heavy: bool = False           # needs the optional `vision` extra
    experimental: bool = False    # measured as net-harmful; opt in explicitly
    covers: tuple[str, ...] = ()  # defect types this detector can emit
    group: str = CORE             # core runs by default; others need --checks
    origin: str = "builtin"       # "builtin" or the plugin that supplied it


REGISTRY: dict[str, Detector] = {}


def register(name: str, description: str, *, reads_pixels: bool = False,
             heavy: bool = False, experimental: bool = False,
             covers: tuple[str, ...] = (), group: str = CORE,
             origin: str = "builtin"):
    def deco(fn):
        REGISTRY[name] = Detector(name=name, description=description, fn=fn,
                                  reads_pixels=reads_pixels, heavy=heavy,
                                  experimental=experimental, covers=covers,
                                  group=group, origin=origin)
        return fn
    return deco


def available(include_experimental: bool = False,
              groups: tuple[str, ...] | list[str] | None = None) -> list[Detector]:
    """Detectors the audit should offer.

    Experimental detectors are excluded by default. That is a measured
    decision, not caution: see README.md, Improvement changelog, iteration 4.

    Non-core groups are excluded by default for a different reason: they have
    not been through the twelve-case evaluation, so including them would make
    the published numbers describe a configuration nobody measured.
    """
    wanted = {CORE, *(groups or ())}
    return [d for d in REGISTRY.values()
            if d.group in wanted
            and (include_experimental or not d.experimental)]


def resolve_groups(spec: str | None) -> list[str]:
    """Turn a ``--checks`` value into a group list.

    Accepts "all", or a comma-separated subset of EXTRA_GROUPS. Unknown names
    raise rather than being ignored - silently running fewer checks than the
    user asked for is exactly the failure this tool exists to prevent.
    """
    if not spec:
        return []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if "all" in parts:
        return list(EXTRA_GROUPS)
    unknown = [p for p in parts if p not in EXTRA_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown check group(s): {', '.join(unknown)}. "
            f"Available: {', '.join(EXTRA_GROUPS)}, or 'all'.")
    return parts


def run(name: str, ds: Dataset) -> list[Finding]:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name].fn(ds)


# Importing the modules is what populates REGISTRY.
from . import structure, geometry, classes, duplicates  # noqa: E402,F401
from . import shapes                                   # noqa: E402,F401
from . import split, metadata, privacy, training        # noqa: E402,F401
from . import provenance                               # noqa: E402,F401

try:  # optional, pulls in torch via the `vision` extra
    from . import consistency  # noqa: F401
except ImportError:  # pragma: no cover - depends on install extras
    pass


def load_plugins() -> list[str]:
    """Load third-party detectors advertised on the ``dsdoctor.detectors``
    entry-point group.

    A plugin is any callable that takes no arguments and calls ``register``.
    An organisation with its own conventions - a naming scheme, a licence
    policy, a class taxonomy - can ship those checks as a normal package
    instead of forking this one. A plugin that raises on import is reported
    and skipped: a broken third-party check must not take the audit down.
    """
    from importlib.metadata import entry_points

    loaded: list[str] = []
    try:
        eps = entry_points(group="dsdoctor.detectors")
    except TypeError:  # pragma: no cover - Python <3.10 selection API
        eps = entry_points().get("dsdoctor.detectors", [])
    for ep in eps:
        try:
            before = set(REGISTRY)
            ep.load()()
            for name in set(REGISTRY) - before:
                REGISTRY[name].origin = ep.name
            loaded.append(ep.name)
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"warning: detector plugin {ep.name!r} failed to load: "
                  f"{type(exc).__name__}: {exc}")
    return loaded
