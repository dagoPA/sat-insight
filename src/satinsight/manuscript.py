"""Shared pieces of the manuscript figures: the canon, the palette, and the guard.

The paper states that every figure regenerates from one canonical results file. That claim
is only worth making if a drifted artifact breaks the build instead of quietly redrawing
the page with different numbers, so the drivers do not merely read `data/` and plot it.
They recompute each published quantity from the per-seed artifacts and pass it through
`agrees`, which raises when the recomputation departs from the value the canon carries.
A rerun that changes a number therefore fails loudly, and the figure and the text can
never disagree without someone noticing.

The palette is fixed here because three figures share it and a split that is blue in one
panel and orange in the next would be read as a different quantity. Blue is always the
fourteen validation cities, used for selection; red is always the fourteen held-out test
cities.
"""

from __future__ import annotations

import json
from pathlib import Path

CANON_PATH = Path("docs/manuscript/canonical_results.json")

VALIDATION = "#3274a1"
"""The fourteen cities that selected every configuration."""

TEST = "#c44e52"
"""The fourteen held-out test cities, scored with validation-selected models."""

ORACLE = "#2f8f4e"
"""The fully supervised upper bound, in every panel that draws one."""

MUTED = "#9aa5ab"
"""Reference marks that carry no split: chance lines, incumbents, zero."""

TRANSFER = "#7d5ba6"
"""Quantities belonging to neither split, which today means the zero-shot countries.

They would otherwise have to borrow a split color and be read as a Mexican number.
"""

INK = "#2c3e50"

STYLE = {"style": "whitegrid", "context": "talk", "font_scale": 0.68}
"""Seaborn theme every manuscript figure opens with, so panel text matches across pages."""


def canon(path: Path = CANON_PATH) -> dict:
    """The canonical results file the manuscript quotes."""
    return json.loads(Path(path).read_text())


def agrees(value: float, expected: float, *, name: str, tolerance: float = 1e-3) -> float:
    """Returns `value` when it matches the published number, and raises when it does not.

    The tolerance covers the rounding in the canon, which stores four decimals, and nothing
    else. A recomputation that lands outside it means the artifact under `data/` is no
    longer the one the manuscript was written from, and the figure must not be drawn from
    it silently.
    """
    if expected is None:
        raise KeyError(f"{name} is absent from the canon")
    if abs(float(value) - float(expected)) > tolerance:
        raise ValueError(
            f"{name}: recomputed {float(value):.4f} against {float(expected):.4f} in the "
            f"canon, beyond the {tolerance} tolerance. The artifact and the manuscript "
            f"have diverged; regenerate the canon deliberately or fix the artifact."
        )
    return float(value)


BACKBONES = (("", "DOFA-B"), ("dofal", "DOFA-L"))
"""The backbones every figure and table reports side by side: tag and printed name.

The empty tag is DOFA base, whose block sits at the root of the canon and whose files
carry no suffix; every other tag is a block of the canon under its own name and a
suffix on every artifact.
"""

LARGE_HATCH = "///"
"""Bars of the second backbone are hatched so the split colors stay for the split alone."""


def block(book: dict, tag: str) -> dict:
    """The canon block of one backbone: the root for DOFA base, a named block otherwise."""
    return book if not tag else book[tag]


def suffixed(path: str, tag: str) -> str:
    """An artifact path with the backbone suffix before its extension."""
    if not tag:
        return path
    stem, dot, extension = path.rpartition(".")
    return f"{stem}_{tag}{dot}{extension}"
