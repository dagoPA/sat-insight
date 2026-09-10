"""The frozen backbone a run uses, chosen through the environment.

Every training and evaluation tool under scripts/ takes its sensor names and names its
result files through this module, so the whole protocol runs on another backbone's
vectors by setting one variable. With SATINSIGHT_BACKBONE unset the run is DOFA base:
the plain sensor names s2, s1 and s2deg, and unsuffixed files. With
SATINSIGHT_BACKBONE=dofal the sensors become s2_dofal, s1_dofal and s2deg_dofal, the
names backbone_extract.py writes the DOFA large vectors under, and every result file
carries the suffix _dofal, so the two backbones' results sit side by side and nothing is
overwritten. Sensors that do not come from a backbone, such as the WorldCover fractions,
keep their names.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

TAG = os.environ.get("SATINSIGHT_BACKBONE", "").strip()
HEADS = ("cumulative", "coral", "softmax")
"""Parameterisations of the instance head: independent cumulative thresholds (the
reference), CORAL's shared weights with one bias per threshold, and a five-class softmax
trained with cross-entropy on the class shares. Chosen through SATINSIGHT_HEAD."""
HEAD = os.environ.get("SATINSIGHT_HEAD", "").strip() or HEADS[0]
if HEAD not in HEADS:
    raise ValueError(f"SATINSIGHT_HEAD must be one of {HEADS}, got {HEAD!r}")
SUFFIX = (f"_{TAG}" if TAG else "") + (f"_{HEAD}" if HEADS[0] != HEAD else "")
"""Result files carry the backbone tag and, for the ablation heads, the head name."""
ENCODED = ("s2", "s1", "s2deg")
"""Sensor names whose vectors come from the backbone."""

MODELS = {
    "": "dofa_base_patch16_224",
    "dofal": "dofa_large_patch16_224",
    "ov": "dofa_base_patch16_224",
    "dofalov": "dofa_large_patch16_224",
}
"""Which pretrained weights each tag stands for; the `ov` tags are the same weights
extracted with overlapping windows (stride 112 px) and per-token averaging."""

STRIDE = 112 if TAG.endswith("ov") else 224
"""Window stride of this tag's extraction; 224 is the non-overlapping reference."""


def sensor(name: str) -> str:
    """The sensor name under which this backbone's vectors of `name` are stored."""
    return f"{name}_{TAG}" if TAG and name in ENCODED else name


def suffixed(path: str | Path) -> str:
    """`path` with the backbone suffix before its extension."""
    p = Path(path)
    return str(p.with_name(p.stem + SUFFIX + p.suffix))


def suffix_of(source: str | Path) -> str:
    """The backbone suffix a persisted predictions file carries, or an empty string.

    The analysis tools take the predictions file as their argument, so the suffix of their
    own outputs must follow the file rather than the environment.
    """
    match = re.search(r"predictions_(?:val|test)(?:_city)?(_[A-Za-z0-9_]+)?\.parquet", str(source))
    return match.group(1) or "" if match else ""


def encoder():
    """The encoder of this backbone, for the tools that extract vectors."""
    from satinsight import encoders

    if TAG == "cfm":
        return encoders.CopernicusFmEncoder()
    return encoders.DofaEncoder(MODELS[TAG])
