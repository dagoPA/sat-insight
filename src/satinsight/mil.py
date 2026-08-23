"""Attention-based multiple instance learning over the bags of a city.

A bag is a municipality and its instances are the 160 m tokens the frozen foundation model
produced. The bag carries a single ordinal label and nothing says which instance explains
it; recovering that is the whole point of the project.

The architecture is gated attention as Ilse et al. introduced it in 2018. Each instance
gets a scalar weight, the weights sum to one over the bag, and the bag vector is their
weighted average. That weight is the heat map: it is what gets aggregated per AGEB and
scored against the grade the model never saw.

CLAM adds a per-class attention branch and an instance-level clustering loss on top of the
same mechanism. It is left as an ablation rather than the starting point: the extra branch
would have to be collapsed somehow to produce one map per bag, and a map that is an
artefact of how the branches were merged is harder to defend than one the model produced
directly.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

HIDDEN = 256
"""Width the instance vectors are projected to before attention."""

ATTENTION = 128
"""Width of the attention hidden layer."""

DROPOUT = 0.25
"""Applied after the projection. Bags are few —412— and instances many, so the model can
memorise which municipality it is looking at long before it learns what deprivation looks
like."""


def build(dim_in: int, n_classes: int = 5, hidden: int = HIDDEN, attention: int = ATTENTION):
    """Builds the gated attention MIL network. Torch is imported here, not at module level.

    Keeping the import inside means the bag assembly, the partition and the evaluation
    harness all stay importable without a deep learning stack, which is what lets the tests
    run without one.
    """
    import torch
    from torch import nn

    class GatedAttentionMIL(nn.Module):
        """One weight per instance, one prediction per bag.

        The gate is what separates this from plain attention: one branch proposes how
        interesting an instance looks and the other decides how much of that proposal to
        let through. Without it the attention saturates on whatever dominates the bag,
        which over a municipality is the residential fabric that covers most of it.
        """

        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Dropout(DROPOUT))
            self.attend = nn.Sequential(nn.Linear(hidden, attention), nn.Tanh())
            self.gate = nn.Sequential(nn.Linear(hidden, attention), nn.Sigmoid())
            self.weigh = nn.Linear(attention, 1)
            self.classify = nn.Linear(hidden, n_classes)

        def forward(self, instances):
            """Takes (n_instances, dim_in) and returns the logits and the attention.

            One bag at a time rather than batched: bags run from 32 to 13,298 instances, and
            padding them to a common length would spend most of the compute on padding and
            force a mask through every layer.
            """
            h = self.project(instances)
            scores = self.weigh(self.attend(h) * self.gate(h)).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            bag = weights @ h
            return self.classify(bag), weights

    return GatedAttentionMIL()


def attention_per_ageb(weights: np.ndarray, cvegeo: np.ndarray) -> dict[str, float]:
    """Aggregates the attention of a bag into one number per AGEB.

    The sum and not the mean: an AGEB holding many instances the model looked at is more
    of an explanation of the bag label than one holding a single instance it looked at
    hard. The sum is also what makes the numbers comparable to the share of the label an
    AGEB accounts for, since the weights of a bag add to one.
    """
    if len(weights) != len(cvegeo):
        raise ValueError(f"{len(weights)} weights against {len(cvegeo)} keys")
    totals: dict[str, float] = {}
    for key, weight in zip(cvegeo, weights, strict=True):
        totals[key] = totals.get(key, 0.0) + float(weight)
    return totals
