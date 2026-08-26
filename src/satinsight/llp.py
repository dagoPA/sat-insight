"""Learning from label proportions: predict every instance, then average.

The attention MIL pools the instance features and predicts from the pooled vector. That
lets the bag label be solved from the plain mean, and the sweeps confirmed it: flattening
the attention until it was nearly uniform cost no accuracy at all. With the prediction
independent of where the model looks, nothing ever teaches it to look anywhere.

This turns the composition around. Each instance gets its own predicted share of belonging
to a deprived AGEB, and the bag prediction is the average of those instance predictions.
The bag label constrains that average directly, so every instance has to commit to a
number, and the only way the average can be right across bags of very different
composition is for the individual numbers to be right too.

The instance predictions are the map. There is no attention to interpret and no pooled
vector in between: what the model says about a token is what gets plotted.

The setting has a name of its own. Bags labelled by the proportion of positives they hold
are learning from label proportions, studied apart from multiple instance learning, and
the census labels of this project are proportions in the literal sense: the share of a
municipality's population living in AGEB at grade k or above.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

HIDDEN = 256
DROPOUT = 0.25


def build(dim_in: int, n_thresholds: int = 4, hidden: int = HIDDEN):
    """Builds the per-instance predictor. Torch is imported here, not at module level."""
    import torch
    from torch import nn

    class ProportionNet(nn.Module):
        """One prediction per instance; the bag is their mean.

        The head is deliberately small. It sits on frozen foundation model vectors that
        already carry what a 160 m token looks like, and the supervision that reaches it is
        one number per bag: a few hundred over the whole training set.
        """

        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Dropout(DROPOUT))
            self.score = nn.Linear(hidden, n_thresholds)

        def forward(self, instances):
            """Takes (n_instances, dim_in) and returns the bag shares and the instance ones.

            The mean is taken over probabilities and not over logits. Averaging logits and
            then squashing gives a different number, and the one the label refers to is the
            share of instances above the threshold, which is the mean of probabilities.
            """
            per_instance = torch.sigmoid(self.score(self.project(instances)))
            return per_instance.mean(dim=0), per_instance

    return ProportionNet()


def instance_scores(per_instance: np.ndarray) -> np.ndarray:
    """Collapses the per-threshold predictions of an instance into one ordinal score.

    Summing the four thresholds recovers the expected grade: an instance the model puts
    above every threshold scores four, one it puts below all of them scores zero. It is the
    same decoding the cumulative parameterisation implies at bag level, applied one step
    down.
    """
    return per_instance.sum(axis=1)
