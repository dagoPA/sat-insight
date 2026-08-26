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

from satinsight.context import build_layer

log = logging.getLogger(__name__)

HIDDEN = 256
DROPOUT = 0.25


def build(dim_in: int, n_thresholds: int = 4, hidden: int = HIDDEN, *, radius: int = 0):
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
            self.neighbourhood = build_layer(hidden) if radius else None
            width = self.neighbourhood.out_features if radius else hidden
            self.score = nn.Linear(width, n_thresholds)

        def forward(self, instances, src=None, dst=None):
            """Takes (n_instances, dim_in) and returns the bag shares and the instance ones.

            The mean is taken over probabilities and not over logits. Averaging logits and
            then squashing gives a different number, and the one the label refers to is the
            share of instances above the threshold, which is the mean of probabilities.

            With a radius the hidden representation is widened with the mean of the
            neighbouring tokens before scoring, so what the model says about a token
            depends on its surroundings. The bag stays the average of the instance
            predictions either way: the composition does not change, only how much ground
            each instance sees.
            """
            h = self.project(instances)
            if self.neighbourhood is not None:
                if src is None:
                    raise ValueError("a model built with a radius needs the bag adjacency")
                h = self.neighbourhood(h, src, dst)
            per_instance = torch.sigmoid(self.score(h))
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


def evaluate_map(model, bags, links, grades, torch, device):
    """Bag error and whether the instance scores find the deprived AGEB.

    The map is scored twice on purpose, and the two answer different questions.

    Pooling every instance of every bag measures whether the model orders AGEB across the
    whole country. Much of that is easy: knowing a municipality is deprived on average
    already ranks its AGEB above those of a comfortable one, and no disaggregation is
    involved.

    Averaging the correlation computed inside each bag measures what the project claims:
    telling apart the deprived parts of one municipality from its comfortable parts. That
    is the honest figure for the contribution, and it is the lower of the two.

    The correlation of every bag comes back under `per_bag` beside its mean. Two
    configurations are compared on the same municipalities, and a mean over a few dozen
    bags carries enough noise to swallow the differences that matter: the comparison has to
    be paired, which needs the individual numbers.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    model.eval()
    bag_true, bag_pred, scores, truths, within = [], [], [], [], []
    with torch.inference_mode():
        for bag, (src, dst) in zip(bags, links, strict=True):
            x = torch.from_numpy(bag.instances).float().to(device)
            shares, per_instance = model(x, src, dst)
            bag_true.append(bag.shares)
            bag_pred.append(shares.cpu().numpy())
            g = np.array([grades.get(c, -1) for c in bag.cvegeo])
            keep = g >= 0
            if not keep.any():
                continue
            s = instance_scores(per_instance.cpu().numpy())[keep]
            scores.extend(s)
            truths.extend(g[keep])
            # a bag whose AGEB all share one grade has no internal order to recover
            if len(set(g[keep])) > 1 and len(s) >= 20:
                within.append((bag.municipality, float(spearmanr(s, g[keep]).statistic)))
    bag_true, bag_pred = np.vstack(bag_true), np.vstack(bag_pred)
    scores, truths = np.array(scores), np.array(truths)
    return {
        "bag_mae": float(np.abs(bag_true - bag_pred).mean()),
        "auroc_high": float(roc_auc_score((truths >= 3).astype(int), scores)),
        "spearman_pooled": float(spearmanr(scores, truths).statistic),
        "spearman_within": float(np.mean([r for _, r in within])) if within else float("nan"),
        "bags_scored": len(within),
        "instances": len(truths),
        "per_bag": within,
    }
