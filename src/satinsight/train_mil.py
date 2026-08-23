"""Training loop of the attention MIL and the validation of its heat map.

The model never sees which instance explains the bag label. What the heat map is worth is
settled afterwards, by aggregating the attention per AGEB and correlating it against the
grade of that AGEB, which was held out of training entirely.

That correlation is the gate of stage three. If the attention does not line up with the
AGEB grade, the interpretability premise of the project falls, whatever the bag-level
accuracy says.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from satinsight.bagdata import Bag
from satinsight.mil import attention_per_ageb, build

log = logging.getLogger(__name__)

EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 0

PATIENCE = 6
"""Epochs without improving the validation kappa before stopping.

With 110 training bags an epoch is cheap and overfitting is fast: the model can learn to
recognise which municipality it is looking at long before it learns what deprivation looks
like."""


def _device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bag_weights(bags: list[Bag], n_classes: int) -> np.ndarray:
    """Class weights inversely proportional to how often each grade appears.

    Rounding the municipal aggregate to five classes leaves four bags at the top grade
    against a hundred and seventy at the second. Without reweighting, predicting the
    majority everywhere is a local optimum the model reaches in two epochs and never
    leaves.
    """
    counts = np.bincount([b.ordinal for b in bags], minlength=n_classes).astype("float64")
    return np.where(counts > 0, len(bags) / (n_classes * np.maximum(counts, 1)), 0.0)


def train(
    train_bags: list[Bag],
    val_bags: list[Bag],
    *,
    n_classes: int = 5,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    patience: int = PATIENCE,
    seed: int = SEED,
):
    """Trains one bag at a time and keeps the state that validated best.

    Returns the model and the history. Bags are shuffled every epoch because they arrive
    grouped by city, and a model that sees a whole city in a row drifts towards it.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = _device(torch)
    model = build(train_bags[0].instances.shape[1], n_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    weights = torch.tensor(_bag_weights(train_bags, n_classes), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    rng = np.random.default_rng(seed)

    def as_tensor(bag: Bag):
        return torch.from_numpy(bag.instances).float().to(device)

    best_kappa, best_state, waited = -np.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for index in rng.permutation(len(train_bags)):
            bag = train_bags[index]
            optimiser.zero_grad()
            logits, _ = model(as_tensor(bag))
            target = torch.tensor([bag.ordinal], device=device)
            loss = criterion(logits.unsqueeze(0), target)
            loss.backward()
            optimiser.step()
            total += float(loss)

        scored = predict(model, val_bags, device=device)
        kappa = float(cohen_kappa_score(scored["truth"], scored["prediction"], weights="quadratic"))
        history.append({"epoch": epoch, "loss": total / len(train_bags), "val_kappa": kappa})
        log.info("epoch %d · loss %.4f · val kappa %.3f", epoch, total / len(train_bags), kappa)

        if kappa > best_kappa:
            best_kappa, waited = kappa, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                log.info("stopped at epoch %d, best val kappa %.3f", epoch, best_kappa)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict(model, bags: list[Bag], *, device: str | None = None) -> dict[str, np.ndarray]:
    """Bag-level predictions, with the attention of every instance kept alongside."""
    import torch

    device = device or _device(torch)
    model.eval()
    predictions, truths, probabilities, attentions = [], [], [], []
    with torch.inference_mode():
        for bag in bags:
            logits, weights = model(torch.from_numpy(bag.instances).float().to(device))
            probabilities.append(torch.softmax(logits, dim=0).cpu().numpy())
            predictions.append(int(logits.argmax()))
            truths.append(bag.ordinal)
            attentions.append(weights.cpu().numpy())
    return {
        "prediction": np.array(predictions),
        "truth": np.array(truths),
        "probability": np.vstack(probabilities),
        "attention": attentions,
    }


def score_heatmap(bags: list[Bag], attentions: list[np.ndarray], grades: dict[str, int]) -> dict:
    """Correlates the attention aggregated per AGEB against the grade of that AGEB.

    This is the measurement the project exists for. The attention of a bag adds to one, so
    the numbers are comparable between municipalities of very different size only after
    ranking within each bag; Spearman is computed per bag and averaged, and also pooled
    over every AGEB for a single figure.

    Bags whose AGEB all share one grade contribute nothing and are skipped: there is no
    ordering to recover inside them.
    """
    per_bag, pooled_attention, pooled_grade = [], [], []
    for bag, weights in zip(bags, attentions, strict=True):
        totals = attention_per_ageb(weights, bag.cvegeo)
        keys = [k for k in totals if k in grades]
        if len(keys) < 3:
            continue
        values = np.array([totals[k] for k in keys])
        truth = np.array([grades[k] for k in keys])
        pooled_attention.extend(values / values.sum())
        pooled_grade.extend(truth)
        if len(set(truth)) < 2:
            continue
        per_bag.append(
            {
                "city": bag.city,
                "municipality": bag.municipality,
                "agebs": len(keys),
                "spearman": float(spearmanr(values, truth).statistic),
            }
        )

    detail = pd.DataFrame(per_bag)
    grade_array = np.array(pooled_grade)
    attention_array = np.array(pooled_attention)
    high = (grade_array >= 3).astype(int)
    return {
        "detail": detail,
        "spearman_mean": float(detail.spearman.mean()) if len(detail) else np.nan,
        "spearman_pooled": float(spearmanr(attention_array, grade_array).statistic),
        "auroc_high": float(roc_auc_score(high, attention_array)) if high.any() else np.nan,
        "bags_scored": len(detail),
        "agebs_scored": len(grade_array),
    }
