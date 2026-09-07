"""Trains in Colombia and Brazil on their own municipal aggregates and scores the map.

The Mexican result says that municipal aggregates recover most of what tract labels would
teach. Whether that is a property of the method or of Mexico is answered by repeating the
design where the aggregate and the fine ground are different variables from different
offices: Colombia's municipal poverty rate against Bogota's strata by block, Brazil's
municipal mean income of the household head against the tract median of the same variable.

Every bag is a municipality inside one of the composited boxes and every instance a built
token. Three rows per country, all under the same grouped folds by municipality, so every
token is scored by a model that never saw its municipality's label:

- zero-shot: the three Mexican heads, unadapted, the number the paper already reports;
- aggregates: a one-output label-proportion head trained on the municipal figure alone;
- oracle: the same head trained on the token truth, the fully supervised upper bound;
- mexico-init: the aggregate head started from the Mexican weights and fine-tuned on the
  local aggregates, the transfer a practitioner with both would run.

Colombia's truth exists in Bogota alone, so holding Bogota out would leave the oracle
nothing to learn from; its folds are 3.2 km spatial cells inside the city instead, trained
on the other cells. Tokens of neighbouring municipalities that snap to Bogota blocks across
the boundary are not truth and are dropped.

The label is a share in [0, 1]. Colombia's poverty rate is one already; Brazil's mean
income is turned into a deprivation share by scaling minus its log to [0, 1] over the
country's bags, a monotone map that adds no spatial information. Truth for the oracle is
scaled the same way. Bags are few, forty to fifty per country, the fifty-bag point of the
Mexican curve, so the honest expectation is that end of the curve.

Usage: transfer_train.py [epochs] [folds]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from satinsight.context import adjacency  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.encoders import load  # noqa: E402
from satinsight.llp import build, instance_scores  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEEDS = (0, 1, 2)
RADIUS = 1
MIN_TOKENS = 32
MIN_TRUTH = 20
CELL_PX = 320
"""Side of the spatial cells the Colombian oracle folds over: 3.2 km on the 10 m grid."""
BOGOTA = "11001"
SIZES = (50, 100, 200, 400, None)
"""Bag counts of the transfer curve; None is the whole pool of the country."""


def country_keys() -> dict[str, list[str]]:
    """Every box with bags on disk, grouped by country through the catalogue."""
    sys.path.insert(0, "scripts")
    from transfer_bags import COUNTRY

    out: dict[str, list[str]] = {"colombia": [], "brazil": []}
    for key, country in COUNTRY.items():
        if (DATA_ROOT / "transfer" / f"bags_{key}.parquet").exists():
            out[country].append(key)
    return out


COUNTRIES = country_keys()
LABEL = {"colombia": "poor_share", "brazil": "mean_income"}
OUT = "data/transfer_training.csv"
log = logging.getLogger("transfer")


def device_of(torch):
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def scaled_share(values: pd.Series) -> pd.Series:
    """Minus log income, scaled to [0, 1] over the values present: a deprivation share."""
    d = -np.log(values.astype(float))
    return (d - d.min()) / max(d.max() - d.min(), 1e-9)


def country_tokens(country: str) -> tuple[pd.DataFrame, np.ndarray]:
    parts, vectors = [], []
    for key in COUNTRIES[country]:
        table = pd.read_parquet(DATA_ROOT / "transfer" / f"bags_{key}.parquet")
        matrix, _ = load(DATA_ROOT / "transfer" / f"vectors_{key}.npz")
        table["key"] = key
        parts.append(table)
        vectors.append(matrix[table.row.to_numpy()])
    tokens = pd.concat(parts, ignore_index=True)
    matrix = np.vstack(vectors)
    # a municipality covered by a hand metro box and by its own seat box would enter twice
    # on two grids; the box holding more of its tokens keeps it
    counts = tokens.groupby(["municipality", "key"]).size().reset_index(name="n")
    winner = counts.sort_values("n", ascending=False).drop_duplicates("municipality")
    keep = tokens.set_index(["municipality", "key"]).index.isin(
        winner.set_index(["municipality", "key"]).index
    )
    tokens, matrix = tokens[keep].reset_index(drop=True), matrix[keep]
    label = LABEL[country]
    if country == "brazil":
        per_bag = tokens.groupby("municipality")[label].first()
        share = scaled_share(per_bag)
        tokens["target"] = tokens.municipality.map(share).to_numpy()
    else:
        tokens["target"] = tokens[label].to_numpy()
        tokens.loc[tokens.municipality != BOGOTA, "truth"] = np.nan
    counts = tokens.groupby("municipality").size()
    usable = counts[counts >= MIN_TOKENS].index
    keep = tokens.municipality.isin(usable) & tokens.target.notna()
    tokens, matrix = tokens[keep].reset_index(drop=True), matrix[keep.to_numpy()]
    log.info(
        "%s: %d built tokens in %d bags · %d tokens with truth",
        country,
        len(tokens),
        tokens.municipality.nunique(),
        int(tokens.truth.notna().sum()),
    )
    return tokens, matrix


class Bags:
    """Tokens of one country grouped by municipality, with adjacency and tensors ready."""

    def __init__(self, tokens, matrix, torch, device):
        self.torch, self.device = torch, device
        self.names = sorted(tokens.municipality.unique())
        self.rows, self.links, self.targets, self.truths, self.cells = {}, {}, {}, {}, {}
        self.x = torch.from_numpy(matrix).float().to(device)
        for name in self.names:
            rows = np.flatnonzero(tokens.municipality.to_numpy() == name)
            self.rows[name] = rows
            src, dst = adjacency(tokens.y0.to_numpy()[rows], tokens.x0.to_numpy()[rows], RADIUS)
            self.links[name] = (
                torch.from_numpy(src).to(device),
                torch.from_numpy(dst).to(device),
            )
            self.targets[name] = float(tokens.target.to_numpy()[rows[0]])
            self.truths[name] = tokens.truth.to_numpy()[rows]
            self.cells[name] = np.array(
                [
                    f"{k}:{y // CELL_PX}:{x // CELL_PX}"
                    for k, y, x in zip(
                        tokens.key.to_numpy()[rows],
                        tokens.y0.to_numpy()[rows],
                        tokens.x0.to_numpy()[rows],
                        strict=True,
                    )
                ]
            )

    def features(self, name):
        return self.x[self.torch.from_numpy(self.rows[name]).to(self.device)]


def folds_of(names: list[str], seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    order = list(names)
    rng.shuffle(order)
    return [order[i::FOLDS] for i in range(FOLDS)]


def mexican_start(model, seed: int, torch, device) -> None:
    """Loads the Mexican projection and scaler into a one-output head.

    The Mexican head scores four cumulative thresholds; the local label is one share, so
    the new scoring row starts as the mean of the four and learns from there. The scaler
    stays Mexican so the projection sees inputs on the scale it was trained on.
    """
    state = torch.load(f"data/weights/llp_final_s{seed}.pt", map_location=device)
    own = model.state_dict()
    for key, value in state.items():
        if key.startswith("score."):
            own[key] = value.mean(dim=0, keepdim=True)
        else:
            own[key] = value
    model.load_state_dict(own)


def fit(
    bags: Bags,
    train: list[str],
    seed: int,
    supervised: bool,
    truth_scale,
    exclude=None,
    init: str = "scratch",
):
    """One head fit on the training bags, on bag targets or on token truth.

    `exclude` masks tokens per bag that the supervised fit must not see, which is how the
    spatial folds of the Colombian oracle hold cells out inside a training bag.
    """
    torch = bags.torch
    from torch import nn

    torch.manual_seed(seed)
    dim = bags.x.shape[1]
    model = build(dim, n_thresholds=1, radius=RADIUS, standardize=True).to(bags.device)
    rng = np.random.default_rng(seed)
    if init == "mexico":
        mexican_start(model, seed % len(SEEDS), torch, bags.device)
        learning_rate = 5e-5
    else:
        sample = np.vstack(
            [
                bags.x[bags.rows[n][rng.permutation(len(bags.rows[n]))[:200]]].cpu().numpy()
                for n in train
            ]
        )
        model.fit_scaler(sample.mean(axis=0), sample.std(axis=0))
        learning_rate = 2e-4
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCELoss()
    for _ in range(EPOCHS):
        model.train()
        for name in rng.permutation(train):
            x = bags.features(name)
            src, dst = bags.links[name]
            optimiser.zero_grad()
            shares, per_instance = model(x, src, dst)
            if supervised:
                truth = bags.truths[name]
                keep = np.isfinite(truth)
                if exclude is not None:
                    keep &= ~exclude[name]
                if not keep.any():
                    continue
                target = torch.from_numpy(truth_scale(truth[keep])).float().to(bags.device)
                loss = criterion(per_instance[torch.from_numpy(keep).to(bags.device), 0], target)
            else:
                target = torch.tensor([bags.targets[name]], device=bags.device)
                loss = criterion(shares, target)
            loss.backward()
            optimiser.step()
    return model


def score(model, bags: Bags, names: list[str]) -> dict[str, np.ndarray]:
    torch = bags.torch
    model.eval()
    out = {}
    with torch.inference_mode():
        for name in names:
            src, dst = bags.links[name]
            _, per_instance = model(bags.features(name), src, dst)
            out[name] = instance_scores(per_instance.cpu().numpy())
    return out


def zero_shot(bags: Bags) -> dict[str, np.ndarray]:
    torch = bags.torch
    scores = {name: [] for name in bags.names}
    for seed in SEEDS:
        model = build(bags.x.shape[1], radius=RADIUS, standardize=True).to(bags.device)
        state = torch.load(f"data/weights/llp_final_s{seed}.pt", map_location=bags.device)
        model.load_state_dict(state)
        for name, s in score(model, bags, bags.names).items():
            scores[name].append(s)
    return {name: np.mean(s, axis=0) for name, s in scores.items()}


def evaluate(bags: Bags, scores: dict[str, np.ndarray]) -> dict:
    """Within-municipality Spearman against the truth, pooled Spearman, and top-quartile AUROC."""
    within, pooled_s, pooled_t = [], [], []
    for name in bags.names:
        truth, s = bags.truths[name], scores[name]
        keep = np.isfinite(truth)
        if keep.sum() < MIN_TRUTH or len(np.unique(truth[keep])) < 2:
            continue
        within.append((name, float(spearmanr(s[keep], truth[keep]).statistic)))
        pooled_s.extend(s[keep])
        pooled_t.extend(truth[keep])
    pooled_s, pooled_t = np.array(pooled_s), np.array(pooled_t)
    high = (pooled_t >= np.quantile(pooled_t, 0.75)).astype(int)
    rng = np.random.default_rng(0)
    rhos = np.array([r for _, r in within])
    boot = (
        [rng.choice(rhos, len(rhos)).mean() for _ in range(2000)]
        if len(rhos) > 1
        else [rhos.mean()]
    )
    return {
        "within": float(rhos.mean()),
        "within_ci_low": float(np.percentile(boot, 2.5)),
        "within_ci_high": float(np.percentile(boot, 97.5)),
        "municipalities_scored": len(within),
        "pooled": float(spearmanr(pooled_s, pooled_t).statistic),
        "auroc_top_quartile": float(roc_auc_score(high, pooled_s)),
        "tokens_scored": len(pooled_t),
        "per_bag": within,
    }


def cross_validated(
    bags: Bags,
    seed: int,
    supervised: bool,
    truth_scale,
    init: str = "scratch",
    size: int | None = None,
) -> dict[str, np.ndarray]:
    """Scores every bag under grouped folds; `size` caps the training bags per fold.

    The cap draws a nested random subset of the training bags of each fold, the same
    prefix for every size of one seed, which is what turns the folds into a curve over
    the supply of aggregates like the Mexican one.
    """
    scores = {}
    order = list(bags.names)
    np.random.default_rng(seed).shuffle(order)
    for held in folds_of(bags.names, seed):
        train = [n for n in order if n not in held]
        if size is not None:
            train = train[:size]
        model = fit(bags, train, seed, supervised, truth_scale, init=init)
        scores.update(score(model, bags, held))
    return scores


def cross_validated_cells(bags: Bags, seed: int, truth_scale) -> dict[str, np.ndarray]:
    """Oracle under spatial folds: cells with truth are held out, the rest trains."""
    with_truth = sorted(
        {c for name in bags.names for c in bags.cells[name][np.isfinite(bags.truths[name])]}
    )
    scores = {name: np.full(len(bags.rows[name]), np.nan) for name in bags.names}
    for held in folds_of(with_truth, seed):
        masks = {name: np.isin(bags.cells[name], held) for name in bags.names}
        model = fit(bags, bags.names, seed, True, truth_scale, exclude=masks)
        for name, s in score(model, bags, bags.names).items():
            scores[name][masks[name]] = s[masks[name]]
    return scores


def main() -> None:
    import torch

    device = device_of(torch)
    rows, per_bag_rows = [], []
    for country in COUNTRIES:
        tokens, matrix = country_tokens(country)
        bags = Bags(tokens, matrix, torch, device)
        all_truth = np.concatenate([t[np.isfinite(t)] for t in bags.truths.values()])
        low, high = all_truth.min(), all_truth.max()

        def truth_scale(values, low=low, high=high):
            return (values - low) / max(high - low, 1e-9)

        methods = {
            "zero-shot": lambda seed, bags=bags: zero_shot(bags),
            "aggregates": lambda seed, bags=bags, scale=truth_scale: cross_validated(
                bags, seed, False, scale
            ),
            **{
                f"aggregates@{size}": (
                    lambda seed, bags=bags, scale=truth_scale, size=size: cross_validated(
                        bags, seed, False, scale, size=size
                    )
                )
                for size in SIZES
                if size is not None and size < len(bags.names)
            },
            "mexico-init": lambda seed, bags=bags, scale=truth_scale: cross_validated(
                bags, seed, False, scale, init="mexico"
            ),
            "oracle": lambda seed, bags=bags, scale=truth_scale, country=country: (
                cross_validated_cells(bags, seed, scale)
                if country == "colombia"
                else cross_validated(bags, seed, True, scale)
            ),
        }
        for method, run in methods.items():
            for seed in SEEDS if method != "zero-shot" else (0,):
                result = evaluate(bags, run(seed))
                per_bag = result.pop("per_bag")
                rows.append(
                    {
                        "country": country,
                        "method": method,
                        "seed": seed,
                        "bags": len(bags.names),
                        **result,
                    }
                )
                per_bag_rows.extend(
                    {
                        "country": country,
                        "method": method,
                        "seed": seed,
                        "municipality": n,
                        "rho": r,
                    }
                    for n, r in per_bag
                )
                log.info(
                    "%s · %s · seed %d · within %+.3f [%+.3f, %+.3f] over %d municipalities · "
                    "pooled %+.3f · AUROC %.3f",
                    country,
                    method,
                    seed,
                    result["within"],
                    result["within_ci_low"],
                    result["within_ci_high"],
                    result["municipalities_scored"],
                    result["pooled"],
                    result["auroc_top_quartile"],
                )
                pd.DataFrame(rows).to_csv(OUT, index=False)
                pd.DataFrame(per_bag_rows).to_csv(OUT.replace(".csv", "_bags.csv"), index=False)
    summary = (
        pd.DataFrame(rows)
        .groupby(["country", "method"])[["within", "pooled", "auroc_top_quartile"]]
        .mean()
    )
    print(
        "\n===== TRANSFER TRAINING "
        "(within-municipality Spearman vs fine truth, grouped folds) ====="
    )
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
