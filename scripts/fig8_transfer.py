"""Figure 8: the Mexican model applied unchanged to Bogota and Rio de Janeiro.

One row per city, three panels: the local fine truth painted on the token lattice
(Bogota's block strata, inverted so that higher is poorer; Rio's tract median income of
the household head, as minus its logarithm), and the zero-shot score of the Mexican heads
of each feature extractor on the same tokens, averaged over the three seeds. Built tokens
only, as in the transfer protocol. The annotated correlation is Spearman over the tokens
with truth: for Bogota the whole city (one municipality); for Rio the mean over the
municipalities in the box with at least twenty such tokens.

Usage: fig8_transfer.py [output stem]
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from satinsight.context import adjacency  # noqa: E402
from satinsight.download import DATA_ROOT  # noqa: E402
from satinsight.encoders import load  # noqa: E402
from satinsight.llp import build, instance_scores  # noqa: E402
from satinsight.manuscript import BACKBONES  # noqa: E402

sys.path.insert(0, "scripts")
from fig1_design import CMAP, rgb_of, token_raster  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/manuscript/figures/fig8_transfer"
CITIES = (
    ("bogota", "Bogotá", "block strata (inverted)"),
    ("riodejaneiro", "Rio de Janeiro", "tract income (minus log)"),
)
SEEDS = (0, 1, 2)
MIN_TRUTH = 20

plt.rcParams.update({"font.size": 8, "font.family": "Helvetica Neue", "figure.dpi": 300})


def zero_shot(key: str, tag: str, tokens: pd.DataFrame) -> np.ndarray:
    import torch

    suffix = f"_{tag}" if tag else ""
    matrix, _ = load(DATA_ROOT / "transfer" / f"vectors_{key}{suffix}.npz")
    x = torch.from_numpy(matrix[tokens.row.to_numpy()]).float()
    src, dst = adjacency(tokens.y0.to_numpy(), tokens.x0.to_numpy(), radius=1)
    src, dst = torch.from_numpy(src), torch.from_numpy(dst)
    scores = []
    for seed in SEEDS:
        head = build(x.shape[1], radius=1, standardize=True)
        head.load_state_dict(
            torch.load(f"data/weights/llp_final{suffix}_s{seed}.pt", map_location="cpu")
        )
        head.eval()
        with torch.inference_mode():
            _, per_instance = head(x, src, dst)
        scores.append(instance_scores(per_instance.numpy()))
    return np.mean(scores, axis=0)


def correlation(tokens: pd.DataFrame, score: np.ndarray, single: bool) -> float:
    frame = tokens.assign(score=score).dropna(subset=["truth"])
    if single:
        return float(spearmanr(frame.score, frame.truth).statistic)
    values = [
        spearmanr(g.score, g.truth).statistic
        for _, g in frame.groupby("municipality")
        if len(g) >= MIN_TRUTH and g.truth.nunique() > 1
    ]
    return float(np.mean(values))


def paint(ax, rgb, tokens, values, title):
    finite = np.isfinite(values)
    low, high = np.nanpercentile(values[finite], [2, 98])
    norm = colors.Normalize(vmin=low, vmax=high)
    ax.imshow(rgb * 0.35)
    ax.imshow(
        token_raster(tokens[finite], values[finite], rgb.shape[:2]),
        cmap=CMAP,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=7.5)


def main() -> None:
    fig, axes = plt.subplots(
        len(CITIES), 3, figsize=(6.6, 2.4 * len(CITIES)), constrained_layout=True
    )
    for row, (key, name, truth_label) in enumerate(CITIES):
        rgb, _ = rgb_of(key)
        single = key == "bogota"
        tables = {}
        for tag, _label in BACKBONES:
            # each extraction orders its tokens its own way, so every extractor reads the
            # bag table written beside its own vectors
            suffix = f"_{tag}" if tag else ""
            table = pd.read_parquet(DATA_ROOT / "transfer" / f"bags_{key}{suffix}.parquet")
            if single:
                table = table[table.municipality == "11001"].reset_index(drop=True)
            tables[tag] = table
        first = tables[BACKBONES[0][0]]
        truth = first.truth.to_numpy(dtype=float)
        paint(axes[row, 0], rgb, first, truth, f"{name}\ntruth: {truth_label}")
        for column, (tag, label) in enumerate(BACKBONES, start=1):
            tokens = tables[tag]
            score = zero_shot(key, tag, tokens)
            rho = correlation(tokens, score, single)
            paint(
                axes[row, column],
                rgb,
                tokens,
                score,
                f"{label}, Mexican model unadapted\n$\\rho$ = {rho:.2f}",
            )
    fig.text(
        0.5,
        -0.01,
        "colors: each panel on its own 2nd to 98th percentile scale, blue low to red high",
        ha="center",
        fontsize=6.5,
    )
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", bbox_inches="tight")
    print(f"{OUT} saved", flush=True)


if __name__ == "__main__":
    main()
