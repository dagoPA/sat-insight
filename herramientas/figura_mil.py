"""ROC curves of the MIL with the band across folds. Usage: figura_mil.py <destino> <pickle>"""

import pathlib
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

GRADE_LABELS = ["Very low", "Low", "Medium", "High", "Very high"]
REJILLA = np.linspace(0, 1, 201)
"""Common false positive rate the folds are interpolated onto.

Each fold produces its curve at its own thresholds, so averaging them point by point would
compare rates measured at different places. Interpolating onto one grid first is what makes
the band mean anything.
"""


def _banda(ax, curvas, color, label):
    """Draws the mean curve of the folds with the spread between them shaded."""
    apiladas = np.vstack(curvas)
    media = apiladas.mean(axis=0)
    ax.plot(REJILLA, media, color=color, lw=2.2, label=label)
    ax.fill_between(REJILLA, apiladas.min(axis=0), apiladas.max(axis=0), color=color, alpha=0.18)


def _interpolar(y_true, score):
    fpr, tpr, _ = roc_curve(y_true, score)
    return np.interp(REJILLA, fpr, tpr)


def dibujar(folds: list[dict], destino: str) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    figura, ejes = plt.subplots(1, 3, figsize=(19, 5.8))
    paleta = sns.color_palette("viridis", 5)

    ax = ejes[0]
    for grade in range(5):
        curvas, aucs = [], []
        for f in folds:
            y = (f["truth"] == grade).astype(int)
            if y.sum() in (0, len(y)):
                continue
            curvas.append(_interpolar(y, f["prob"][:, grade]))
            aucs.append(roc_auc_score(y, f["prob"][:, grade]))
        if curvas:
            _banda(ax, curvas, paleta[grade], f"{GRADE_LABELS[grade]} · {np.mean(aucs):.3f}")
    ax.plot([0, 1], [0, 1], color="#c0392b", ls="--", lw=1.2)
    ax.set_title("Bag level · one grade against the rest")
    ax.legend(frameon=False, loc="lower right", title="mean AUC")

    ax = ejes[1]
    for corte, color in zip((2, 3, 4), sns.color_palette("rocket", 3), strict=True):
        curvas, aucs = [], []
        for f in folds:
            y = (f["gr"] >= corte).astype(int)
            if y.sum() in (0, len(y)):
                continue
            curvas.append(_interpolar(y, f["at"]))
            aucs.append(roc_auc_score(y, f["at"]))
        if curvas:
            _banda(ax, curvas, color, f"AGEB ≥ {GRADE_LABELS[corte]} · {np.mean(aucs):.3f}")
    ax.plot([0, 1], [0, 1], color="#333", ls="--", lw=1.2, label="chance")
    ax.set_title("Attention map · does it find the deprived AGEB?")
    ax.legend(frameon=False, loc="lower right", title="mean AUC")

    for ax in ejes[:2]:
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    resumen = pd.read_csv("data/mil_kfold.csv")
    largo = resumen.melt(
        id_vars="fold",
        value_vars=["kappa", "auroc", "auroc_high", "spearman_mean"],
        var_name="metrica",
        value_name="valor",
    )
    nombres = {
        "kappa": "bag kappa",
        "auroc": "bag AUROC",
        "auroc_high": "map AUROC",
        "spearman_mean": "map Spearman",
    }
    largo["metrica"] = largo.metrica.map(nombres)
    ax = ejes[2]
    sns.barplot(
        data=largo,
        y="metrica",
        x="valor",
        hue="metrica",
        legend=False,
        errorbar=("pi", 100),
        capsize=0.25,
        err_kws={"linewidth": 1.6},
        palette=["#4c9be8", "#6bbf59", "#e8834c", "#c0392b"],
        ax=ax,
    )
    ax.axvline(0, color="#333", lw=1.2)
    ax.axvline(0.5, color="#c0392b", ls="--", lw=1.2)
    for i, nombre in enumerate(nombres.values()):
        media = largo.loc[largo.metrica == nombre, "valor"].mean()
        ax.text(
            media + (0.03 if media >= 0 else -0.03),
            i,
            f"{media:.3f}",
            va="center",
            ha="left" if media >= 0 else "right",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlim(-0.4, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Five folds · bar is the mean, whiskers the full range")

    figura.suptitle(
        "Gated attention MIL · the bag label is predicted well, the attention map is not",
        fontsize=15,
        y=0.99,
    )
    figura.tight_layout(rect=[0, 0, 1, 0.93])
    figura.savefig(destino, dpi=125, bbox_inches="tight")
    print(f"wrote {destino}")


if __name__ == "__main__":
    with pathlib.Path(sys.argv[2]).open("rb") as archivo:
        dibujar(pickle.load(archivo), sys.argv[1])
