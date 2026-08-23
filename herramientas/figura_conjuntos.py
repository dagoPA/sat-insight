"""Bar chart of the feature sets across modalities. Usage: figura_conjuntos.py <destino>"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SETS = ["cobertura", "densidad", "textura", "completo"]
ETIQUETAS = {
    "cobertura": "land cover",
    "densidad": "intensity",
    "textura": "texture",
    "completo": "all combined",
}
MODALIDADES = {"óptico": "optical", "radar": "radar", "fusión": "fused"}
COLOR = {"óptico": "#4c9be8", "radar": "#e8834c", "fusión": "#6bbf59"}
METRICAS = [
    ("kappa", "Quadratic weighted kappa", 0.0, 0.68),
    ("auroc_macro", "AUROC, macro over the five grades", 0.4, 0.95),
    ("auroc_ge_3", "AUROC of the «grade ≥ High» threshold", 0.4, 0.98),
]


def dibujar(resultados: pd.DataFrame, destino: str) -> None:
    figura, ejes = plt.subplots(1, 3, figsize=(18, 5.4))
    ancho = 0.26
    for eje, (metrica, titulo, piso, techo) in zip(ejes, METRICAS, strict=True):
        x = np.arange(len(SETS))
        for i, (modalidad, etiqueta) in enumerate(MODALIDADES.items()):
            sub = resultados[resultados.modalidad == modalidad].set_index("conjunto")
            sub = sub.loc[SETS]
            valor = sub[metrica].to_numpy()
            eje.bar(
                x + (i - 1) * ancho,
                valor,
                ancho,
                label=etiqueta,
                color=COLOR[modalidad],
                yerr=[valor - sub[f"{metrica}_ic_bajo"], sub[f"{metrica}_ic_alto"] - valor],
                capsize=3,
                ecolor="#333",
                error_kw={"lw": 1.1},
            )
            for xi, vi in zip(x + (i - 1) * ancho, valor, strict=True):
                eje.text(xi, vi + 0.008, f"{vi:.3f}", ha="center", fontsize=7.4)
        eje.set_xticks(x)
        eje.set_xticklabels([ETIQUETAS[c] for c in SETS], fontsize=9)
        eje.set_title(titulo, fontsize=11.5)
        eje.grid(axis="y", alpha=0.25)
        eje.set_axisbelow(True)
        eje.set_ylim(piso, techo)
        if piso:
            eje.axhline(0.5, color="#c0392b", ls="--", lw=1.1)
        eje.legend(frameon=False, fontsize=9, ncols=3)

    cities = int(resultados.ciudades_mide.iloc[0])
    agebs = int(resultados.n_mide.iloc[0])
    figura.suptitle(
        f"Validation on {cities} held-out cities · {agebs:,} AGEB · "
        "intervals from 400 city-level bootstrap replicates",
        fontsize=12.5,
    )
    figura.tight_layout(rect=[0, 0, 1, 0.93])
    figura.savefig(destino, dpi=120, bbox_inches="tight")
    print(f"wrote {destino}")


if __name__ == "__main__":
    dibujar(pd.read_csv("data/baseline_val.csv"), sys.argv[1])
