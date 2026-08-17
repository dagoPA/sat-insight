"""Descarga una muestra mínima sobre el AOI piloto y renderiza los paneles de la figura.

Produce cuatro PNG que ilustran los tipos de dato del proyecto:
  1. Sentinel-2 en fecha única, con nubes
  2. Sentinel-2 en fecha única, despejada
  3. Sentinel-2 compuesto mediana 2020 con enmascarado SCL
  4. Sentinel-1 RTC compuesto mediana 2020, falso color VV/VH/ratio
"""

import json
import warnings
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore")

BBOX = [-93.135, 16.740, -93.095, 16.768]  # Tuxtla Gutiérrez, Chiapas
YEAR = "2020-01-01/2020-12-31"
OUT = Path("docs/figs")
OUT.mkdir(parents=True, exist_ok=True)

N_S2_COMPOSITE = 36
N_S1_COMPOSITE = 24
SIZE = (440, 308)  # ancho, alto de salida

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)


def read_window(href, bbox, out_shape=None):
    """Lee solo la ventana del AOI desde el COG remoto."""
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(left, bottom, right, top, src.transform)
        shape = out_shape or (int(win.height), int(win.width))
        return src.read(1, window=win, out_shape=shape, boundless=True, fill_value=0)


def stretch(a, lo=2, hi=98):
    """Estiramiento por percentiles a 0-255."""
    a = np.asarray(a, dtype="float32")
    valid = a[np.isfinite(a) & (a != 0)]
    if valid.size == 0:
        return np.zeros(a.shape, dtype="uint8")
    p1, p2 = np.percentile(valid, [lo, hi])
    if p2 <= p1:
        p2 = p1 + 1e-6
    return np.clip((a - p1) / (p2 - p1) * 255, 0, 255).astype("uint8")


def save_rgb(r, g, b, name):
    rgb = np.dstack([r, g, b])
    Image.fromarray(rgb).resize(SIZE, Image.LANCZOS).save(OUT / name, optimize=True)
    print(f"  escrito {name}")


# ----------------------------------------------------------------- Sentinel-2
s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=YEAR).items())
s2.sort(key=lambda i: i.properties["eo:cloud_cover"])

stats = {
    "aoi_bbox": BBOX,
    "aoi_nombre": "Tuxtla Gutiérrez, Chiapas",
    "s2_escenas_2020": len(s2),
    "s2_pct_mayor_50_nubes": round(
        100 * sum(1 for i in s2 if i.properties["eo:cloud_cover"] > 50) / len(s2)
    ),
    "s2_pct_mayor_80_nubes": round(
        100 * sum(1 for i in s2 if i.properties["eo:cloud_cover"] > 80) / len(s2)
    ),
}

clear, cloudy = s2[0], s2[-1]
H, W = None, None

print("Sentinel-2 fecha despejada...")
bands = [read_window(clear.assets[b].href, BBOX) for b in ("B04", "B03", "B02")]
H, W = bands[0].shape
save_rgb(*[stretch(b) for b in bands], "s2_despejada.png")
stats["s2_fecha_despejada"] = clear.datetime.strftime("%Y-%m-%d")
stats["s2_nubes_despejada"] = round(clear.properties["eo:cloud_cover"], 1)

print("Sentinel-2 fecha con nubes...")
bands = [read_window(cloudy.assets[b].href, BBOX, (H, W)) for b in ("B04", "B03", "B02")]
save_rgb(*[stretch(b) for b in bands], "s2_nublada.png")
stats["s2_fecha_nublada"] = cloudy.datetime.strftime("%Y-%m-%d")
stats["s2_nubes_nublada"] = round(cloudy.properties["eo:cloud_cover"], 1)

print(f"Sentinel-2 compuesto mediana ({N_S2_COMPOSITE} escenas)...")
# SCL: 4=vegetación, 5=suelo desnudo, 6=agua, 7=no clasificado -> píxel válido
VALID_SCL = {4, 5, 6, 7}
stacks = {b: [] for b in ("B04", "B03", "B02")}
usadas = 0
for item in s2[:N_S2_COMPOSITE]:
    try:
        scl = read_window(item.assets["SCL"].href, BBOX, (H, W))
        mask = np.isin(scl, list(VALID_SCL))
        if mask.mean() < 0.05:
            continue
        for b in stacks:
            arr = read_window(item.assets[b].href, BBOX, (H, W)).astype("float32")
            arr[~mask] = np.nan
            stacks[b].append(arr)
        usadas += 1
    except Exception as exc:
        print(f"    saltada {item.id[:28]}: {type(exc).__name__}")

med = {b: np.nanmedian(np.dstack(v), axis=2) for b, v in stacks.items()}
save_rgb(*[stretch(med[b]) for b in ("B04", "B03", "B02")], "s2_compuesto.png")
stats["s2_escenas_en_compuesto"] = usadas

# ----------------------------------------------------------------- Sentinel-1
s1 = list(cat.search(collections=["sentinel-1-rtc"], bbox=BBOX, datetime=YEAR).items())
stats["s1_escenas_2020"] = len(s1)

# Una sola geometría de adquisición para que el compuesto sea coherente
orbits = {}
for it in s1:
    key = (it.properties.get("sat:orbit_state"), it.properties.get("sat:relative_orbit"))
    orbits.setdefault(key, []).append(it)
key = max(orbits, key=lambda k: len(orbits[k]))
sel = orbits[key][:N_S1_COMPOSITE]
stats["s1_orbita"] = f"{key[0]} · relativa {key[1]}"
stats["s1_escenas_en_compuesto"] = len(sel)

print(f"Sentinel-1 RTC compuesto mediana ({len(sel)} escenas, órbita {key})...")
vv_s, vh_s = [], []
for item in sel:
    try:
        vv_s.append(read_window(item.assets["vv"].href, BBOX, (H, W)).astype("float32"))
        vh_s.append(read_window(item.assets["vh"].href, BBOX, (H, W)).astype("float32"))
    except Exception as exc:
        print(f"    saltada {item.id[:28]}: {type(exc).__name__}")

vv = np.nanmedian(np.dstack(vv_s), axis=2)
vh = np.nanmedian(np.dstack(vh_s), axis=2)
with np.errstate(divide="ignore", invalid="ignore"):
    vv_db = 10 * np.log10(np.where(vv > 0, vv, np.nan))
    vh_db = 10 * np.log10(np.where(vh > 0, vh, np.nan))
    ratio = vv_db - vh_db

save_rgb(stretch(vv_db), stretch(vh_db), stretch(ratio), "s1_compuesto.png")

# Rango real de retrodispersión, útil para la leyenda
stats["s1_vv_db_p5_p95"] = [round(float(np.nanpercentile(vv_db, 5)), 1),
                            round(float(np.nanpercentile(vv_db, 95)), 1)]
stats["s1_vh_db_p5_p95"] = [round(float(np.nanpercentile(vh_db, 5)), 1),
                            round(float(np.nanpercentile(vh_db, 95)), 1)]
stats["resolucion_px_m"] = 10
stats["aoi_px"] = [int(W), int(H)]

(OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print("\n" + json.dumps(stats, indent=2, ensure_ascii=False))
