"""Texture features per AGEB from the gray level co-occurrence matrix.

The GLCM is defined over rectangular windows and an AGEB is an irregular polygon. The way
out is to cut the AGEB to its bounding window, mark whatever falls outside the polygon as
invalid, and drop from the count any pair one of whose pixels is invalid. That is achieved
by reserving level zero for the invalid and then removing its row and its column from the
matrix.

Quantisation is never per AGEB: normalising each polygon's brightness on its own would
erase the level signal that separates a precarious settlement from a consolidated
neighbourhood. For optical the scale is estimated from the whole city, because there are
atmospheric and BRDF residuals there that are not signal. For radar a fixed range in
decibels is used, because gamma0 is calibrated and estimating it per city would throw away
the cross-country comparability that justified choosing that sensor.

The absolute level is never lost: first order features are computed over the raw band, in
physical units, and the GLCM describes the spatial arrangement alone.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from shapely.ops import clip_by_rect
from skimage.feature import graycomatrix, graycoprops

from satinsight.grid import polygon_window

log = logging.getLogger(__name__)

LEVELS = 8
"""Gray levels of the quantisation, and the most delicate decision in the module.

The matrix has `levels²` cells and an AGEB contributes on the order of its pixels in
pairs. The 2,703 pilot AGEB have a median of 2,396 pixels at 10 m, so:

    32 levels = 1024 cells ->  2.3 pairs per cell
    16 levels =  256 cells ->  9.4 pairs per cell
     8 levels =   64 cells -> 37.4 pairs per cell

At two pairs per cell, entropy and energy measure sampling noise. What makes it serious is
that the bias is monotone in the number of pixels: entropy is underestimated and energy
overestimated the emptier the matrix gets. The size of an AGEB correlates with urban
density, and density with deprivation, so an undersampled feature points at the target
without carrying information about it. And since the typical size changes between cities,
every validation fold would see a different noise structure.

Eight levels lose texture detail and buy statistics that mean something. The choice is
checked with `split_half_reliability`, which measures whether a feature reproduces when the
same polygon is cut in two.
"""

DISTANCES = (1, 2, 4)
"""Separations in pixels. At 10 m they span from the scale of a roof to that of a block.

Distances are reported separately rather than averaged: they are different scales and
carry different information. In narrow AGEB nearly every pair at 4 pixels crosses the
border and is dropped, so averaging that distance with the one at 1 dilutes the good signal
with the noisy one. Keeping them apart lets the model weigh each scale on its own.
"""

ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
"""The four canonical orientations. These are averaged: rotation invariance is desirable,
because the urban fabric has no privileged orientation of interest."""

PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")

MIN_PIXELS = 640
"""Minimum pixels to compute texture, chosen together with `LEVELS`.

With 64 cells, 640 pixels give ten pairs per cell, which is the usual rule of thumb for
Haralick statistics to be stable. It leaves out 11% of the pilot AGEB.

Those AGEB stay in the set with texture null and first order features intact, and the
exclusion is declared when evaluating. Dropping them silently would bias the sample
precisely towards the large AGEB.
"""

FIXED_RANGES_S1 = {
    "s1vv": (-25.0, 5.0),
    "s1vh": (-30.0, 0.0),
    "s1ratio": (0.0, 15.0),
}
"""Fixed quantisation ranges for radar, in decibels.

Deriving the range from each city's data throws away the property that justified choosing
Sentinel-1: gamma0 is a calibrated quantity, comparable across countries without
recalibration. If quantisation adapts to each city, that comparability is lost to an
implementation decision, and with it the transfer argument for Brazil and Colombia.

The edges are set by physics and come out identical in every city. Optical admits
per-scene normalisation, because it carries atmospheric and BRDF residuals foreign to the
signal.
"""

FIXED_RANGES_S2 = {
    "s2red": (0.0, 4000.0),
    "s2nir": (0.0, 5000.0),
    "s2ndvi": (-0.5, 1.0),
    "s2ndbi": (-0.6, 0.6),
}
"""Fixed optical ranges, in reflectance scaled by ten thousand and in index units.

They exist for the mirror ablation of the radar one. Sentinel-2 L2A reflectance is
calibrated too, so the question of whether fixing the scale helps can be asked in both
modalities, and comparing them under different treatments confounds the sensor with the
preprocessing.

The edges cover the envelope measured over the five pilot cities: the 98th percentile of
red runs from 2,036 in Acapulco to 3,719 in Mérida, and that of the near infrared reaches
4,596. NDVI uses its natural limits.
"""

FIXED_RANGES = FIXED_RANGES_S1 | FIXED_RANGES_S2
"""Every fixed range, so any channel can be quantised without looking at the data."""


def robust_range(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> tuple:
    """Percentiles of the whole band, which set the common quantisation scale."""
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        raise ValueError("the band has not one finite pixel")
    return float(np.percentile(valid, p_low)), float(np.percentile(valid, p_high))


def quantise(band: np.ndarray, value_range: tuple, levels: int = LEVELS) -> np.ndarray:
    """Takes the band to integers from 1 to `levels`, reserving 0 for the invalid.

    Values outside the range are clipped to the ends and kept: a very bright sheet-metal
    roof is still information even when it lands above the 98th percentile.

    Non-finite values are taken to zero before converting to integer. `np.clip` leaves NaN
    untouched, and converting a NaN to an unsigned integer gives a result the specification
    does not define and that depends on the platform. The final `np.where` overwrites those
    positions anyway, but the intermediate value must not be left to chance: `to_db`
    produces NaN at every pixel with no measurable return, so the radar arm passes through
    here constantly.
    """
    low, high = value_range
    if not high > low:
        raise ValueError(f"degenerate range: {value_range}")
    finite = np.isfinite(band)
    scaled = np.zeros(band.shape, dtype="float64")
    np.divide(band - low, high - low, out=scaled, where=finite)
    scaled = np.clip(scaled, 0.0, 1.0)
    valid_levels = 1 + np.round(scaled * (levels - 1)).astype(np.uint8)
    return np.where(finite, valid_levels, 0).astype(np.uint8)


def entropy(glcm: np.ndarray) -> np.ndarray:
    """Shannon entropy per distance, averaging the angles of each one."""
    p = glcm.astype(np.float64)
    total = p.sum(axis=(0, 1), keepdims=True)
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    logarithm = np.zeros_like(p)
    np.log2(p, out=logarithm, where=p > 0)
    return (-(p * logarithm)).sum(axis=(0, 1)).mean(axis=1)


def _matrix(patch: np.ndarray, levels: int, distances: Sequence[int] = DISTANCES) -> np.ndarray:
    """Computes the GLCM of the patch and removes the level reserved for invalid pixels."""
    glcm = graycomatrix(
        patch,
        distances=list(distances),
        angles=list(ANGLES),
        levels=levels + 1,
        symmetric=True,
        normed=False,
    )
    return glcm[1:, 1:, :, :]


def feature_names(distances: Sequence[int] = DISTANCES) -> list[str]:
    """Names of the texture columns, one per property and distance."""
    families = [*PROPERTIES, "entropy", "anisotropy"]
    return [f"{family}_d{d}" for family in families for d in distances]


def features_of_patch(
    patch: np.ndarray, levels: int = LEVELS, distances: Sequence[int] = DISTANCES
) -> dict[str, float]:
    """Haralick properties of an already quantised patch, one per distance.

    Angles are averaged, because the urban fabric has no privileged orientation of
    interest. Distances are kept apart, because they are different scales.

    Of each property the spread across angles is also kept, under the name `anisotropy`,
    which separates an oriented fabric from one with no dominant direction.
    """
    empty = dict.fromkeys(feature_names(distances), np.nan)
    glcm = _matrix(patch, levels, distances)
    if glcm.sum() == 0:
        return empty

    features: dict[str, float] = {}
    for prop in PROPERTIES:
        values = graycoprops(glcm, prop)  # (distances, angles)
        for index, distance in enumerate(distances):
            features[f"{prop}_d{distance}"] = float(np.nanmean(values[index]))

    contrast = graycoprops(glcm, "contrast")
    for index, distance in enumerate(distances):
        features[f"anisotropy_d{distance}"] = float(np.nanstd(contrast[index]))

    for index, distance in enumerate(distances):
        features[f"entropy_d{distance}"] = float(entropy(glcm)[index])
    return features


def first_order_features(values: np.ndarray) -> dict[str, float]:
    """Statistics of the intensity distribution, with no regard for its spatial arrangement.

    They are the honest point of comparison for the GLCM: if texture adds nothing over the
    mean and the spread, it is better to know before defending it in the paper.
    """
    if values.size == 0:
        return dict.fromkeys(["mean", "std", "p10", "p50", "p90", "iqr"], np.nan)
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "iqr": float(p75 - p25),
    }


def split_half_reliability(
    band: np.ndarray,
    transform,
    geometries: Sequence,
    keys: Sequence[str],
    *,
    prefix: str = "c",
    levels: int = LEVELS,
    min_pixels: int = MIN_PIXELS,
    value_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Correlation of each feature between the two spatial halves of the same polygon.

    A feature that does not agree with itself when the AGEB is cut in two is not measuring
    the AGEB: it is measuring sampling noise. Since the cut is at the median of the
    horizontal coordinate, both halves share urban morphology and differ only in which
    pixels they touched, so the correlation between them bounds how much reproducible
    signal the feature has.

    It serves to choose which features enter the model on an objective criterion, decided
    before looking at performance and therefore immune to picking whatever suits.
    """
    lefts, rights, split_keys = [], [], []
    for key, geometry in zip(keys, geometries, strict=True):
        x_min, y_min, x_max, y_max = geometry.bounds
        middle = (x_min + x_max) / 2
        left = clip_by_rect(geometry, x_min, y_min, middle, y_max)
        right = clip_by_rect(geometry, middle, y_min, x_max, y_max)
        if left.is_empty or right.is_empty:
            continue
        lefts.append(left)
        rights.append(right)
        split_keys.append(key)

    shared = {
        "prefix": prefix,
        "levels": levels,
        "min_pixels": min_pixels,
        "value_range": value_range,
    }
    one = features_per_ageb(band, transform, lefts, split_keys, **shared)
    other = features_per_ageb(band, transform, rights, split_keys, **shared)

    rows = []
    for column in one.columns:
        if column == "cvegeo" or column.endswith("_n_px"):
            continue
        a, b = one[column], other[column]
        valid = a.notna() & b.notna()
        if valid.sum() < 3 or a[valid].nunique() < 2 or b[valid].nunique() < 2:
            rows.append({"feature": column, "n": int(valid.sum()), "r": np.nan})
            continue
        rows.append(
            {
                "feature": column,
                "n": int(valid.sum()),
                "r": float(np.corrcoef(a[valid], b[valid])[0, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)


def correlation_with_size(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Correlation of each feature with the number of pixels of the AGEB.

    The GLCM undersampling bias is monotone in the size of the polygon, and size correlates
    with urban density, which in turn correlates with deprivation. A feature strongly
    correlated with area is suspect: it may be pointing at the target by construction
    rather than by measuring morphology.
    """
    px_column = f"{prefix}_n_px"
    if px_column not in table:
        raise KeyError(f"the table does not carry {px_column}")

    rows = []
    for column in table.columns:
        if not column.startswith(f"{prefix}_") or column == px_column:
            continue
        valid = table[column].notna() & table[px_column].notna()
        if valid.sum() < 3 or table.loc[valid, column].nunique() < 2:
            continue
        r = np.corrcoef(table.loc[valid, column], table.loc[valid, px_column])[0, 1]
        rows.append({"feature": column, "n": int(valid.sum()), "r_with_n_px": float(r)})

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.reindex(
        output["r_with_n_px"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def features_per_ageb(
    band: np.ndarray,
    transform,
    geometries: Sequence,
    keys: Sequence[str],
    *,
    prefix: str,
    levels: int = LEVELS,
    min_pixels: int = MIN_PIXELS,
    value_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Extracts texture and first order features for every polygon over one band.

    The band is quantised once with the range of the whole city, and each AGEB is cut out
    of that. Returns one row per AGEB with the columns prefixed by the channel name, so
    several channels can be concatenated without a clash of names.

    Geometries have to arrive in the same reference system as `transform`, which for the
    composites is the UTM zone of the scenes and not geographic coordinates.

    With `value_range` the quantisation scale is fixed in advance. That is what radar
    needs: gamma0 is calibrated and deriving the range from each city would make the same
    backscatter fall into different levels depending on where it was measured.
    """
    if len(geometries) != len(keys):
        raise ValueError(f"{len(geometries)} geometries against {len(keys)} keys")

    value_range = value_range or robust_range(band)
    quantised = quantise(band, value_range, levels)
    rows = []

    for key, geometry in zip(keys, geometries, strict=True):
        base = {"cvegeo": key, f"{prefix}_n_px": 0}
        window = polygon_window(transform, geometry, band.shape)
        if window is None:
            rows.append(base)
            continue

        row_slice, col_slice, inside = window
        patch = quantised[row_slice, col_slice]
        raw = band[row_slice, col_slice]
        valid = inside & (patch > 0)
        n_px = int(valid.sum())
        base[f"{prefix}_n_px"] = n_px
        if n_px < min_pixels:
            rows.append(base)
            continue

        masked = np.where(valid, patch, 0).astype(np.uint8)
        features = features_of_patch(masked, levels)
        features.update(first_order_features(raw[valid & np.isfinite(raw)]))
        base.update({f"{prefix}_{k}": v for k, v in features.items()})
        rows.append(base)

    table = pd.DataFrame(rows)
    useful = int((table[f"{prefix}_n_px"] >= min_pixels).sum())
    log.info("%s: %d of %d AGEB with enough pixels", prefix, useful, len(table))
    return table
