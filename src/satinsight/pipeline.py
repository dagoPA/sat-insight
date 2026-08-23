"""From a city to its table of features per AGEB.

Chains the pieces of phase one in the only order they can run: the AGEB define the box, the
box defines the catalogue search, the scenes define the grid, the grid allows compositing,
and the composite allows measuring texture inside each polygon.

Every composite is saved as it is built, so an interrupted run resumes without paying the
hour of downloading per city and sensor again.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from satinsight import cache
from satinsight.agebs import CITIES, agebs_of_city
from satinsight.aoi import AOI
from satinsight.catalog import COLLECTION_S1, COLLECTION_S2, open_catalogue, search
from satinsight.composite import composite_s1, composite_s2
from satinsight.download import DATA_ROOT
from satinsight.grid import Grid, grid_from_scenes
from satinsight.landcover import fractions_per_ageb, mosaic
from satinsight.raster import to_db
from satinsight.texture import FIXED_RANGES, FIXED_RANGES_S1, features_per_ageb

log = logging.getLogger(__name__)

CENSUS_PERIOD = "2020-01-01/2020-12-31"

BANDS_S2 = ("B04", "B03", "B02", "B08", "B11")
"""Red, green, blue, near infrared and shortwave infrared.

The first four are native at 10 m. B11 arrives at 20 m and is resampled, and it enters
because it enables NDBI, the standard built-up index. The central risk of the project is
that the model reads built density rather than the morphology of deprivation, so it is
worth holding the channel that best describes the former.

Blue and green feed no feature: they exist so natural colour can be rendered for any city.
"""

MARGIN_M = 200.0
"""Slack around the AGEB, so none is cut by the edge of the box."""

SENSORS = ("s2", "s1")

SCALES = ("native", "fixed", "percentiles")
"""How the texture quantisation scale is set, and why there are three options.

- `native` is the phase one criterion: a fixed range in decibels for radar, because gamma0
  is calibrated, and per-city percentiles for optical, because it carries atmospheric
  residuals.
- `fixed` quantises both modalities with fixed edges.
- `percentiles` estimates the range of each city in both.

The last two exist so the modalities can be compared under the same treatment. Measuring
them under different criteria confounds the sensor with the preprocessing, and that
confusion is exactly what the comparison has to rule out.
"""


def _channel_range(name: str, scale: str) -> tuple[float, float] | None:
    """Quantisation range of a channel under the requested scale, or `None` to estimate it."""
    if scale == "native":
        return FIXED_RANGES_S1.get(name)
    if scale == "fixed":
        return FIXED_RANGES.get(name)
    if scale == "percentiles":
        return None
    raise ValueError(f"unknown scale: {scale!r}. Valid: {', '.join(SCALES)}")


CAP_S2 = 20
CAP_S1 = 16
"""Scenes that enter each composite.

The same cap for every city: validation holds cities out, so a composite built from fewer
scenes in one of them would be confused with signal from that city. Sentinel-2 is walked
from the clearest scene to the cloudiest, so the first twenty are the best available.
"""

MIN_DEPTH = 8
"""Observations the typical pixel needs for the composite to count as comparable.

`composite` already aborts when too many reads fail, which is the symptom of an outage.
This second check looks at something else: how many observations the median of the typical
pixel was computed from. A city composited with half the observations of another carries
more residual noise, and since validation splits by city, that difference would read as
signal from that city. It answers to the experimental design, and that is why it lives here
while outage detection lives in the library.

It counts per pixel and not per scene because a city split between two MGRS tiles receives
scenes that cover only its half of the box: counting them whole gives a number no part of
the image ever had.
"""


def _require_depth(key: str, sensor: str, depth: int, minimum: int) -> None:
    """Warns when the median of the typical pixel came from too few observations."""
    if depth < minimum:
        raise RuntimeError(
            f"{key}/{sensor}: the typical pixel was composited from {depth} observations, "
            f"fewer than the {minimum} required. Comparing cities built at different depths "
            "mixes signal with sampling noise, and validation splits precisely by city."
        )


def city_aoi(
    key: str,
    root: Path = DATA_ROOT,
    *,
    margin_m: float = MARGIN_M,
    catalogue: dict | None = None,
) -> tuple[AOI, gpd.GeoDataFrame]:
    """Box wrapping the AGEB of a city, together with those AGEB.

    `catalogue` allows working over the national set `agebs.cities_by_size` returns instead
    of the five pilots written by hand.
    """
    catalogue = catalogue or CITIES
    agebs = agebs_of_city(key, root, catalogue=catalogue)
    city = catalogue[key]
    area = AOI.from_polygons(key, city.name, city.state, agebs, margin_m=margin_m)
    height, width = area.approximate_shape()
    log.info("%s: %d AGEB, box ~%dx%d px @10 m", city.name, len(agebs), width, height)
    return area, agebs


def build_composite(
    key: str,
    sensor: str,
    area: AOI,
    *,
    period: str = CENSUS_PERIOD,
    max_scenes: int | None = None,
    catalogue=None,
) -> tuple[dict[str, np.ndarray], Grid, dict]:
    """Composites a city and a sensor from the catalogue, without consulting disk."""
    if sensor not in SENSORS:
        raise ValueError(f"unknown sensor: {sensor!r}. Valid: {', '.join(SENSORS)}")

    catalogue = catalogue or open_catalogue()
    collection = COLLECTION_S2 if sensor == "s2" else COLLECTION_S1
    scenes = search(collection, area.bbox, period, catalogue)
    if not scenes:
        raise RuntimeError(f"the catalogue returned no {sensor} scenes for {key}")

    # for radar the UTM zone is chosen by the coverage each one reaches and not by how many
    # scenes it brings: the two zones of a city on the edge can see different halves
    score = None
    if sensor == "s1":

        def score(group):
            from satinsight.catalog import group_by_orbit
            from satinsight.composite import useful_coverage

            orbits = group_by_orbit(group)
            return max((useful_coverage(v, area.bbox) for v in orbits.values()), default=0.0)

    grid, scenes = grid_from_scenes(area.bbox, scenes, score=score)
    log.info("%s/%s: %d scenes, grid %.1f MP", key, sensor, len(scenes), grid.megapixels)

    if sensor == "s2":
        cap = max_scenes or CAP_S2
        bands, meta = composite_s2(scenes, area.bbox, grid.shape, BANDS_S2, cap)
        tags = {"scenes_available": len(scenes), **meta}
        depth = int(meta["median_depth"])
    else:
        cap = max_scenes or CAP_S1
        bands, meta = composite_s1(scenes, area.bbox, grid.shape, cap)
        tags = dict(meta)
        depth = int(meta["scenes_used"])

    _require_depth(key, sensor, depth, min(MIN_DEPTH, cap))
    tags |= {"ciudad": key, "sensor": sensor, "period": period, "bbox": list(area.bbox)}
    return bands, grid, tags


def _same_box(stored, current, tolerance: float = 1e-6) -> bool:
    """Compares the box a composite was built with against the one in force.

    The composite is saved under the city's name, so a change in the rule that defines the
    box would leave a file on disk covering another area and it would go on being reused
    with no warning. Comparing the stored box turns that silent corruption into a rebuild.
    """
    if stored is None:
        return False
    return len(stored) == len(current) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(stored, current, strict=True)
    )


def _same_bands(stored, sensor: str) -> bool:
    """Confirms a composite on disk carries the bands the pipeline expects today.

    Adding a band leaves earlier composites obsolete, and the file keeps loading without
    complaint until a derived channel looks for the missing one. Worse still if the new
    channel exists in some cities and not others: that difference would end up correlated
    with which cities were composited first, exactly the confound that validation by city
    has to avoid.
    """
    expected = set(BANDS_S2) if sensor == "s2" else {"vv", "vh"}
    return set(stored) == expected


def ensure_composite(
    key: str,
    sensor: str,
    *,
    area: AOI | None = None,
    root: Path = DATA_ROOT,
    period: str = CENSUS_PERIOD,
    force: bool = False,
    **kwargs,
) -> tuple[dict[str, np.ndarray], Grid, dict]:
    """Returns the composite from disk, building it the first time.

    Whoever already resolved the box can pass it in `area` to save a second read of the
    state shapefile, which for Mexico City runs to some eighty megabytes.
    """
    destination = cache.composite_path(key, sensor, root / "composites")
    if destination.exists() and not force:
        stored = cache.load(destination)
        box_ok = area is None or _same_box(stored[2].get("bbox"), area.bbox)
        bands_ok = _same_bands(stored[0], sensor)
        if box_ok and bands_ok:
            log.info("composite cached: %s", destination.name)
            return stored
        reason = "covers another box" if not box_ok else "is missing bands"
        log.warning("%s/%s cached %s; rebuilding", key, sensor, reason)

    if area is None:
        area, _ = city_aoi(key, root)
    bands, grid, tags = build_composite(key, sensor, area, period=period, **kwargs)
    cache.save(bands, grid, destination, **tags)
    return bands, grid, tags


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Quotient that returns NaN wherever the denominator vanishes."""
    output = np.full(numerator.shape, np.nan, dtype="float32")
    np.divide(numerator, denominator, out=output, where=np.abs(denominator) > 1e-6)
    return output


def channels_s2(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Channels texture is measured on in the optical arm.

    Red carries the signal of built material, the near infrared separates vegetation from
    bare soil, and NDVI summarises both in a bounded index that does not depend on absolute
    calibration. NDBI adds the shortwave infrared response, which is where built material
    is best told apart from bare soil.
    """
    red = bands["B04"].astype("float32")
    nir = bands["B08"].astype("float32")
    swir = bands["B11"].astype("float32")
    return {
        "s2red": red,
        "s2nir": nir,
        "s2ndvi": _safe_divide(nir - red, nir + red),
        "s2ndbi": _safe_divide(swir - nir, swir + nir),
    }


def channels_s1(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Channels of the radar arm, in decibels.

    The conversion happens here and never before saving: the composite lives in linear
    power because averaging in decibels biases the result towards the low values.
    """
    vv = to_db(bands["vv"])
    vh = to_db(bands["vh"])
    return {"s1vv": vv, "s1vh": vh, "s1ratio": vv - vh}


CHANNELS = {"s2": channels_s2, "s1": channels_s1}


def features_of_city(
    key: str,
    sensor: str,
    *,
    root: Path = DATA_ROOT,
    period: str = CENSUS_PERIOD,
    force: bool = False,
    max_scenes: int | None = None,
    scale: str = "native",
    catalogue: dict | None = None,
) -> pd.DataFrame:
    """Table of features per AGEB for one city and one sensor, with its ordinal label.

    `scale` chooses how texture is quantised; see `SCALES`.
    """
    area, agebs = city_aoi(key, root, catalogue=catalogue)
    bands, grid, _ = ensure_composite(
        key,
        sensor,
        area=area,
        root=root,
        period=period,
        force=force,
        max_scenes=max_scenes,
    )

    projected = agebs.to_crs(grid.crs)
    geometries = list(projected.geometry)
    keys = list(projected["cvegeo"])

    table = pd.DataFrame({"cvegeo": keys})
    for name, channel in CHANNELS[sensor](bands).items():
        partial = features_per_ageb(
            channel,
            grid.transform,
            geometries,
            keys,
            prefix=name,
            value_range=_channel_range(name, scale),
        )
        table = table.merge(partial, on="cvegeo", how="left")

    # Land cover does not depend on the sensor, but it is computed over this grid so that
    # its fractions and the texture features look at exactly the same pixels.
    classes = mosaic(area, grid)
    table = table.merge(
        fractions_per_ageb(classes, grid.transform, geometries, keys),
        on="cvegeo",
        how="left",
    )

    labels = agebs[["cvegeo", "ciudad", "grado", "ordinal", "poblacion", "viviendas"]]
    table = table.merge(labels, on="cvegeo", how="left")
    table["area_km2"] = projected.geometry.area.to_numpy() / 1e6
    return table


def features_of_all(
    sensor: str,
    cities: tuple[str, ...] = tuple(CITIES),
    *,
    root: Path = DATA_ROOT,
    max_scenes: int | None = None,
    scale: str = "native",
    catalogue: dict | None = None,
) -> pd.DataFrame:
    """Stacks the feature tables of several cities for one sensor.

    A city that fails does not stop the rest: with 138 cities, aborting over one forces
    repeating hours of work already done, and what was missed shows in the log.
    """
    parts = []
    for city in cities:
        try:
            parts.append(
                features_of_city(
                    city,
                    sensor,
                    root=root,
                    max_scenes=max_scenes,
                    scale=scale,
                    catalogue=catalogue,
                )
            )
        except Exception:
            log.warning("no features for %s", city, exc_info=True)
    if not parts:
        raise RuntimeError(f"no city yielded features for {sensor}")
    log.info("features of %d of %d cities", len(parts), len(cities))
    return pd.concat(parts, ignore_index=True)


def reliability_of_cities(
    sensor: str,
    cities: tuple[str, ...] | None = None,
    *,
    root: Path = DATA_ROOT,
    scale: str = "fixed",
    catalogue: dict | None = None,
) -> pd.DataFrame:
    """Split-half correlation of each feature, aggregated over many cities.

    Every city contributes one correlation per feature, and the median across cities is
    kept together with the worst. Measuring it over few cities leaves the criterion at the
    mercy of their peculiarities: a feature can reproduce in five southern cities and be
    noise in the north, and the filter would let it through with nothing to warn about it.
    """
    from satinsight.agebs import cities_by_size
    from satinsight.texture import split_half_reliability

    catalogue = catalogue or cities_by_size(root=root, stratify=True)
    keys = cities or tuple(
        p.stem.replace(f"_{sensor}", "")
        for p in sorted((root / "composites").glob(f"*_{sensor}.tif"))
    )

    parts = []
    for key in keys:
        try:
            area, agebs = city_aoi(key, root, catalogue=catalogue)
            bands, grid, _ = ensure_composite(key, sensor, area=area, root=root)
            agebs = agebs.to_crs(grid.crs)
            channels = channels_s2(bands) if sensor == "s2" else channels_s1(bands)
            for name, band in channels.items():
                parts.append(
                    split_half_reliability(
                        band,
                        grid.transform,
                        list(agebs.geometry),
                        list(agebs.cvegeo),
                        prefix=name,
                        value_range=_channel_range(name, scale),
                    ).assign(ciudad=key)
                )
        except Exception:
            log.warning("no reliability for %s", key, exc_info=True)

    if not parts:
        raise RuntimeError(f"no city yielded reliability for {sensor}")
    together = pd.concat(parts, ignore_index=True)
    summary = (
        together.groupby("feature", observed=True)["r"]
        .agg(r_median="median", r_min="min", cities="size")
        .reset_index()
        .sort_values("r_median", ascending=False)
    )
    log.info("reliability of %d features over %d cities", len(summary), together.ciudad.nunique())
    return summary
