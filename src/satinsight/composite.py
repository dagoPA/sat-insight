"""Annual median composites of Sentinel-1 and Sentinel-2.

Compositing serves a single purpose here: suppressing cloud in the optical arm and speckle
in the radar one. The object of analysis is still a static image of a single annual slice.
"""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pystac import Item

from satinsight.aoi import Bbox
from satinsight.catalog import VALID_SCL, by_cloud_cover, group_by_orbit
from satinsight.raster import read_window

log = logging.getLogger(__name__)

RGB_BANDS = ("B04", "B03", "B02")
MIN_SCENE_COVERAGE = 0.05
"""Fraction of valid pixels below which a scene is dropped."""

FAILURE_FRACTION = 0.3
"""Fraction of failed reads above which the composite is taken as broken.

Dropping the scene that cannot be read and carrying on is right in the face of one broken
scene, and ruinous in the face of a general outage: if the access signature expires
mid-run nearly all of them fail, and the composite comes back built from a handful of
scenes with nothing to warn about it. That is how an Acapulco composite once got saved
with four scenes out of thirty.

The threshold counts failed reads and not used scenes, because they are different things.
A scene dropped for cloud is a fact about that city's sky; one dropped on a read error is a
symptom of an outage. Mixing them would abort Tapachula, among the cloudiest cities in the
country, for a reason that is in no way anomalous.
"""


def _check_failures(sensor: str, failed: int, attempted: int, fraction: float | None) -> None:
    """Raises when too many reads failed for the result to be trusted.

    `None` turns the check off. Zero is its opposite and means what it looks like: not one
    failed read is tolerated.
    """
    if fraction is None or not attempted:
        return
    if failed > fraction * attempted:
        raise RuntimeError(
            f"{failed} of {attempted} {sensor} scenes failed to read. "
            "A composite built from what remains is not representative; it usually means "
            "the access signature expired mid-run or the service is not answering."
        )


TILE_COVERAGE = 0.02
"""Fraction of the box an MGRS tile has to reach to enter the composite."""


def useful_tiles(
    items: list["Item"],
    bbox: Bbox,
    samples: int = 2,
    minimum: float = TILE_COVERAGE,
    read=None,
) -> list["Item"]:
    """Drops the scenes whose MGRS tile does not reach the box.

    Sentinel-2 is delivered in fixed tiles, and the catalogue returns every scene whose
    tile intersects the box asked for, however little. A city that falls split between two
    tiles then receives scenes that carry not one pixel over its box: outside their
    footprint the read is filled with zeros and an SCL of zero means no data, so the mask
    comes out empty and the scene is dropped inside the loop anyway.

    The cost is not dropping them, it is having chosen them: the selection keeps the
    clearest of the year without looking at where they fall, and over San Pedro Tlaquepaque
    nineteen of the twenty best turned out to be from the tile that does not touch the
    city. One scene was left for the whole median.

    Each tile is therefore probed once, at low resolution, and those that do contribute are
    kept. A box split between two tiles keeps both, and the per-pixel median combines them
    wherever each one has data.
    """
    read = read or read_window
    groups: dict[str, list] = defaultdict(list)
    for item in items:
        groups[item.properties.get("s2:mgrs_tile", "?")].append(item)

    kept: list = []
    for tile, scenes in sorted(groups.items()):
        fractions = []
        for scene in by_cloud_cover(scenes)[:samples]:
            try:
                scl = read(scene.assets["SCL"].href, bbox, PROBE_SHAPE)
            except Exception:
                log.warning("probe failed on %s", scene.id, exc_info=True)
                continue
            fractions.append(float((scl > 0).mean()))
        coverage = max(fractions) if fractions else 0.0
        if coverage >= minimum:
            kept.extend(scenes)
        else:
            log.info("tile %s dropped: covers %.0f%% of the box", tile, 100 * coverage)
    if not kept:
        raise RuntimeError(f"none of the {len(groups)} Sentinel-2 tiles reaches the box")
    log.info("%d scenes in useful tiles of %d", len(kept), len(items))
    return kept


def composite_s2(
    items: list["Item"],
    bbox: Bbox,
    shape: tuple[int, int] | None = None,
    bands: tuple[str, ...] = RGB_BANDS,
    max_scenes: int = 36,
    failure_fraction: float | None = FAILURE_FRACTION,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Per-pixel median of the clearest Sentinel-2 scenes, with the SCL mask applied.

    Returns the composited bands together with the metadata of the run, among it how many
    observations went into the median of the typical pixel. Scenes are walked from the
    clearest to the cloudiest.

    Aborts when too many reads fail. `failure_fraction` at `None` turns that check off for
    whoever wants a partial composite on purpose; at zero it tolerates not one failed read.
    """
    if not items:
        raise ValueError("there are no Sentinel-2 scenes to composite")

    stacks: dict[str, list[np.ndarray]] = {band: [] for band in bands}
    used = 0
    failed = 0
    # the cap applies within each tile and not over the mixture: a box split between two
    # tiles needs a full stack on each side, because every scene contributes pixels only in
    # its half and the median of the other would be built from whatever is left over
    useful = useful_tiles(items, bbox)
    per_tile: dict[str, list] = defaultdict(list)
    for item in useful:
        per_tile[item.properties.get("s2:mgrs_tile", "?")].append(item)
    selection = [s for group in per_tile.values() for s in by_cloud_cover(group)[:max_scenes]]

    for item in selection:
        try:
            scl = read_window(item.assets["SCL"].href, bbox, shape)
            mask = np.isin(scl, list(VALID_SCL))
            if mask.mean() < MIN_SCENE_COVERAGE:
                continue
            for band in bands:
                array = read_window(item.assets[band].href, bbox, shape).astype("float32")
                array[~mask] = np.nan
                stacks[band].append(array)
            used += 1
        except Exception:
            failed += 1
            log.warning("S2 scene skipped: %s", item.id, exc_info=True)

    _check_failures("Sentinel-2", failed, len(selection), failure_fraction)
    if used == 0:
        raise RuntimeError("no Sentinel-2 scene contributed valid pixels")

    stack = np.dstack(stacks[bands[0]])
    depth = np.isfinite(stack).sum(axis=2)
    composite = {band: np.nanmedian(np.dstack(layers), axis=2) for band, layers in stacks.items()}
    meta = {
        "scenes_used": used,
        "scenes_selected": len(selection),
        "tiles": len(per_tile),
        # how many observations went into the median of the typical pixel: this is what
        # fixes the residual noise, and unlike the scene count it is not fooled by a box
        # split between tiles, where every scene covers only its half
        "median_depth": int(np.median(depth)),
        "minimum_depth": int(np.percentile(depth, 5)),
    }
    return composite, meta


VALID_FRACTION = 0.80
"""Minimum fraction of observed pixels demanded of a radar composite."""


def _check_composite_s1(composite: dict[str, np.ndarray], minimum: float = VALID_FRACTION) -> float:
    """Rejects a radar composite that comes out unobserved or with impossible values.

    Gamma0 in linear power is strictly positive. A pixel at zero or negative can only come
    from the scene's no-data entering the median, and that defect does not announce itself:
    with an even number of scenes the median averages the sentinel with a good value and
    returns an intermediate number that looks like data. The only way to see it is to count
    signs.

    A high fraction of NaN means the chosen orbit does not pass over the city. Better that
    the city fail than that it enter the set with a hollow composite.
    """
    for polarisation, array in composite.items():
        finite = np.isfinite(array)
        fraction = float(finite.mean())
        if fraction < minimum:
            raise RuntimeError(
                f"the Sentinel-1 composite observed only {100 * fraction:.0f}% of the box "
                f"in {polarisation}; no orbit covers the city"
            )
        improper = float((array[finite] <= 0).mean())
        if improper > 0:
            raise RuntimeError(
                f"{100 * improper:.1f}% of {polarisation} came out zero or negative, "
                "which linear gamma0 does not admit: the scene's no-data entered the median"
            )
    return min(float(np.isfinite(a).mean()) for a in composite.values())


ORBIT_SAMPLES = 4
"""Scenes probed per orbit to estimate how much data it leaves over the box."""

PROBE_SHAPE = (64, 64)
"""Probe grid. Enough to measure what fraction of the box falls inside the swath."""


def useful_coverage(
    items: list["Item"],
    bbox: Bbox,
    samples: int = ORBIT_SAMPLES,
    read=None,
) -> float:
    """Mean fraction of observed pixels that some scenes leave over the box.

    The footprint the catalogue declares does not answer this question. Over Mexicali, the
    orbit whose footprint covers 99% of the box delivers scenes that alternate between 1%
    and 99% of pixels with data, because the city falls on the edge of the swath. A few
    scenes are probed at low resolution and what actually arrives is averaged.

    A read that fails stays out of the average rather than counting as zero. Counting it
    confuses "this orbit does not see the city" with "this request was cut", and under a
    congested link the second is common: over Guasave one lost read out of four was enough
    to push an orbit covering the whole box below one covering half of it, and the city was
    left with no radar composite over a network problem.

    Zero is returned only when no read arrived, which really is indistinguishable from
    having no coverage, and the composite's guard catches it afterwards.

    `read` resolves at call time rather than at definition, so replacing `read_window` in
    the module is enough to leave the tests without network.
    """
    read = read or read_window
    fractions = []
    for item in items[:samples]:
        try:
            got = read(item.assets["vv"].href, bbox, PROBE_SHAPE).astype("float32")
        except Exception:
            log.warning("probe failed on %s, left out of the average", item.id, exc_info=True)
            continue
        fractions.append(float(np.isfinite(got).mean()))
    return float(np.mean(fractions)) if fractions else 0.0


def useful_orbit(
    items: list["Item"],
    bbox: Bbox,
    samples: int = ORBIT_SAMPLES,
    read=None,
) -> tuple[tuple[str, int], list["Item"], float]:
    """Acquisition geometry that leaves the most observed pixels over the box.

    Measured coverage decides and the scene count only breaks ties, rounding to hundredths
    so a difference of nothing does not topple an orbit with many more passes.
    """
    groups = group_by_orbit(items)
    if not groups:
        raise ValueError("there are no SAR scenes to group")
    coverage = {k: useful_coverage(v, bbox, samples, read) for k, v in groups.items()}
    key = max(groups, key=lambda k: (round(coverage[k], 2), len(groups[k])))
    return key, groups[key], coverage[key]


def composite_s1(
    items: list["Item"],
    bbox: Bbox,
    shape: tuple[int, int] | None = None,
    max_scenes: int = 24,
    failure_fraction: float | None = FAILURE_FRACTION,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Per-pixel median of Sentinel-1 RTC scenes from a single orbit geometry.

    Returns the composited polarisations in linear power together with the metadata of the
    chosen acquisition.

    Aborts when too many reads fail. `failure_fraction` at `None` turns that check off for
    whoever wants a partial composite on purpose; at zero it tolerates not one failed read.
    """
    if not items:
        raise ValueError("there are no Sentinel-1 scenes to composite")

    (state, relative), available, coverage = useful_orbit(items, bbox)
    log.info("S1 orbit %s relative %d: covers %.0f%%", state, relative, 100 * coverage)
    selection = available[:max_scenes]

    stacks: dict[str, list[np.ndarray]] = {"vv": [], "vh": []}
    failed = 0
    for item in selection:
        # Both polarisations are read before either is stored: appending inside the loop
        # would leave VV stacked and VH not when the second read fails, and the medians of
        # one and the other would come out computed over different sets of scenes.
        try:
            got = {
                polarisation: read_window(item.assets[polarisation].href, bbox, shape).astype(
                    "float32"
                )
                for polarisation in stacks
            }
        except Exception:
            failed += 1
            log.warning("S1 scene skipped: %s", item.id, exc_info=True)
            continue
        for polarisation, array in got.items():
            stacks[polarisation].append(array)

    _check_failures("Sentinel-1", failed, len(selection), failure_fraction)
    if not stacks["vv"]:
        raise RuntimeError("no Sentinel-1 scene contributed valid pixels")

    composite = {
        polarisation: np.nanmedian(np.dstack(layers), axis=2)
        for polarisation, layers in stacks.items()
    }
    valid = _check_composite_s1(composite)
    meta = {
        "observed_fraction": round(valid, 3),
        "orbit": f"{state} · relative {relative}",
        "scenes_used": len(stacks["vv"]),
        "scenes_available": len(available),
        "orbit_coverage": round(coverage, 3),
    }
    return composite, meta
