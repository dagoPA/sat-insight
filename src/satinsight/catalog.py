"""Access to the STAC catalogue of Microsoft Planetary Computer.

Scenes are queried and signed here; reading pixels lives in `raster`.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

import planetary_computer as pc
import pystac_client

if TYPE_CHECKING:
    from pystac import Item

from satinsight.aoi import Bbox

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLLECTION_S1 = "sentinel-1-rtc"
COLLECTION_S2 = "sentinel-2-l2a"

VALID_SCL = frozenset({4, 5, 6, 7})
"""SCL mask classes that count as usable pixels: vegetation, bare soil, water and
unclassified. The rest is cloud, shadow, snow or saturation."""


SIGNABLE_DOMAIN = "blob.core.windows.net"


def open_catalogue() -> pystac_client.Client:
    """STAC client that signs links to the COGs automatically."""
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def sign(href: str) -> str:
    """Renews a link's signature just before it is read.

    Signing when the catalogue is queried is enough for an immediate read and fails for a
    composite: Planetary Computer tokens expire in about an hour, and compositing a city
    takes longer than that. Late reads get a 403, and since compositing drops the scene
    that fails, the result is a composite built from a fraction of the scenes asked for
    with no error in sight.

    `planetary_computer` caches the token in memory per container and only asks again when
    it expires, so renewing on every read costs no extra request.

    Anything not pointing at an Azure container comes back untouched, so tests can read
    local files through this same path.
    """
    if SIGNABLE_DOMAIN not in href:
        return href
    return pc.sign(href.split("?", 1)[0])


def search(
    collection: str,
    bbox: Bbox,
    period: str,
    catalogue: pystac_client.Client | None = None,
) -> list["Item"]:
    """Scenes of a collection intersecting the box over the given period.

    The period uses STAC interval syntax, for example ``"2020-01-01/2020-12-31"``.
    """
    catalogue = catalogue or open_catalogue()
    found = catalogue.search(collections=[collection], bbox=bbox, datetime=period)
    return list(found.items())


def cloud_summary(items: list["Item"]) -> dict[str, float | int]:
    """Cloud cover statistics of a set of Sentinel-2 scenes."""
    if not items:
        raise ValueError("there are no scenes to summarise")
    clouds = sorted(item.properties["eo:cloud_cover"] for item in items)
    total = len(clouds)
    return {
        "scenes": total,
        "minimum": round(clouds[0], 1),
        "maximum": round(clouds[-1], 1),
        "median": round(clouds[total // 2], 1),
        "pct_over_50": round(100 * sum(c > 50 for c in clouds) / total),
        "pct_over_80": round(100 * sum(c > 80 for c in clouds) / total),
    }


def by_cloud_cover(items: list["Item"]) -> list["Item"]:
    """Sentinel-2 scenes ordered from the clearest to the cloudiest."""
    return sorted(items, key=lambda item: item.properties["eo:cloud_cover"])


def group_by_orbit(items: list["Item"]) -> dict[tuple[str, int], list["Item"]]:
    """Groups SAR scenes by orbit state and relative orbit number.

    A SAR composite is only coherent within one acquisition geometry: mixing ascending
    with descending changes the incidence angle and the direction of the radar shadows.
    """
    groups: dict[tuple[str, int], list[Item]] = defaultdict(list)
    for item in items:
        key = (
            item.properties.get("sat:orbit_state"),
            item.properties.get("sat:relative_orbit"),
        )
        groups[key].append(item)
    return dict(groups)


def dominant_orbit(items: list["Item"]) -> tuple[tuple[str, int], list["Item"]]:
    """Acquisition geometry with the most scenes available, with its scenes.

    Choosing by count alone works when any orbit covers the box. Over a border or coastal
    city the data that actually arrives has to be measured, and `composite.useful_orbit`
    takes care of that, since it can read pixels.
    """
    groups = group_by_orbit(items)
    if not groups:
        raise ValueError("there are no SAR scenes to group")
    key = max(groups, key=lambda k: len(groups[k]))
    return key, groups[key]
