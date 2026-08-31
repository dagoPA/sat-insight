"""Join of the urban AGEB with their Grado de Rezago Social.

CONEVAL publishes the label in an Excel workbook with no geometry; INEGI publishes the
geometry with no label. The thirteen character key —state, municipality, locality and AGEB
concatenated— is what joins them.

The result of that join is the unit of analysis of phase one: a polygon with an ordinal
class and seventeen deprivation indicators.
"""

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from satinsight.download import DATA_ROOT, ensure_coneval, ensure_inegi, urban_ageb_layer

log = logging.getLogger(__name__)

GRADES = ("Muy bajo", "Bajo", "Medio", "Alto", "Muy alto")
"""The five classes of the Grado de Rezago Social, in their natural order.

The labels keep the Spanish CONEVAL publishes them in: they are the values of a published
dataset, not text of ours, and translating them would break the join with the workbook."""

ORDINAL = {grade: i for i, grade in enumerate(GRADES)}

INDICATORS = (
    "analfabeta",
    "sin_escuela_6_14",
    "sin_escuela_15_24",
    "basica_incompleta",
    "sin_salud",
    "hacinamiento",
    "sin_agua",
    "sin_excusado",
    "sin_drenaje",
    "sin_electricidad",
    "piso_tierra",
    "sin_lavadora",
    "sin_refrigerador",
    "sin_telefono",
    "sin_celular",
    "sin_computadora",
    "sin_internet",
)
"""The seventeen deprivation indicators, in the order the workbook brings them."""

CONEVAL_COLUMNS = (
    "cve_ent",
    "entidad",
    "cve_mun",
    "municipio",
    "cve_loc",
    "localidad",
    "folio",
    "cvegeo",
    "poblacion",
    "viviendas",
    *INDICATORS,
    "grado",
)

HEADER_ROWS = 6
"""The workbook carries a title, a two-level header and a blank row before the data."""

WORKING_CRS = "EPSG:4326"
"""Boxes and STAC queries travel in geographic coordinates."""

METRIC_CRS = "EPSG:6372"
"""Lambert conformal conic for Mexico. Conurbation distances are measured here."""

NEIGHBOURHOOD_M = 5000.0
"""Distance to the core within which a neighbouring municipality counts as conurbated."""

BRIDGE_M = 2500.0
"""Maximum gap between AGEB for them to count as part of the same urban mass."""


@dataclass(frozen=True)
class City:
    """City, identified by the municipality that holds it."""

    key: str
    name: str
    state: str
    municipality: str


CITIES: dict[str, City] = {
    "tuxtla": City("tuxtla", "Tuxtla Gutiérrez", "07", "07101"),
    "merida": City("merida", "Mérida", "31", "31050"),
    "iztapalapa": City("iztapalapa", "Iztapalapa", "09", "09007"),
    "tapachula": City("tapachula", "Tapachula", "07", "07089"),
    "acapulco": City("acapulco", "Acapulco de Juárez", "12", "12001"),
}
"""The five cities of phase one.

The first three come from phase zero, chosen for their contrast in urban form and cloud
cover. Between them they cover the high end of deprivation badly: they add up to 1,199
AGEB of which 69 are high grade, 5.8% against the national 23%, and in Iztapalapa barely 3
of 454.

Tapachula and Acapulco enter to correct that bias. They have 48.6% and 36.5% of AGEB at
high grade, and with them the pilot sample approaches the composition of the country.
Without deprived AGEB in the set, the decision gate of phase one does not measure what it
claims to.
"""


def load_grs(root: Path = DATA_ROOT, *, use_cache: bool = True) -> pd.DataFrame:
    """Reads the Grado de Rezago Social of the country's urban AGEB.

    Reading the whole Excel workbook costs about half a minute, so the result is stored as
    parquet the first time.
    """
    cache = root / "grs_ageb_2020.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    workbook = ensure_coneval(root)
    log.info("reading %s", workbook.name)
    table = pd.read_excel(
        workbook, skiprows=HEADER_ROWS, header=None, names=list(CONEVAL_COLUMNS), dtype=str
    )

    numeric = ["poblacion", "viviendas", *INDICATORS]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")
    table["grado"] = table["grado"].str.strip()

    unknown = set(table["grado"].dropna().unique()) - set(GRADES)
    if unknown:
        raise ValueError(f"unexpected grades in the CONEVAL workbook: {sorted(unknown)}")
    table["ordinal"] = table["grado"].map(ORDINAL).astype("Int8")

    table = table.dropna(subset=["cvegeo", "grado"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    log.info("%d urban AGEB with a grade", len(table))
    return table


def load_geometry(state: str, root: Path = DATA_ROOT) -> gpd.GeoDataFrame:
    """Reads the urban AGEB polygons of one state, reprojected to WGS84."""
    layer = urban_ageb_layer(ensure_inegi(state, root))
    log.info("reading %s", layer.name)
    polygons = gpd.read_file(layer)
    polygons.columns = [c.lower() for c in polygons.columns]
    if polygons.crs is None:
        raise ValueError(f"layer {layer} arrives with no reference system")
    return polygons.to_crs(WORKING_CRS)


def _connected_mass(
    agebs: gpd.GeoDataFrame, core_municipality: str, bridge_m: float
) -> gpd.GeoDataFrame:
    """Keeps the continuous urban component holding the core municipality.

    A Mexican municipality usually includes urban localities separated from its seat by
    tens of kilometres. Tapachula drags in the coastal port, and with it the enclosing box
    grows fourfold to cover space that is almost all empty.

    Dilating every AGEB and joining what touches rebuilds the urban masses; keeping the one
    that holds the seat drops those satellites. In Tapachula that takes the box from 12.4
    to 1.0 megapixels while keeping 162 of 207 AGEB.
    """
    union = agebs.geometry.buffer(bridge_m).union_all()
    parts = list(getattr(union, "geoms", [union]))
    if len(parts) <= 1:
        return agebs

    points = agebs.geometry.representative_point()
    best, best_n = agebs, -1
    for part in parts:
        inside = agebs[points.within(part)]
        own = int((inside["cvegeo"].str[:5] == core_municipality).sum())
        if own > best_n:
            best, best_n = inside, own

    log.info(
        "connected mass: %d of %d AGEB, %d satellites dropped",
        len(best),
        len(agebs),
        len(agebs) - len(best),
    )
    return best


def agebs_of_city(
    key: str,
    root: Path = DATA_ROOT,
    *,
    min_population: int = 0,
    conurbation: bool = True,
    neighbourhood_m: float = NEIGHBOURHOOD_M,
    bridge_m: float = BRIDGE_M,
    catalogue: dict[str, "City"] | None = None,
) -> gpd.GeoDataFrame:
    """Returns the AGEB of a city with geometry and ordinal label.

    With `conurbation` the unit of analysis is the continuous urban mass: AGEB of any
    municipality of the state within `neighbourhood_m` of the core are admitted, and the
    result is cut to the connected component. A city rarely ends where its municipality
    ends, and the conurbated periphery is exactly where deprivation varies.

    The join with the labels is inner on purpose: an AGEB with no polygon or no label stays
    out of the set, and the count of what was dropped is logged so it can be audited.
    """
    catalogue = catalogue or CITIES
    city = catalogue.get(key)
    if city is None:
        known = ", ".join(sorted(catalogue)[:8])
        raise KeyError(f"unknown city: {key!r}. Among the known ones: {known}…")

    geometry = load_geometry(city.state, root)
    core = geometry[geometry["cvegeo"].str[:5] == city.municipality]
    if core.empty:
        raise ValueError(f"state {city.state} carries no AGEB of municipality {city.municipality}")

    if conurbation:
        metric = geometry.to_crs(METRIC_CRS)
        envelope = metric[metric["cvegeo"].isin(core["cvegeo"])].geometry.union_all()
        near = metric[metric.geometry.intersects(envelope.buffer(neighbourhood_m))]
        geometry = _connected_mass(near, city.municipality, bridge_m).to_crs(WORKING_CRS)
    else:
        geometry = core

    labels = load_grs(root)

    joined = geometry.merge(
        labels.drop(columns=["cve_ent", "cve_mun", "cve_loc", "folio"]),
        on="cvegeo",
        how="inner",
    )
    log.info(
        "%s: %d polygons, %d join with a label, %d municipalities",
        city.name,
        len(geometry),
        len(joined),
        joined["cve_mun"].nunique() if len(joined) else 0,
    )

    if min_population:
        before = len(joined)
        joined = joined[joined["poblacion"] >= min_population]
        log.info(
            "dropped %d AGEB with fewer than %d inhabitants", before - len(joined), min_population
        )

    joined["ciudad"] = city.key
    return joined.reset_index(drop=True)


def grade_summary(agebs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Counts AGEB and population per grade, in the ordinal order of the classes."""
    counts = (
        agebs.groupby("grado", observed=True)
        .agg(agebs=("cvegeo", "size"), poblacion=("poblacion", "sum"))
        .reindex(GRADES)
        .fillna(0)
        .astype(int)
    )
    counts["pct_agebs"] = (100 * counts["agebs"] / max(counts["agebs"].sum(), 1)).round(1)
    return counts


MIN_AGEBS_PER_CITY = 150
"""Minimum bag size for a city to enter the national set.

A hundred and fifty AGEB leave 81 cities and 24,080 AGEB, sixteen times the pilot sample.
Lowering the threshold adds ever smaller cities: at 100 they are 132 and at 50 they are
256, and a bag of fifty instances contributes little to the MIL against what compositing
its whole city costs.
"""


def _municipality_key(name: str) -> str:
    """Short, stable identifier built from the name of the municipality."""
    flat = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return "".join(c for c in flat if c.isalnum())


MIN_AGEBS_DEPRIVED = 50
DEPRIVED_SHARE = 0.25
"""Thresholds of the stratified branch of the selection.

Choosing cities by size alone produces a systematically well-off sample: the largest 81
have 9.8% of AGEB at high grade against 23.1% for the country, because large cities are
less deprived and conurbation adds well-off periphery. Widening that way would worsen the
very thing Tapachula and Acapulco came to correct in phase one.

The stratified branch admits small cities with concentrated high deprivation. There are 62
of them adding 5,302 AGEB, among them Ocosingo with 96.6% of its territory at high grade.
"""


MIN_AGEBS_EXTRA = 30
"""Floor for the expansion beyond the 138 of the national set.

The selected 138 hold 28,073 AGEB and 14.4% of them at high grade. The 2,331 municipalities
left out hold 33,357 and 30.5%, so the sample the project has been training and measuring
on is the well-off half of urban Mexico. Everything below thirty AGEB is a handful of
blocks whose bag carries almost no internal order to recover; above it there are 292
municipalities and 17,160 AGEB.
"""


def cities_extra(
    min_agebs: int = MIN_AGEBS_EXTRA, root: Path = DATA_ROOT, **kwargs
) -> dict[str, City]:
    """Municipalities outside the national set, keyed so the existing ones never move.

    The keys of `cities_by_size` are disambiguated against the selected set, which makes
    them a function of the selection. Widening the floor to thirty AGEB introduces homonyms
    of five cities already on disk — Cancún, La Paz, Matamoros, Tonalá and Cuauhtémoc — and
    every one of them would come back with a suffix it did not have before. Their
    composites are named after the old key, the partition refers to the old key, and
    nothing would raise: the cities would quietly be downloaded again as new ground.

    So the base keys are inherited whole, and only what the base does not already contain
    gets a key of its own.
    """
    base = cities_by_size(root=root, stratify=True)
    known = {city.municipality for city in base.values()}
    # a city's box is a conurbation: its bags cover every municipality the box touches,
    # and nine of those neighbours sat in the validation cities while also qualifying for
    # the expansion. excluding only the seat municipality let the same ground train and
    # evaluate. every municipality that any base city already turned into a bag is off
    # limits, whichever split that city belongs to.
    from satinsight.dataset import paths

    bags_dir = paths(root)["bags"]
    for key in base:
        bag_file = bags_dir / f"{key}.parquet"
        if bag_file.exists():
            known |= set(pd.read_parquet(bag_file, columns=["municipio"]).municipio)
    wide = cities_by_size(min_agebs=min_agebs, root=root, stratify=True, **kwargs)
    extra: dict[str, City] = {}
    for key, city in wide.items():
        if city.municipality in known:
            continue
        # a key of the wide set cannot equal a base key, because a municipality sharing a
        # name with one of the base is a homonym there too and both come back suffixed.
        # the guard stays anyway: the cost of being wrong is silent, and it is 12 GB.
        if key in base or key in extra:
            key = f"{key}{city.municipality}"
        extra[key] = city
    log.info("%d municipalities beyond the national set", len(extra))
    return extra


def catalogue_with_extra(
    min_agebs: int = MIN_AGEBS_EXTRA, root: Path = DATA_ROOT
) -> dict[str, City]:
    """The national set plus the expansion, under one set of stable keys."""
    return {**cities_by_size(root=root, stratify=True), **cities_extra(min_agebs, root=root)}


def cities_by_size(
    min_agebs: int = MIN_AGEBS_PER_CITY,
    root: Path = DATA_ROOT,
    *,
    stratify: bool = False,
    min_deprived: int = MIN_AGEBS_DEPRIVED,
    deprived_share: float = DEPRIVED_SHARE,
) -> dict[str, City]:
    """Derives from the census the list of cities above a minimum size.

    Writing five cities by hand was reasonable; writing eighty-one invites typos in
    municipality keys that nobody would catch until a join came out empty. The list is
    computed from the same table that supplies the labels.

    The five pilot cities keep the key their composites were already named with on disk.
    Changing it would force recompositing them, which is several hours of downloading.

    Homonyms across states are disambiguated with the municipality key, because the name
    alone does not identify: there is more than one Guadalupe and more than one Zaragoza in
    the country.
    """
    table = load_grs(root)
    counts = (
        table.assign(high=table["grado"].isin(GRADES[3:]))
        .groupby(["cve_ent", "cve_mun", "municipio"], observed=True)
        .agg(agebs=("cvegeo", "size"), altos=("high", "sum"))
        .reset_index()
    )
    large = counts["agebs"] >= min_agebs
    if stratify:
        deprived = (counts["agebs"] >= min_deprived) & (
            counts["altos"] / counts["agebs"] >= deprived_share
        )
        counts = counts[large | deprived]
    else:
        counts = counts[large]
    counts = counts.sort_values("agebs", ascending=False)

    inherited = {c.municipality: key for key, c in CITIES.items()}
    proposed = [_municipality_key(n) for n in counts["municipio"]]
    repeated = {c for c in proposed if proposed.count(c) > 1}

    cities: dict[str, City] = {}
    for (_, row), proposal in zip(counts.iterrows(), proposed, strict=True):
        key = inherited.get(row.cve_mun)
        if key is None:
            key = f"{proposal}{row.cve_mun}" if proposal in repeated else proposal
        cities[key] = City(key, row.municipio, row.cve_ent, row.cve_mun)

    log.info("%d cities with at least %d AGEB", len(cities), min_agebs)
    return cities
