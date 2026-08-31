"""Download of the tabular and vector inputs of phase one.

Two open sources hold up the AGEB-level baseline:

- CONEVAL publishes the Grado de Rezago Social of the 61,430 urban AGEB of the 2020
  census, in a single compressed Excel workbook.
- INEGI publishes the geometry of those AGEB inside the Marco Geoestadístico 2020,
  packaged per state.

Both portals hand the files over without authentication. The URLs are pinned here so that
`data/` can be deleted and regenerated whole, which is what the project asks for.

The CONEVAL server answers with a block page to clients that send no browser user agent,
so every request carries one.
"""

import logging
import os
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DATA_ROOT = Path("data")
"""Folder where the raw data lands. Ignored by git."""

CONEVAL_URL = "https://www.coneval.org.mx/Medicion/Documents/GRS_AGEB_2020/GRS_AGEB_urbana_2020.zip"
CONEVAL_WORKBOOK = "GRS_AGEB_urbana_2020.xlsx"

INEGI_BASE = (
    "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/"
    "productos/geografia/marcogeo/889463807469"
)

STATES: dict[str, str] = {
    "01": "01_aguascalientes",
    "02": "02_bajacalifornia",
    "03": "03_bajacaliforniasur",
    "04": "04_campeche",
    "05": "05_coahuiladezaragoza",
    "06": "06_colima",
    "07": "07_chiapas",
    "08": "08_chihuahua",
    "09": "09_ciudaddemexico",
    "10": "10_durango",
    "11": "11_guanajuato",
    "12": "12_guerrero",
    "13": "13_hidalgo",
    "14": "14_jalisco",
    "15": "15_mexico",
    "16": "16_michoacandeocampo",
    "17": "17_morelos",
    "18": "18_nayarit",
    "19": "19_nuevoleon",
    "20": "20_oaxaca",
    "21": "21_puebla",
    "22": "22_queretaro",
    "23": "23_quintanaroo",
    "24": "24_sanluispotosi",
    "25": "25_sinaloa",
    "26": "26_sonora",
    "27": "27_tabasco",
    "28": "28_tamaulipas",
    "29": "29_tlaxcala",
    "30": "30_veracruzignaciodelallave",
    "31": "31_yucatan",
    "32": "32_zacatecas",
}
"""The 32 states with the name of their package in the Marco Geoestadístico.

The names follow the code then the name in lowercase without accents or spaces, with one
exception verified against the server: Veracruz drops the preposition and goes as
`30_veracruzignaciodelallave`. Every URL was checked with a head request before being
pinned here.
"""

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

TIMEOUT = (30, 300)
"""Seconds to wait to connect and to read. Government portals are slow."""


def _is_zip(path: Path) -> bool:
    """Confirms the downloaded file is a real zip and not an error page."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download(url: str, destination: Path, *, force: bool = False, attempts: int = 3) -> Path:
    """Brings a file to disk when it is missing, checking that it arrives whole.

    A corrupt zip or an error page dressed as a download are detected and retried. The file
    is written first as `.partial` and renamed at the end, so an interruption never leaves
    a half-written file that looks valid.
    """
    if destination.exists() and not force:
        log.info("already here: %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    # The suffix carries the process id because several parallel downloads can ask for
    # the same state: the pipeline fetches it on demand while a bulk download job goes for
    # it too. With a shared temporary path, both processes write over the same file and the
    # zip arrives corrupt without either of them failing.
    partial = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.partial")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            log.info("downloading %s (attempt %d/%d)", destination.name, attempt, attempts)
            with requests.get(
                url, stream=True, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                written = 0
                with partial.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        written += len(chunk)
                if total and written != total:
                    raise OSError(f"truncated download: {written} of {total} bytes")

            if destination.suffix == ".zip" and not _is_zip(partial):
                raise OSError("the file received is not a valid zip")

            partial.replace(destination)
            log.info("done %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
            return destination

        except (requests.RequestException, OSError) as e:
            last_error = e
            log.warning("%s failed: %s", destination.name, e)
            partial.unlink(missing_ok=True)

    raise RuntimeError(f"could not download {url} after {attempts} attempts") from last_error


def _extract(zip_path: Path, destination: Path, pattern: str | None = None) -> Path:
    """Unpacks a zip once, marking the result with a stamp.

    With `pattern` only members whose name contains it are extracted. A state package of
    the Marco Geoestadístico weighs some 430 MB unpacked and carries fifteen layers, city
    blocks, rural AGEB, services, of which this work uses a single one taking 3 MB.
    Unpacking whole packages for 27 states would cost eleven gigabytes of layers nobody
    opens.
    """
    stamp = destination / (".extracted" if pattern is None else f".extracted_{pattern}")
    if stamp.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist() if pattern is None else [n for n in z.namelist() if pattern in n]
        if pattern is not None and not members:
            available = sorted({Path(n).name for n in z.namelist() if n.endswith(".shp")})
            raise FileNotFoundError(
                f"{zip_path.name} holds nothing matching {pattern!r}. Layers present: {available}"
            )
        log.info("extracting %d members from %s", len(members), zip_path.name)
        z.extractall(destination, members=members)
    stamp.touch()
    return destination


def ensure_coneval(root: Path = DATA_ROOT, *, force: bool = False) -> Path:
    """Leaves the Grado de Rezago Social workbook on disk and returns its path."""
    raw = root / "raw"
    archive = download(CONEVAL_URL, raw / "GRS_AGEB_urbana_2020.zip", force=force)
    folder = _extract(archive, root / "coneval")
    workbook = folder / CONEVAL_WORKBOOK
    if not workbook.exists():
        candidates = sorted(folder.glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError(f"no .xlsx turned up inside {archive}")
        workbook = candidates[0]
    return workbook


def ensure_inegi(state: str, root: Path = DATA_ROOT, *, force: bool = False) -> Path:
    """Leaves the Marco Geoestadístico of one state on disk and returns its folder."""
    if state not in STATES:
        known = ", ".join(sorted(STATES))
        raise KeyError(f"state with no registered URL: {state!r}. Known: {known}")
    name = STATES[state]
    archive = download(f"{INEGI_BASE}/{name}.zip", root / "raw" / f"{name}.zip", force=force)
    return _extract(archive, root / "inegi" / state, pattern=f"{state}a.")


NATURALEARTH_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
)
"""State boundaries of the world, public domain, at 1:10 million scale.

The 1:50 million edition weighs fifteen times less and covers only nine large countries;
Mexico shows up only in the 1:10 million one.

It serves only to place the pilot cities inside the country. The INEGI Marco
Geoestadístico would give the same outline with more precision, and downloading it whole
to draw 32 polygons would cost several gigabytes.
"""


def ensure_naturalearth(root: Path = DATA_ROOT, *, force: bool = False) -> Path:
    """Leaves the context state boundaries on disk and returns their shapefile."""
    archive = download(
        NATURALEARTH_URL,
        root / "raw" / "ne_10m_admin_1_states_provinces.zip",
        force=force,
    )
    folder = _extract(archive, root / "naturalearth")
    layers = sorted(folder.rglob("*.shp"))
    if not layers:
        raise FileNotFoundError(f"there is no shapefile inside {archive}")
    return layers[0]


def urban_ageb_layer(state_folder: Path, state: str | None = None) -> Path:
    """Locates the urban AGEB shapefile inside a state package.

    The Marco Geoestadístico names each layer with the state code followed by a suffix that
    identifies it: `a` is urban AGEB, `ar` rural AGEB, `m` municipality, `sia`
    infrastructure services. A state package carries fifteen layers and several end in the
    letter `a`, so the name is matched whole. Searching by loose suffix picks the wrong
    layer and the mistake goes unnoticed until the join comes out empty.
    """
    state = state or state_folder.name
    expected = f"{state}a"
    candidates = [p for p in state_folder.rglob("*.shp") if p.stem == expected]
    if not candidates:
        available = sorted(p.name for p in state_folder.rglob("*.shp"))
        raise FileNotFoundError(
            f"{expected}.shp is not in {state_folder}. Layers present: {available}"
        )
    return candidates[0]
