"""Descarga de los insumos tabulares y vectoriales de la fase 1.

Dos fuentes abiertas sostienen el baseline a nivel AGEB:

- CONEVAL publica el Grado de Rezago Social de las 61,430 AGEB urbanas del censo 2020,
  en un solo libro de Excel comprimido.
- INEGI publica la geometría de esas AGEB dentro del Marco Geoestadístico 2020,
  empaquetado por entidad federativa.

Ambos portales entregan los archivos sin autenticación. Las URLs quedan fijadas aquí para
que `data/` se pueda borrar y regenerar completo, que es lo que pide el proyecto.

El servidor de CONEVAL responde con una página de bloqueo a los clientes que no mandan un
agente de usuario de navegador, así que todas las peticiones lo llevan.
"""

import logging
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)

RAIZ_DATOS = Path("data")
"""Carpeta donde aterrizan los datos crudos. Ignorada por git."""

URL_CONEVAL = "https://www.coneval.org.mx/Medicion/Documents/GRS_AGEB_2020/GRS_AGEB_urbana_2020.zip"
LIBRO_CONEVAL = "GRS_AGEB_urbana_2020.xlsx"

BASE_INEGI = (
    "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/"
    "productos/geografia/marcogeo/889463807469"
)

ENTIDADES: dict[str, str] = {
    "07": "07_chiapas",
    "09": "09_ciudaddemexico",
    "31": "31_yucatan",
}
"""Entidades que cubren las tres ciudades piloto, con el nombre de su paquete en INEGI.

Agregar una entidad nueva es escribir su renglón: el patrón de URL es el mismo para las 32.
"""

AGENTE = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

TIEMPO_ESPERA = (30, 300)
"""Segundos de espera para conectar y para leer. Los portales de gobierno son lentos."""


def _es_zip(ruta: Path) -> bool:
    """Confirma que el archivo descargado sea un zip real y no una página de error."""
    try:
        with zipfile.ZipFile(ruta) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def descargar(url: str, destino: Path, *, forzar: bool = False, intentos: int = 3) -> Path:
    """Trae un archivo a disco si falta, verificando que llegue completo.

    Un zip corrupto o una página de error disfrazada de descarga se detectan y se
    reintentan. El archivo se escribe primero como `.parcial` y se renombra al final, de
    modo que una interrupción nunca deja un archivo a medias que parezca válido.
    """
    if destino.exists() and not forzar:
        log.info("ya está: %s (%.1f MB)", destino.name, destino.stat().st_size / 1e6)
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")

    ultimo_error: Exception | None = None
    for intento in range(1, intentos + 1):
        try:
            log.info("descargando %s (intento %d/%d)", destino.name, intento, intentos)
            with requests.get(
                url, stream=True, timeout=TIEMPO_ESPERA, headers={"User-Agent": AGENTE}
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                escrito = 0
                with parcial.open("wb") as f:
                    for trozo in r.iter_content(chunk_size=1 << 20):
                        f.write(trozo)
                        escrito += len(trozo)
                if total and escrito != total:
                    raise OSError(f"descarga truncada: {escrito} de {total} bytes")

            if destino.suffix == ".zip" and not _es_zip(parcial):
                raise OSError("el archivo recibido no es un zip válido")

            parcial.replace(destino)
            log.info("listo %s (%.1f MB)", destino.name, destino.stat().st_size / 1e6)
            return destino

        except (requests.RequestException, OSError) as e:
            ultimo_error = e
            log.warning("falló %s: %s", destino.name, e)
            parcial.unlink(missing_ok=True)

    raise RuntimeError(f"no se pudo descargar {url} tras {intentos} intentos") from ultimo_error


def _extraer(archivo_zip: Path, destino: Path) -> Path:
    """Descomprime un zip una sola vez, marcando el resultado con un sello."""
    sello = destino / ".extraido"
    if sello.exists():
        return destino
    destino.mkdir(parents=True, exist_ok=True)
    log.info("extrayendo %s", archivo_zip.name)
    with zipfile.ZipFile(archivo_zip) as z:
        z.extractall(destino)
    sello.touch()
    return destino


def asegurar_coneval(raiz: Path = RAIZ_DATOS, *, forzar: bool = False) -> Path:
    """Deja en disco el libro de Excel del Grado de Rezago Social y devuelve su ruta."""
    crudo = raiz / "crudo"
    comprimido = descargar(URL_CONEVAL, crudo / "GRS_AGEB_urbana_2020.zip", forzar=forzar)
    carpeta = _extraer(comprimido, raiz / "coneval")
    libro = carpeta / LIBRO_CONEVAL
    if not libro.exists():
        candidatos = sorted(carpeta.glob("*.xlsx"))
        if not candidatos:
            raise FileNotFoundError(f"no apareció ningún .xlsx dentro de {comprimido}")
        libro = candidatos[0]
    return libro


def asegurar_inegi(entidad: str, raiz: Path = RAIZ_DATOS, *, forzar: bool = False) -> Path:
    """Deja en disco el Marco Geoestadístico de una entidad y devuelve su carpeta."""
    if entidad not in ENTIDADES:
        conocidas = ", ".join(sorted(ENTIDADES))
        raise KeyError(f"entidad sin URL registrada: {entidad!r}. Conocidas: {conocidas}")
    nombre = ENTIDADES[entidad]
    comprimido = descargar(
        f"{BASE_INEGI}/{nombre}.zip", raiz / "crudo" / f"{nombre}.zip", forzar=forzar
    )
    return _extraer(comprimido, raiz / "inegi" / entidad)


def capa_ageb_urbana(carpeta_entidad: Path) -> Path:
    """Localiza el shapefile de AGEB urbanas dentro del paquete de una entidad.

    El Marco Geoestadístico nombra sus capas con el código de entidad y un sufijo de una
    o dos letras: `a` para AGEB urbana, `ar` para AGEB rural, `m` para municipio. Buscar
    por patrón evita depender de cómo quedó anidada la carpeta al descomprimir.
    """
    candidatos = [p for p in carpeta_entidad.rglob("*a.shp") if not p.stem.endswith("ar")]
    if not candidatos:
        disponibles = sorted(p.name for p in carpeta_entidad.rglob("*.shp"))
        raise FileNotFoundError(
            f"no hay capa de AGEB urbana en {carpeta_entidad}. Capas presentes: {disponibles}"
        )
    return candidatos[0]
