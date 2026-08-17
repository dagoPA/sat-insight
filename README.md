# satinsight

Desagregación del rezago social a partir de imágenes satelitales, con aprendizaje
débilmente supervisado.

Un modelo MIL con atención, en la línea de CLAM, recibe una sola etiqueta ordinal por
municipio —el Grado de Rezago Social que publica CONEVAL— y produce un mapa de atención
que señala dónde se concentra la privación dentro de ese municipio. Como el censo publica
el mismo indicador a nivel AGEB, ese mapa se valida contra una verdad de campo que el
modelo nunca vio durante el entrenamiento.

**[Propuesta completa](docs/propuesta.html)** — indicador, diseño experimental, diagrama
de secuencia del proceso, paneles de datos reales, riesgos y plan por fases.

## Instalación

```bash
uv sync --all-groups
```

Python 3.12, fijado en `.python-version`. El entorno se reproduce exacto desde `uv.lock`.

## Uso

La librería se instala con un ejecutable propio:

```bash
uv run satinsight aoi
uv run satinsight probe tuxtla
uv run satinsight panels tuxtla --salida docs/figs -v
```

| Comando | Qué hace |
|---|---|
| `aoi` | Lista los recuadros piloto con su tamaño en píxeles |
| `probe` | Cuenta escenas disponibles y resume la nubosidad sobre un AOI |
| `panels` | Descarga una muestra y renderiza los cuatro paneles de inspección |

Como API:

```python
from satinsight import PILOTO, abrir_catalogo, buscar, compuesto_s1, COLECCION_S1

area = PILOTO["tuxtla"]
escenas = buscar(COLECCION_S1, area.bbox, "2020-01-01/2020-12-31")
sar, meta = compuesto_s1(escenas, area.bbox)
```

## Estructura

```
src/satinsight/
├── aoi.py         recuadros de análisis y sus validaciones
├── catalog.py     consultas al STAC de Planetary Computer
├── raster.py      lectura por ventana de COG remotos y transformaciones
├── composite.py   compuestos mediana anuales de Sentinel-1 y Sentinel-2
├── render.py      paneles PNG y codificación a data URI
└── cli.py         ejecutable satinsight
tests/             pruebas sin red de la lógica pura
docs/              propuesta y paneles generados
```

## Desarrollo

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

Las pruebas no tocan la red: cubren validación de recuadros, transformaciones de
intensidad y agrupación de escenas con dobles de prueba. Lo que sí requiere red se
verifica con `satinsight probe`.

## Estado

Fase 0 cerrada. Propuesta lista, librería reproducible, datos de ilustración descargados.

Lo siguiente es el baseline de la fase 1 — GLCM y gradient boosting a nivel AGEB sobre
tres ciudades piloto, corrido en ambos brazos, que funciona como puerta de decisión del
proyecto.
