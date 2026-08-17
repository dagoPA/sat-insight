# sat-insight

Desagregación del rezago social a partir de imágenes satelitales, con aprendizaje
débilmente supervisado.

Un modelo MIL con atención, en la línea de CLAM, recibe una sola etiqueta ordinal por
municipio —el Grado de Rezago Social que publica CONEVAL— y produce un mapa de atención
que señala dónde se concentra la privación dentro de ese municipio. Como el censo publica
el mismo indicador a nivel AGEB, ese mapa se valida contra una verdad de campo que el
modelo nunca vio durante el entrenamiento.

**[Propuesta completa](docs/propuesta.html)** — indicador, diseño experimental, diagrama
de secuencia del proceso, paneles de datos reales, riesgos y plan por fases.

## Entorno

```bash
uv sync
```

Requiere Python 3.12, fijado en `.python-version`. El Python del sistema queda intacto.

## Scripts

| Script | Qué hace |
|---|---|
| `scripts/probe_stac.py` | Sondea disponibilidad de Sentinel-1 y Sentinel-2 sobre el AOI piloto |
| `scripts/build_figure_data.py` | Descarga una muestra mínima y renderiza los cuatro paneles de datos |
| `scripts/embed_figs.py` | Incrusta esos paneles en la propuesta como data URIs |

```bash
uv run python scripts/probe_stac.py
```

## Estado

Fase 0. Propuesta cerrada, entorno montado, datos de ilustración descargados.
Lo siguiente es el baseline de la fase 1 — GLCM + gradient boosting a nivel AGEB sobre
tres ciudades piloto, corrido en ambos brazos, que funciona como puerta de decisión del
proyecto.
