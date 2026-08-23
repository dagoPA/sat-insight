# sat-insight

Proyecto de investigación: desagregación espacial del rezago social mediante MIL
débilmente supervisado sobre imágenes satelitales. El objetivo final es un paper.

## Idea central

Una bolsa MIL es un municipio. Su etiqueta es un escalar: el Grado de Rezago Social
ordinal de 5 clases que publica CONEVAL. Las instancias son parches de imagen. El modelo
nunca recibe supervisión espacial.

El mapa de atención resultante se valida contra el GRS a nivel AGEB, que se retiene por
completo del entrenamiento. Esa validación cuantitativa del heatmap es la contribución
del proyecto, y sobrevive a cualquier cambio de sensor.

Métricas del heatmap: Spearman entre atención agregada por AGEB y el índice de rezago,
AUC para recuperar AGEBs de grado alto, IoU del top-k de parches.

## Decisiones ya tomadas

Están cerradas. Consultar antes de proponer alternativas.

- **Indicador:** Grado de Rezago Social por AGEB urbana, CONEVAL, censo 2020.
  61,430 AGEBs, 5 clases ordinales, descarga abierta.
- **Corte único de 2020.** Sin series temporales ni detección de cambio. Un compuesto
  mediana anual reduce speckle; el objeto de análisis sigue siendo una imagen estática.
- **Dos brazos en paralelo desde el inicio:** Sentinel-1 RTC (estructura) y Sentinel-2 L2A
  (apariencia), más su fusión. El proyecto no depende de que S1 funcione.
- **Backbone congelado.** Los vectores se extraen una vez y se guardan; el entrenamiento
  del MIL corre sobre ellos. Candidatos: SSL4EO-S12, DOFA.
- **Pérdida ordinal (CORAL)** en lugar de cross-entropy, porque el GRS tiene orden.
- **Transferencia:** Brasil vía aglomerados subnormais del IBGE, Colombia vía estrato
  socioeconómico por manzana del DANE, y el Relative Wealth Index de Meta como puente global.

## Lo que ya se verificó con datos

Sobre el AOI piloto en Tuxtla Gutiérrez (recuadro de 426×309 px a 10 m):

- 72 escenas Sentinel-2 en 2020, el 51% con más del 50% de nubes.
- El compuesto mediana de 36 escenas sale sin rastro de nube. **La nubosidad no justifica
  elegir radar en un producto estático** — se resuelve con compositing.
- 118 escenas Sentinel-1 en 2020. La cobertura sobre el sureste está resuelta.
- El argumento que sí sostiene a S1: gamma0 es una magnitud calibrada, comparable entre
  países sin recalibrar, lo cual importa para la evaluación multipaís.
- Handicap confirmado visualmente: S1 tiene ~20×22 m de resolución real frente a los 10 m
  de S2. La traza urbana se ve notoriamente más basta en radar.

## Plan por fases

1. **Baseline tonto** — GLCM + gradient boosting a nivel AGEB, tres ciudades piloto, ambos
   brazos. *Puerta: si no supera al azar, el MIL tampoco funcionará.*
2. **Pipeline de bolsas** — teselado, extracción de features, partición espacial estricta
   entre train y test sin fuga geográfica entre ciudades.
3. **MIL ordinal y validación del heatmap** — primer cálculo del Spearman atención-vs-AGEB.
   *Puerta: si no correlaciona, la premisa de interpretabilidad cae.*
4. **Ablaciones y transferencia** — barrido de los tres brazos, zero-shot en Brasil y Colombia.

## Veredicto de la fase 1

Cerrada el 23 de agosto de 2026 sobre 138 ciudades, con reparto 80/10/10 por ciudad y el
conjunto de prueba abierto una sola vez.

- **La puerta se pasa.** Kappa cuadrático de 0.544 en prueba con la fusión de ambas
  modalidades, contra 0.318 de un producto externo de densidad construida. AUROC macro
  0.790.
- **La configuración es multimodal.** Elegida en validación, donde la fusión superaba a la
  óptica sola por 0.024 con intervalos solapados; en prueba la ventaja crece a 0.052. La
  decisión fue de diseño y hay que declararla como tal.
- **Prueba coincide con validación** —0.544 contra 0.531— así que no hubo sobreajuste al
  conjunto con el que se eligió.
- **La escala de cuantización pesa más que el sensor.** Fijarla en unidades físicas aporta
  +0.253 al radar y +0.209 al óptico.
- **La textura aporta, y de forma acotada.** Sumarla a la intensidad lleva el kappa óptico
  de 0.440 a 0.507. Pero en el umbral que separa el rezago severo del resto, las tres
  modalidades quedan indistinguibles y la cobertura del suelo sola alcanza 0.828. Por
  importancia de permutación, la fracción construida de WorldCover encabeza la lista por
  delante de cualquier rasgo de imagen.

La consecuencia para el paper: lo que justifica el MIL es el mapa de atención, no la
calidad de la predicción. Si el objetivo fuera predecir el GRS, WorldCover y un percentil
de NDVI ya hacen casi todo el trabajo.

## Decisiones de la fase 2

- **Instancia:** token de 16 px del modelo fundacional, 160 m de lado. Se mantiene.
- **Etiqueta de la bolsa:** las cinco clases del GRS, redondeando la media ponderada por
  población de las AGEB del municipio.
- **Partición:** 110 ciudades de entrenamiento, 14 de validación, 14 de prueba,
  estratificadas por tamaño y por proporción de rezago alto. Ninguna ciudad se divide.
- **Filtro de rasgos:** medido solo sobre las 110 de entrenamiento.

## Riesgos abiertos

- **Atajo por ruralidad.** El rezago correlaciona con lo rural; el modelo puede aprender
  solo densidad construida. Estratificar por tamaño de ciudad y comparar contra un baseline
  que use únicamente densidad GHSL.
- **Bolsas pequeñas.** CLAM opera con miles de instancias por bolsa; un municipio da
  órdenes de magnitud menos. Puede requerir subir la bolsa a zona metropolitana.
- **MAUP y falacia ecológica.** Inherente a etiquetas censales agregadas; se discute
  explícitamente en el paper.

## Estructura del código

Librería instalable con layout `src/`, entorno fijado con `uv.lock`.

```
src/satinsight/
├── aoi.py         recuadros de análisis, validados en __post_init__
├── catalog.py     consultas STAC, resumen de nubosidad, agrupación por órbita
├── raster.py      lectura por ventana de COG remotos, estiramiento, dB
├── composite.py   compuestos mediana anuales de Sentinel-1 y Sentinel-2
├── render.py      paneles PNG y codificación a data URI
└── cli.py         ejecutable satinsight (aoi | probe | panels)
```

Lógica nueva reutilizable va en la librería. Los cuadernos y análisis puntuales la
importan; nunca al revés.

## Convenciones

- **Todo con `uv`.** `uv run ...`, `uv add ...`. Nunca `pip` ni el Python del sistema.
- **Ruff antes de cerrar cualquier cambio:** `uv run ruff format . && uv run ruff check .`
  Línea de 100 caracteres, configuración en `pyproject.toml`.
- **Pruebas sin red.** `tests/` cubre lógica pura con dobles de prueba. Lo que necesita
  red se verifica a mano con `satinsight probe`. Correr con `uv run pytest`.
- Datos satelitales desde el STAC de Microsoft Planetary Computer, colecciones
  `sentinel-1-rtc` y `sentinel-2-l2a`. Lectura por ventana directa del COG con rasterio,
  sin descargar escenas completas.
- Los datos crudos van en `data/`, ignorado por git. El código debe poder regenerarlos.
- **Todo el código en inglés**: identificadores, comentarios, docstrings, mensajes de
  registro y el texto de las figuras. El destino del proyecto es un paper en inglés, y un
  repositorio bilingüe obliga a traducir dos veces cada concepto. Esta conversación y los
  documentos de trabajo siguen en español.
- Quedan quince módulos escritos en español de antes de esta decisión. Se traducen de una
  sola pasada, no a pedazos: un módulo a medio traducir es peor que uno entero en español.

## Git

Los commits van **únicamente a nombre del usuario**. Nunca agregar `Co-Authored-By`,
nunca mencionar asistentes ni modelos en mensajes de commit, PRs o issues.

## Redacción

En prosa y documentos, nunca usar la construcción "no es X sino Y". Escribir solo la
afirmación directa. Evitar `no es`, `, no `, `sino`, `en vez de`, `en lugar de` como
recurso retórico. Una comparación real donde ambos lados aportan información —una tabla de
tradeoffs, dos opciones que se están pesando— sí es válida.
