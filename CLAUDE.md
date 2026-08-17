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

## Riesgos abiertos

- **Atajo por ruralidad.** El rezago correlaciona con lo rural; el modelo puede aprender
  solo densidad construida. Estratificar por tamaño de ciudad y comparar contra un baseline
  que use únicamente densidad GHSL.
- **Bolsas pequeñas.** CLAM opera con miles de instancias por bolsa; un municipio da
  órdenes de magnitud menos. Puede requerir subir la bolsa a zona metropolitana.
- **MAUP y falacia ecológica.** Inherente a etiquetas censales agregadas; se discute
  explícitamente en el paper.

## Convenciones

- **Todo con `uv`.** `uv run python ...`, `uv add ...`. Nunca `pip` ni el Python del sistema.
- Datos satelitales desde el STAC de Microsoft Planetary Computer, colecciones
  `sentinel-1-rtc` y `sentinel-2-l2a`. Lectura por ventana directa del COG con rasterio,
  sin descargar escenas completas.
- Los datos crudos van en `data/`, ignorado por git. Los scripts deben poder regenerarlos.
- Código y comentarios en español, como el resto del proyecto.

## Redacción

En prosa y documentos, nunca usar la construcción "no es X sino Y". Escribir solo la
afirmación directa. Evitar `no es`, `, no `, `sino`, `en vez de`, `en lugar de` como
recurso retórico. Una comparación real donde ambos lados aportan información —una tabla de
tradeoffs, dos opciones que se están pesando— sí es válida.
