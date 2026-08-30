# Ruta a Nature Communications — lista de pendientes

Objetivo declarado el 30 de agosto de 2026: Nature Communications como destino, Remote
Sensing of Environment como plan B, y el benchmark a NeurIPS Datasets & Benchmarks como
entregable paralelo. Esta lista es el contrato de qué falta; se actualiza tachando, no
borrando. `[~]` marca herramienta escrita y encolada; `[x]` hecho.

## A. Evidencia científica (lo que decide la aceptación)

- [ ] **A1. Curva de cantidad de agregados.** En cadena, arranca sola al terminar la
      expansión. Quince corridas: 5 tamaños × 3 semillas, evaluación fija en las 14 de
      validación. *Puerta: si la curva no es monótona y estable, el titular cambia.*
- [~] **A2. Eje de granularidad de la etiqueta.** Reagregar el GRS a estado y a nacional,
      re-entrenar, medir la pérdida. Herramienta por escribir; corre sobre vectores ya
      extraídos.
- [~] **A3. Eje de resolución.** Re-extraer el brazo óptico degradado a 20×22 m y repetir
      la curva. Confirma o refuta que el fallo del radar es la resolución. Es el eje más
      caro: re-extracción completa de vectores.
- [~] **A4. Techo del oráculo sobre el pool ampliado.** El +0.239 se midió con las 110
      ciudades; con 292 municipios más, el techo y la fracción recuperada se recalculan
      una vez para que la curva tenga denominador propio.
- [~] **A5. Modelo del proceso de agregación.** Peso poblacional, orden de clases y
      redondeo dentro de la pérdida. Capítulo metodológico; si mejora, entra a la curva
      como configuración; si no, se reporta como ablación negativa.
- [~] **A6. Transferencia a Brasil.** Etiquetas descargadas y verificadas: 13,151
      polígonos AGSN 2019 del IBGE, módulo `transfer.py` con columnas fijadas. Falta el
      circuito de imagen (AOI desde las etiquetas, compuestos, vectores) y la puntuación
      zero-shot.
- [~] **A7. Transferencia a Colombia.** Estratificación de Bogotá descargada: estrato 1–6
      por manzana en GPKG oficial del distrito. Mismo faltante que Brasil: el circuito de
      imagen y la puntuación.
- [ ] **A8. Apertura única del conjunto de prueba.** Al final, con configuración y curva
      congeladas. Decisión pendiente del usuario; véase la memoria del 28/08/2026.

## B. Robustez que los revisores van a pedir

- [x] **B1. Incertidumbre honesta en la curva.** Bootstrap agrupado por ciudad sobre las
      14 de validación en cada punto; sin esto la forma de la curva no es defendible.
- [~] **B2. Baseline sin imágenes.** WorldCover + GHSL + percentil NDVI prediciendo por
      tesela, contra el mismo protocolo. La etapa 1 lo insinuó; el paper lo necesita
      formal, porque es la primera objeción obvia.
- [~] **B3. Sensibilidad al radio de contexto sobre el pool ampliado.** El radio 1 ganó
      con 333 bolsas; verificar que se sostiene con ~600.
- [x] **B4. Los 9 municipios fallidos de la expansión.** `data/zonas_fallidas.csv`:
      3 APIError transitorios que el reintento aún puede levantar (El Marqués, Río Grande,
      Zumpango), 5 de profundidad óptica (Amozoc, Comalcalco, Hidalgo del Parral, Lagos de
      Moreno, Nuevo Casas Grandes) y 1 ValueError por revisar (La Independencia). Los de
      profundidad quedan excluidos con criterio explícito: el compuesto no alcanza las 8
      observaciones por píxel que la guarda exige.
- [ ] **B5. MAUP y falacia ecológica.** Ya está en riesgos; el paper necesita el párrafo
      cuantitativo: cómo cambia el Spearman dentro del municipio con el tamaño de la AGEB.

## C. Manuscrito y formato Nature Comms

- [ ] **C1. Conversión a manuscrito.** El documento actual es una propuesta con
      resultados; Nature Comms pide ~5,000 palabras, 6 figuras principales, métodos al
      final, abstract de 150 palabras sin referencias. Reescritura completa en inglés.
- [ ] **C2. Figura 1: la curva.** El tipo de cambio como figura principal, con los tres
      ejes. Las figuras actuales son de trabajo; las de publicación se rehacen desde los
      datos.
- [ ] **C3. Figura de mecanismo.** La disociación predecir/localizar entre modelos y
      sensores en una sola figura.
- [ ] **C4. Mapa de ejemplo.** Predicho contra verdad AGEB en 2–3 ciudades, mismo
      colormap, con el caso de fallo incluido.
- [~] **C5. Data availability statement.** El benchmark congelado resuelve esto: vectores,
      etiquetas, particiones y protocolo en Zenodo con DOI.
- [x] **C6. Code availability.** El repo ya cumple (uv.lock, tests); falta licencia,
      README en inglés para audiencia externa y un script de reproducción por figura.
- [ ] **C7. Reporting summary de Nature.** Formulario obligatorio de estadística; el
      protocolo apareado + bootstrap de B1 lo llena.

## D. Paralelo, sin bloquear el envío

- [ ] **D1. Benchmark a NeurIPS D&B.** Deadline anual ~junio; independiente del paper.
- [ ] **D2. Preprint en arXiv al momento del envío.** Nature Comms lo permite y protege
      la prioridad.

## Orden crítico

A1 → A4 → A2 → B1/B2 (paralelo) → A5 → A3 → A6/A7 → A8 → C todo → envío.
A3 puede correr en paralelo desde ya porque solo compite por disco y GPU con la curva.
