"""Compone una porción del conjunto nacional. Uso: componer.py [indice total] [ciudad ...]

Vive en el repositorio y no en un directorio temporal: la corrida dura días y un
scratchpad que se limpia deja el trabajo fallando en silencio contra un archivo que
ya no existe.
"""

import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
)

from satinsight.agebs import ciudades_por_tamano  # noqa: E402
from satinsight.pipeline import aoi_de_ciudad, asegurar_compuesto  # noqa: E402

catalogo = ciudades_por_tamano(estratificar=True)
argumentos = sys.argv[1:]
if argumentos and argumentos[0].isdigit():
    indice, total = int(argumentos[0]), int(argumentos[1])
    mias = list(catalogo)[indice::total]
    etiqueta = f"proceso {indice}"
else:
    mias = argumentos or list(catalogo)
    etiqueta = "corrida"

print(f"{etiqueta}: {len(mias)} ciudades", flush=True)
fallidas = []
for n, clave in enumerate(mias, start=1):
    inicio = time.time()
    try:
        area, agebs = aoi_de_ciudad(clave, catalogo=catalogo)
        for sensor in ("s2", "s1"):
            asegurar_compuesto(clave, sensor, area=area)
        minutos = (time.time() - inicio) / 60
        print(f"OK {clave} ({n}/{len(mias)}) {len(agebs)} AGEB en {minutos:.1f} min", flush=True)
    except Exception as e:
        fallidas.append(clave)
        print(f"FALLO {clave} ({n}/{len(mias)}): {type(e).__name__}: {e}", flush=True)

print(f"FIN {etiqueta}: {len(fallidas)} fallidas {fallidas}", flush=True)
# el código de salida distingue la corrida completa de la que dejó ciudades atrás, para
# que un bucle de reintento sepa si volver a llamarla
sys.exit(1 if fallidas else 0)
