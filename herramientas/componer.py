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

from satinsight.agebs import cities_by_size  # noqa: E402
from satinsight.pipeline import city_aoi, ensure_composite  # noqa: E402

catalogue = cities_by_size(stratify=True)
argumentos = sys.argv[1:]
if argumentos and argumentos[0].isdigit():
    indice, total = int(argumentos[0]), int(argumentos[1])
    mias = list(catalogue)[indice::total]
    etiqueta = f"proceso {indice}"
else:
    mias = argumentos or list(catalogue)
    etiqueta = "corrida"

print(f"{etiqueta}: {len(mias)} cities", flush=True)
fallidas = []
for n, clave in enumerate(mias, start=1):
    inicio = time.time()
    try:
        area, agebs = city_aoi(clave, catalogue=catalogue)
        for sensor in ("s2", "s1"):
            ensure_composite(clave, sensor, area=area)
        minutos = (time.time() - inicio) / 60
        print(f"OK {clave} ({n}/{len(mias)}) {len(agebs)} AGEB en {minutos:.1f} min", flush=True)
    except Exception as e:
        fallidas.append(clave)
        print(f"FALLO {clave} ({n}/{len(mias)}): {type(e).__name__}: {e}", flush=True)

print(f"FIN {etiqueta}: {len(fallidas)} fallidas {fallidas}", flush=True)
# el código de salida distingue la corrida completa de la que dejó cities atrás, para
# que un bucle de reintento sepa si volver a llamarla
sys.exit(1 if fallidas else 0)
