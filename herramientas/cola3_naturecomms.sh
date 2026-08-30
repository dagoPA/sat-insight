#!/bin/sh
# Third queue: the adopted slate's compute, after queue 2 frees the GPU.
cd "$(dirname "$0")/.." || exit 1
while [ ! -f logs/cola2_done.log ]; do sleep 300; done
uv run python herramientas/predicciones_val.py 30 > logs/predicciones.log 2>&1
uv run python herramientas/replicacion_imu.py > logs/replicacion_imu.log 2>&1
uv run python herramientas/focalizacion.py > logs/focalizacion.log 2>&1
echo "QUEUE 3 DONE" > logs/cola3_done.log
