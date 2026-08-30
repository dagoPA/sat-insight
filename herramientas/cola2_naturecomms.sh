#!/bin/sh
# Second queue: resolution axis (A3) and aggregation model (A5), after queue 1 frees the GPU.
cd "$(dirname "$0")/.." || exit 1
while [ ! -f logs/cola_done.log ]; do sleep 300; done
uv run python herramientas/degradar_optico.py > logs/degradar.log 2>&1
for s in 0 1 2; do
  uv run python herramientas/llp_val.py 30 1 $s 1 degraded > logs/llp_val_degraded_s${s}.log 2>&1
done
uv run python herramientas/a5_agregacion.py 30 > logs/a5.log 2>&1
echo "QUEUE 2 DONE" > logs/cola2_done.log
