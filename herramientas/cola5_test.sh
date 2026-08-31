#!/bin/sh
# The single test opening: waits for the degraded test vectors, then scores every row.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f degradar_optico.py >/dev/null; do sleep 120; done
uv run python herramientas/apertura_test.py 30 > logs/apertura_test.log 2>&1
for s in 0 1 2; do
  for r in 0 1; do
    uv run python herramientas/oraculo.py 6 $r $s expanded test > logs/oraculo_test_r${r}_s${s}.log 2>&1
  done
done
uv run python herramientas/replicacion_imu.py data/predicciones_test.parquet > logs/replicacion_test.log 2>&1
uv run python herramientas/focalizacion.py data/predicciones_test.parquet > logs/focalizacion_test.log 2>&1
uv run python herramientas/incumbente_pareado.py data/predicciones_test.parquet > logs/incumbente_test.log 2>&1
uv run python herramientas/maup.py data/predicciones_test.parquet > logs/maup_test.log 2>&1
uv run python herramientas/discontinuidad.py data/predicciones_test.parquet > logs/discontinuidad_test.log 2>&1
echo "TEST OPENED ONCE" > logs/cola5_done.log
