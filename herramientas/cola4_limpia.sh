#!/bin/sh
# Re-measures everything the leaky pool contaminated, on the corrected expansion.
cd "$(dirname "$0")/.." || exit 1
uv run python herramientas/curva_supervision.py 30 > logs/curva_v2.log 2>&1
for s in 0 1 2; do
  for r in 0 1; do
    uv run python herramientas/oraculo.py 6 $r $s expanded > logs/oraculo_exp_v2_r${r}_s${s}.log 2>&1
  done
done
uv run python herramientas/curva_granularidad.py 30 > logs/granularidad_v2.log 2>&1
uv run python herramientas/curva_supervision.py 30 0 full > logs/radio0_v2.log 2>&1
uv run python herramientas/curva_supervision.py 30 2 full > logs/radio2_v2.log 2>&1
uv run python herramientas/a5_agregacion.py 30 > logs/a5_v2.log 2>&1
uv run python herramientas/predicciones_val.py 30 > logs/predicciones_v2.log 2>&1
uv run python herramientas/predicciones_ciudad.py > logs/predicciones_ciudad_v2.log 2>&1
uv run python herramientas/transferencia_zeroshot.py > logs/zeroshot_v2.log 2>&1
uv run python herramientas/transferencia_eval.py > logs/transferencia_eval_v2.log 2>&1
uv run python herramientas/replicacion_imu.py > logs/replicacion_v2.log 2>&1
uv run python herramientas/focalizacion.py > logs/focalizacion_v2.log 2>&1
uv run python herramientas/incumbente_pareado.py > logs/incumbente_pareado.log 2>&1
uv run python herramientas/discontinuidad.py data/predicciones_val_ciudad.parquet > logs/discontinuidad_v2.log 2>&1
uv run python herramientas/maup.py > logs/maup_v2.log 2>&1
echo "CLEAN CASCADE DONE" > logs/cola4_done.log
