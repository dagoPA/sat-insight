#!/bin/sh
# Runs the post-curve queue of the Nature Communications checklist, sequentially.
# Launched detached; each step logs on its own. Order: A4 ceiling, A2 granularity,
# B3 radius sensitivity, B2 baseline extraction (network-bound, so it goes last).
cd "$(dirname "$0")/.." || exit 1
while ! grep -q "THE CURVE" logs/curva.log 2>/dev/null; do sleep 300; done
for s in 0 1 2; do
  for r in 0 1; do
    uv run python herramientas/oraculo.py 6 $r $s expanded > logs/oraculo_exp_r${r}_s${s}.log 2>&1
  done
done
uv run python herramientas/curva_granularidad.py 30 > logs/granularidad.log 2>&1
uv run python herramientas/curva_supervision.py 30 0 full > logs/radio0_pool.log 2>&1
uv run python herramientas/curva_supervision.py 30 2 full > logs/radio2_pool.log 2>&1
uv run python herramientas/baseline_tokens.py > logs/baseline_tokens.log 2>&1
echo "QUEUE DONE" > logs/cola_done.log
