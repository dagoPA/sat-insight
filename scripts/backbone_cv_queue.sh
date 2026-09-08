#!/bin/sh
# Grouped cross-validation over the 138 cities for every backbone, sequential on the one
# GPU, then the pooled comparison with city-clustered bootstrap intervals.
cd "$(dirname "$0")/.." || exit 1
for sensor in s2 s2_dofal s2_cfm; do
  tag=${sensor#s2}
  uv run python scripts/backbone_cv.py "$sensor" 5 30 > "logs/backbone_cv${tag}.log" 2>&1
done
uv run python scripts/backbone_cv_compare.py > logs/backbone_cv_compare.log 2>&1
echo "BACKBONE CV DONE" > logs/backbone_cv_done.log
