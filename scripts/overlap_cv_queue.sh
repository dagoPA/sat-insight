#!/bin/sh
# Seam-free vectors under the standard comparison: extract DOFA-L with the central rule,
# then grouped cross-validation over the 138 cities for both seam-free extractors, and the
# pooled comparison of all tags that have a cross-validation.
cd "$(dirname "$0")/.." || exit 1
uv run python scripts/backbone_extract.py dofa_large_ov > logs/backbone_dofa_large_ov.log 2>&1
for sensor in s2_ov s2_dofalov; do
  tag=${sensor#s2}
  uv run python scripts/backbone_cv.py "$sensor" 5 30 > "logs/backbone_cv${tag}.log" 2>&1
done
uv run python scripts/backbone_cv_compare.py base dofal ov dofalov > logs/backbone_cv_compare_ov.log 2>&1
echo "OVERLAP CV DONE" > logs/overlap_cv_done.log
