#!/bin/sh
# Backbone ablation, sequential on the one GPU: Copernicus-FM extraction once the DOFA
# large pass has finished, then the canonical full-pool point (radius 1, three seeds)
# for each backbone on validation, and the held-out test column for each.
cd "$(dirname "$0")/.." || exit 1
until grep -q "^END" logs/backbone_dofa_large.log 2>/dev/null; do sleep 120; done
uv run python scripts/backbone_extract.py copernicusfm > logs/backbone_copernicusfm.log 2>&1
for tag in dofal cfm; do
  uv run python scripts/supervision_curve.py 30 1 full s2_$tag > "logs/backbone_curve_$tag.log" 2>&1
done
echo "BACKBONE QUEUE DONE" > logs/backbone_done.log
