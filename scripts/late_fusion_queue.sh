#!/bin/sh
# Late fusion of the auxiliary product layers, on validation: GHSL and nightlights alone,
# then with WorldCover. Radius 1, expanded pool, three seeds, as in the canonical point.
cd "$(dirname "$0")/.." || exit 1
uv run python scripts/supervision_curve.py 30 1 full s2 aux late > logs/late_fusion_aux.log 2>&1
uv run python scripts/supervision_curve.py 30 1 full s2 wc,aux late > logs/late_fusion_wc_aux.log 2>&1
echo "LATE FUSION DONE" > logs/late_fusion_done.log
