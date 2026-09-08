#!/bin/sh
# Headline check of the overlapping-window vectors: validation-selected seeds with saved
# weights, the test column from them, and the paired comparison against the reference.
cd "$(dirname "$0")/.." || exit 1
tag=${1:-ov}
uv run python scripts/predictions_val.py 30 "s2_$tag" > "logs/predictions_val_$tag.log" 2>&1
uv run python scripts/backbone_test.py "s2_$tag" > "logs/backbone_test_$tag.log" 2>&1
uv run python scripts/backbone_paired.py "$tag" > "logs/backbone_paired_$tag.log" 2>&1
echo "OVERLAP CHECK DONE $tag" > "logs/overlap_check_${tag}_done.log"
