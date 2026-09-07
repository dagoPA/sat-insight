#!/bin/sh
# Resumes the backbone protocol after the curve and the oracle: predictions, test, paired.
cd "$(dirname "$0")/.." || exit 1
sensor=${1:-s2_dofal}
tag=${sensor#s2_}
uv run python scripts/predictions_val.py 30 "$sensor" > "logs/backbone_predictions_$tag.log" 2>&1
uv run python scripts/backbone_test.py "$sensor" > "logs/backbone_test_$tag.log" 2>&1
uv run python scripts/backbone_paired.py "$tag" > "logs/backbone_paired_$tag.log" 2>&1
echo "BACKBONE PROTOCOL DONE $sensor" > "logs/backbone_protocol_${tag}_done.log"
