#!/bin/sh
# Full protocol for one alternative backbone, sequential on the one GPU: the whole
# supervision curve on validation, the oracle on validation and test, the persisted
# validation predictions with saved weights, the test column from those weights, and the
# municipality-paired comparison against the canonical backbone.
cd "$(dirname "$0")/.." || exit 1
sensor=${1:-s2_dofal}
tag=${sensor#s2_}
uv run python scripts/supervision_curve.py 30 1 curve "$sensor" > "logs/backbone_curve_full_$tag.log" 2>&1
for split in val test; do
  for s in 0 1 2; do
    uv run python scripts/oracle.py 6 1 $s expanded $split "" nostd "$sensor" > "logs/backbone_oracle_${tag}_${split}_s${s}.log" 2>&1
  done
done
uv run python scripts/predictions_val.py 30 "$sensor" > "logs/backbone_predictions_$tag.log" 2>&1
uv run python scripts/backbone_test.py "$sensor" > "logs/backbone_test_$tag.log" 2>&1
uv run python scripts/backbone_paired.py "$tag" > "logs/backbone_paired_$tag.log" 2>&1
echo "BACKBONE PROTOCOL DONE $sensor" > "logs/backbone_protocol_${tag}_done.log"
