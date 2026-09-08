#!/bin/sh
# The complete protocol on one alternative backbone, sequential on the one GPU, so its
# results sit beside the DOFA base ones in every table and figure. Everything reads the
# backbone from the environment and writes suffixed files; nothing of DOFA base is
# touched. The r1 curve, the r1 oracle, the validation predictions with saved weights and
# the test scoring already exist from backbone_protocol_queue.sh and are not redone.
#
# Usage: backbone_full_queue.sh [tag]   (default dofal)
cd "$(dirname "$0")/.." || exit 1
export SATINSIGHT_BACKBONE=${1:-dofal}
tag=$SATINSIGHT_BACKBONE
sensor="s2_$tag"
mkdir -p logs

# vectors of the degraded optical arm with this backbone
uv run python scripts/degrade_optical.py > "logs/degrade_optical_$tag.log" 2>&1

# 3. the other axes of the curve on validation and the radius-0 oracle
uv run python scripts/supervision_curve.py 30 0 full > "logs/radius0_$tag.log" 2>&1
uv run python scripts/supervision_curve.py 30 2 full > "logs/radius2_$tag.log" 2>&1
for s in 0 1 2; do
  for split in val test; do
    uv run python scripts/oracle.py 6 0 $s expanded $split "" nostd "$sensor" > "logs/oracle_r0_${split}_s${s}_$tag.log" 2>&1
  done
done
uv run python scripts/granularity_curve.py 30 > "logs/granularity_curve_$tag.log" 2>&1
uv run python scripts/a5_aggregation.py 30 > "logs/a5_aggregation_$tag.log" 2>&1
for arm in optical degraded radar; do
  for s in 0 1 2; do
    uv run python scripts/llp_val.py 30 1 $s 1 $arm > "logs/llp_val_${arm}_s${s}_$tag.log" 2>&1
  done
done
uv run python scripts/kfold_mil.py 5 30 classes 0 0 > "logs/kfold_mil_$tag.log" 2>&1
uv run python scripts/kfold_llp.py 5 30 0 > "logs/kfold_llp_$tag.log" 2>&1

# 4. analyses on the persisted validation predictions
uv run python scripts/predictions_city.py > "logs/predictions_city_$tag.log" 2>&1
for tool in conapo_replication targeting rwi_paired maup global_products; do
  uv run python scripts/$tool.py "data/predictions_val_$tag.parquet" > "logs/${tool}_$tag.log" 2>&1
done
uv run python scripts/border_discontinuity.py "data/predictions_val_city_$tag.parquet" > "logs/border_discontinuity_$tag.log" 2>&1

# 5. the test column and the analyses on its predictions
uv run python scripts/test_column.py 30 > "logs/test_column_$tag.log" 2>&1
for tool in conapo_replication targeting rwi_paired maup border_discontinuity global_products; do
  uv run python scripts/$tool.py "data/predictions_test_$tag.parquet" > "logs/${tool}_test_$tag.log" 2>&1
done
uv run python scripts/uncertainty.py > "logs/uncertainty_$tag.log" 2>&1
uv run python scripts/wc_paired_test.py > "logs/wc_paired_test_$tag.log" 2>&1

# 6. the hyperparameter sweep, for the selection-signal analysis
uv run python scripts/llp_sweep.py 5 30 1 > "logs/llp_sweep_$tag.log" 2>&1

# 7. transfer: encode every composited box with this backbone, bag it, train
uv run python scripts/transfer_encode.py > "logs/transfer_encode_$tag.log" 2>&1
uv run python scripts/transfer_bags.py > "logs/transfer_bags_$tag.log" 2>&1
uv run python scripts/transfer_train.py 40 5 > "logs/transfer_train_$tag.log" 2>&1

echo "BACKBONE FULL PROTOCOL DONE $tag" > "logs/backbone_full_${tag}_done.log"
