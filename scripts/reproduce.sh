#!/bin/sh
# Canonical sequence behind the manuscript, from composited cities to the figures.
# Every step logs on its own under logs/. Steps 1 and 2 need the satellite pipeline
# (days of downloading); everything from step 3 on runs from the frozen vectors and
# reproduces the canonical results file in a few hours on one GPU.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

# 1. composites, tiles, bags and frozen DOFA vectors of the national set and the expansion
uv run python scripts/composite_cities.py > logs/composite_cities.log 2>&1
uv run satinsight bags s2 && uv run satinsight bags s1
uv run satinsight partition
uv run satinsight vectors s2 && uv run satinsight vectors s1
uv run python scripts/extra_zones.py > logs/extra_zones.log 2>&1
uv run python scripts/baseline_tokens.py > logs/baseline_tokens.log 2>&1
uv run python scripts/degrade_optical.py > logs/degrade_optical.log 2>&1

# 2. transfer cities and the auxiliary product layers
uv run python scripts/transfer_composites.py > logs/transfer_composites.log 2>&1
uv run python scripts/aux_tokens.py > logs/aux_tokens.log 2>&1

# 3. supervision-efficiency curve on validation, its axes, and the oracle upper bound
uv run python scripts/supervision_curve.py 30 > logs/supervision_curve.log 2>&1
for s in 0 1 2; do
  for r in 0 1; do
    uv run python scripts/oracle.py 6 $r $s expanded > logs/oracle_r${r}_s${s}.log 2>&1
  done
done
uv run python scripts/granularity_curve.py 30 > logs/granularity_curve.log 2>&1
uv run python scripts/supervision_curve.py 30 0 full > logs/radius0.log 2>&1
uv run python scripts/supervision_curve.py 30 2 full > logs/radius2.log 2>&1
uv run python scripts/a5_aggregation.py 30 > logs/a5_aggregation.log 2>&1

# 4. persisted predictions on validation and the analyses that consume them
uv run python scripts/predictions_val.py 30 > logs/predictions_val.log 2>&1
uv run python scripts/predictions_city.py > logs/predictions_city.log 2>&1
uv run python scripts/transfer_zeroshot.py > logs/transfer_zeroshot.log 2>&1
uv run python scripts/transfer_eval.py > logs/transfer_eval.log 2>&1
uv run python scripts/conapo_replication.py > logs/conapo_replication.log 2>&1
uv run python scripts/targeting.py > logs/targeting.log 2>&1
uv run python scripts/rwi_paired.py > logs/rwi_paired.log 2>&1
uv run python scripts/border_discontinuity.py data/predictions_val_city.parquet > logs/border_discontinuity.log 2>&1
uv run python scripts/maup.py > logs/maup.log 2>&1
uv run python scripts/global_products.py > logs/global_products.log 2>&1
uv run python scripts/uncertainty.py > logs/uncertainty.log 2>&1

# 5. the held-out test column, scored with the validation-selected configurations
uv run python scripts/test_column.py 30 > logs/test_column.log 2>&1
for s in 0 1 2; do
  for r in 0 1; do
    uv run python scripts/oracle.py 6 $r $s expanded test > logs/oracle_test_r${r}_s${s}.log 2>&1
  done
done
for tool in conapo_replication targeting rwi_paired maup border_discontinuity global_products uncertainty; do
  uv run python scripts/$tool.py data/predictions_test.parquet > logs/${tool}_test.log 2>&1
done
uv run python scripts/wc_paired_test.py > logs/wc_paired_test.log 2>&1

# 5b. the backbone ablation: DOFA large and Copernicus-FM vectors through the same protocol,
#     then grouped cross-validation over all 138 cities with bootstrap intervals
uv run python scripts/backbone_extract.py dofa_large > logs/backbone_dofa_large.log 2>&1
sh scripts/backbone_queue.sh
sh scripts/backbone_protocol_queue.sh s2_dofal
sh scripts/backbone_cv_queue.sh   # grouped cross-validation of the three backbones and the pooled comparison

# 6. figures and source data, each checked against docs/manuscript/canonical_results.json
for fig in fig1_design fig2_curve fig3_dissociation fig4_validation fig5_incumbents; do
  uv run python scripts/$fig.py > logs/$fig.log 2>&1
done
uv run python scripts/source_data.py > logs/source_data.log 2>&1
echo "REPRODUCTION DONE" > logs/reproduce_done.log
