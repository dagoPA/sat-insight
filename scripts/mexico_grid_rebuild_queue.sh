#!/bin/sh
# Everything that follows the radar composites of 163 Mexican cities being laid again on
# their optical grid: the radar instance tables and vectors of those cities with every
# feature extractor, then the whole protocol of the manuscript from the frozen vectors,
# the three-country scoring and transfer, the canon, and the figures and tables. It is
# reproduce.sh from step 3 on, with the sliding-window queues that the manuscript rests
# on, run in one sequence on one GPU. Waits for the radar recompositing to finish.
cd "$(dirname "$0")/.." || exit 1
export GDAL_HTTP_TIMEOUT=120 GDAL_HTTP_CONNECTTIMEOUT=30
KEYS=${KEYS:-logs/mexico_grid_keys.txt}
CATALOGUE_KEYS=${CATALOGUE_KEYS:-logs/mexico_grid_catalogue_keys.txt}
step() { echo "$(date '+%F %T') $*" >> logs/mexico_grid_rebuild_queue.log; }

until [ "$(cat logs/recomposite_radar_mx_*.log 2>/dev/null | grep -c '^END')" -ge 4 ]; do sleep 120; done
step "radar recomposited"
for marker in backbone_done backbone_cv_done overlap_cv_done seamfree_protocol_done finetune_done \
  head_ablation_ov_done head_ablation_dofalov_done backbone_protocol_dofal_done backbone_protocol_cfm_done \
  backbone_protocol_ov_done backbone_protocol_dofalov_done backbone_full_ov_done backbone_full_dofalov_done \
  backbone_full_dofal_done national_encode_done national_scores_done; do
  rm -f "logs/$marker.log"
done

# 1. tiles and vectors of the rebuilt cities: catalogue cities with the base extractor,
#    the rest of urban Mexico through the national tiler, then every other extractor
uv run python scripts/retile_radar.py $(cat "$CATALOGUE_KEYS") > logs/retile_radar.log 2>&1
uv run python scripts/national_tiles.py > logs/national_tiles_grid.log 2>&1
for name in dofa_large copernicusfm; do
  SATINSIGHT_EXTRACT_SCOPE=catalogue uv run python scripts/backbone_extract.py $name > "logs/backbone_${name}_grid.log" 2>&1
done
for name in dofa_base_ov dofa_large_ov; do
  uv run python scripts/backbone_extract.py $name > "logs/backbone_${name}_grid.log" 2>&1
done
step "vectors rebuilt"

# 2. the reference extraction (non-overlapping DOFA-B): reproduce.sh steps 3 to 5
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
uv run python scripts/predictions_val.py 30 > logs/predictions_val.log 2>&1
uv run python scripts/predictions_city.py > logs/predictions_city.log 2>&1
uv run python scripts/transfer_zeroshot.py > logs/transfer_zeroshot.log 2>&1
uv run python scripts/transfer_eval.py > logs/transfer_eval.log 2>&1
for tool in conapo_replication targeting rwi_paired maup global_products uncertainty; do
  uv run python scripts/$tool.py > "logs/$tool.log" 2>&1
done
uv run python scripts/border_discontinuity.py data/predictions_val_city.parquet > logs/border_discontinuity.log 2>&1
uv run python scripts/test_column.py 30 > logs/test_column.log 2>&1
for s in 0 1 2; do
  for r in 0 1; do
    uv run python scripts/oracle.py 6 $r $s expanded test > logs/oracle_test_r${r}_s${s}.log 2>&1
  done
done
for tool in conapo_replication targeting rwi_paired maup border_discontinuity global_products uncertainty; do
  uv run python scripts/$tool.py data/predictions_test.parquet > "logs/${tool}_test.log" 2>&1
done
uv run python scripts/wc_paired_test.py > logs/wc_paired_test.log 2>&1
step "reference protocol done"

# 3. DOFA large and Copernicus-FM on the reference extraction, and the cross-validation
: > logs/backbone_dofa_large.log; echo "END" >> logs/backbone_dofa_large.log
sh scripts/backbone_queue.sh
for tag in dofal cfm; do sh scripts/backbone_protocol_queue.sh "s2_$tag"; done
sh scripts/backbone_cv_queue.sh
step "reference extractors done"

# 4. the sliding-window extraction the manuscript rests on: cross-validation, then the
#    full protocol of both extractors, which ends with the transfer training
sh scripts/overlap_cv_queue.sh
SKIP_TRANSFER=1 sh scripts/seamfree_protocol_queue.sh
step "sliding-window protocol done"

# 5. partial fine-tuning and the head ablation
uv run python scripts/finetune_cache.py 10 > logs/finetune_cache.log 2>&1
sh scripts/finetune_queue.sh
sh scripts/head_ablation_queue.sh ov
sh scripts/head_ablation_queue.sh dofalov
uv run python scripts/head_ablation.py > logs/head_ablation.log 2>&1
step "ablations done"

# 6. the canon, the headline intervals, and the classification view
# Copernicus-FM has no block of its own in the results file; it enters only the
# cross-validation comparison
for tag in base dofal ov dofalov; do
  uv run python scripts/canon_backbone.py $tag > "logs/canon_$tag.log" 2>&1
done
uv run python scripts/headline_intervals.py > logs/headline_intervals.log 2>&1
uv run python scripts/classification_view.py > logs/classification_view.log 2>&1
step "canon rewritten"

# 7. the three countries: national scores, zero-shot scores abroad, the transfer training
#    on the full supply, the national targeting and the figure
rm -f data/national/*.parquet data/transfer/scores/*.parquet
for tag in ov dofalov; do
  SATINSIGHT_BACKBONE=$tag uv run python scripts/national_scores.py > "logs/national_scores_$tag.log" 2>&1
  SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_scores.py > "logs/transfer_scores_$tag.log" 2>&1
done
sh scripts/transfer_rebuild_queue.sh
uv run python scripts/national_targeting.py > logs/national_targeting.log 2>&1
step "three countries done"

# 8. figures, source data and tables
for fig in fig1_design fig2_curve fig3_dissociation fig4_validation fig5_incumbents fig6_gallery fig7_calibration fig8_transfer fig9_three_countries; do
  uv run python scripts/$fig.py > "logs/$fig.log" 2>&1
done
uv run python scripts/source_data.py > logs/source_data.log 2>&1
uv run python scripts/tables.py > logs/tables.log 2>&1
step "MEXICO GRID REBUILD DONE"
echo "MEXICO GRID REBUILD DONE" > logs/mexico_grid_rebuild_done.log
