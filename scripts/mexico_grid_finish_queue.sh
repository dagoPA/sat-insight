#!/bin/sh
# The tail of the Mexico grid rebuild: the transfer training on each country's full
# supply for both feature extractors, the national targeting, and the figures, source
# data and tables. Separate from the rebuild queue because a running shell script must
# never be edited: the interpreter resumes at a byte offset and executes a fragment, which
# is how a run once lost its extractor variable and wrote a table mixing both.
cd "$(dirname "$0")/.." || exit 1
step() { echo "$(date '+%F %T') $*" >> logs/mexico_grid_rebuild_queue.log; }

sh scripts/transfer_rebuild_queue.sh
step "transfer training done"
uv run python scripts/national_targeting.py > logs/national_targeting.log 2>&1
step "national targeting done"
for fig in fig1_design fig2_curve fig3_dissociation fig4_validation fig5_incumbents fig6_gallery fig7_calibration fig8_transfer fig9_three_countries; do
  uv run python scripts/$fig.py > "logs/$fig.log" 2>&1
done
uv run python scripts/source_data.py > logs/source_data.log 2>&1
uv run python scripts/tables.py > logs/tables.log 2>&1
uv run python scripts/head_ablation.py > logs/head_ablation.log 2>&1
uv run python scripts/headline_intervals.py > logs/headline_intervals.log 2>&1
uv run python scripts/classification_view.py > logs/classification_view.log 2>&1
step "MEXICO GRID REBUILD DONE"
echo "MEXICO GRID REBUILD DONE" > logs/mexico_grid_rebuild_done.log
