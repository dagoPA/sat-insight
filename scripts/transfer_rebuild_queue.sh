#!/bin/sh
# Rebuilds the transfer bags of every catalogued box with the seat's own municipality
# always inside (half of Colombia's boxes came out empty before that rule), then the
# zero-shot scores and the transfer training on each country's full supply, for both
# feature extractors. The late composites that landed after the last encoding pass are
# encoded first. Resumable except for the bags, which are rebuilt on purpose.
cd "$(dirname "$0")/.." || exit 1
rm -f data/transfer/bags_*_ov.parquet data/transfer/bags_*_dofalov.parquet
rm -f data/transfer/scores/*_ov.parquet data/transfer/scores/*_dofalov.parquet
for tag in ov dofalov; do
  SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_encode.py >> "logs/national_transfer_encode_$tag.log" 2>&1
  SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_bags.py > "logs/transfer_rebag_$tag.log" 2>&1
  SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_scores.py > "logs/transfer_scores_$tag.log" 2>&1
done
for tag in ov dofalov; do
  SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_train.py 40 5 > "logs/national_transfer_train_$tag.log" 2>&1
done
echo "NATIONAL ENCODING DONE" > logs/national_encode_done.log
