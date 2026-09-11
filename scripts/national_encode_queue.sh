#!/bin/sh
# Encoding and bagging behind the three-country maps, one GPU consumer at a time, in
# passes while the composites land: the Mexican municipalities beyond the catalogue are
# tiled and encoded with both feature extractors, and the Colombian and Brazilian boxes
# are encoded and bagged with both. Once compositing is done, a last pass and the
# transfer training on each country's full supply, per extractor. Resumable throughout.
cd "$(dirname "$0")/.." || exit 1
pass() {
  uv run python scripts/national_tiles.py >> logs/national_tiles.log 2>&1
  uv run python scripts/backbone_extract.py dofa_base_ov >> logs/national_extract_ov.log 2>&1
  uv run python scripts/backbone_extract.py dofa_large_ov >> logs/national_extract_dofalov.log 2>&1
  for tag in ov dofalov; do
    SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_encode.py >> "logs/national_transfer_encode_$tag.log" 2>&1
    SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_bags.py >> "logs/national_transfer_bags_$tag.log" 2>&1
  done
}
while [ ! -f logs/national_composites_done.log ]; do
  pass
  sleep 1800
done
pass
for tag in ov dofalov; do
  SATINSIGHT_BACKBONE=$tag uv run python scripts/transfer_train.py 40 5 > "logs/national_transfer_train_$tag.log" 2>&1
done
echo "NATIONAL ENCODING DONE" > logs/national_encode_done.log
