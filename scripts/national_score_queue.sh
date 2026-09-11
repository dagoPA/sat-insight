#!/bin/sh
# Scores every encoded municipality with the saved heads of both feature extractors as
# the vectors land, hourly, and once more when the encoding queue is done. Resumable.
cd "$(dirname "$0")/.." || exit 1
pass() {
  for tag in ov dofalov; do
    SATINSIGHT_BACKBONE=$tag uv run python scripts/national_scores.py >> "logs/national_scores_$tag.log" 2>&1
  done
}
while [ ! -f logs/national_encode_done.log ]; do
  pass
  sleep 3600
done
pass
echo "NATIONAL SCORES DONE" > logs/national_scores_done.log
