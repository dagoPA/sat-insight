#!/bin/sh
# Rebuilds the transfer bags of every catalogued box with the seat's own municipality
# always inside (half of Colombia's boxes came out empty before that rule), then the
# zero-shot scores and the transfer training on each country's full supply, for both
# feature extractors. The late composites that landed after the last encoding pass are
# encoded first. Every step is resumable, so each one runs under a watchdog: a remote
# read that hangs without a timeout leaves the process idle, and a step whose log has not
# moved for a while is killed and run again from where it stopped. The bags are wiped
# once, on the first run, and a marker keeps a restart from wiping them again.
cd "$(dirname "$0")/.." || exit 1
export GDAL_HTTP_TIMEOUT=120 GDAL_HTTP_CONNECTTIMEOUT=30 GDAL_HTTP_MAX_RETRY=3 GDAL_HTTP_RETRY_DELAY=5
STALL=${STALL:-900}
TAGS=${TAGS:-"ov dofalov"}
# the extractors to run, so that a finished one is skipped when the queue is relaunched
MARKER=logs/transfer_rebuild_started

watched() {
  log=$1
  shift
  attempt=0
  while [ "$attempt" -lt 30 ]; do
    attempt=$((attempt + 1))
    uv run python "$@" >>"$log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      age=$(( $(date +%s) - $(stat -f %m "$log") ))
      if [ "$age" -gt "$STALL" ]; then
        echo "STALLED for ${age}s; killing and resuming (attempt $attempt)" >>"$log"
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
        sleep 5
        break
      fi
    done
    if tail -5 "$log" | grep -q "^END"; then
      return 0
    fi
  done
  echo "GAVE UP after $attempt attempts" >>"$log"
  return 1
}

if [ ! -f "$MARKER" ]; then
  rm -f data/transfer/bags_*_ov.parquet data/transfer/bags_*_dofalov.parquet
  rm -f data/transfer/scores/*_ov.parquet data/transfer/scores/*_dofalov.parquet
  date > "$MARKER"
fi
for tag in $TAGS; do
  export SATINSIGHT_BACKBONE=$tag
  watched "logs/national_transfer_encode_$tag.log" scripts/transfer_encode.py
  watched "logs/transfer_rebag_$tag.log" scripts/transfer_bags.py
  watched "logs/transfer_scores_$tag.log" scripts/transfer_scores.py
done
for tag in $TAGS; do
  export SATINSIGHT_BACKBONE=$tag
  : > "logs/national_transfer_train_$tag.log"
  # an epoch over thousands of bags logs nothing for a long while; the training stall is
  # measured in hours, and a restart repeats the run rather than losing it
  STALL=7200 watched "logs/national_transfer_train_$tag.log" scripts/transfer_train.py 40 5
done
echo "NATIONAL ENCODING DONE" > logs/national_encode_done.log
