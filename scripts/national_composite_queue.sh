#!/bin/sh
# Compositing for the three-country maps: the Colombian and Brazilian catalogue widened
# to every seat above 0.5 km2 and 20,000 urban residents, and every Mexican municipality
# with urban tracts, eight processes on the Planetary Computer, each index walked from
# both ends. Resumable, and each process is supervised: a remote tile that fails to read
# raises through the compositing and kills the run, so a process that stops without
# printing its END line is started again, which costs seconds because every composite
# already on disk is skipped.
cd "$(dirname "$0")/.." || exit 1
ATTEMPTS=${ATTEMPTS:-40}

supervise() {
  log=$1
  shift
  attempt=0
  while [ "$attempt" -lt "$ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    uv run python "$@" >>"$log" 2>&1
    if tail -20 "$log" | grep -q "^END"; then
      return 0
    fi
    echo "RESTART $attempt after an interrupted run" >>"$log"
    sleep 30
  done
  echo "GAVE UP after $ATTEMPTS attempts" >>"$log"
  return 1
}

uv run python scripts/transfer_catalogue.py 0.5 20000 > logs/national_catalogue.log 2>&1
for i in 0 1; do
  supervise "logs/national_transfer_composites_$i.log" scripts/transfer_composites.py "$i" 2 &
  supervise "logs/national_composites_$i.log" scripts/national_composites.py "$i" 2 &
  supervise "logs/national_transfer_composites_${i}_reverse.log" scripts/transfer_composites.py "$i" 2 reverse &
  supervise "logs/national_composites_${i}_reverse.log" scripts/national_composites.py "$i" 2 reverse &
done
wait
echo "TRANSFER COMPOSITES DONE" > logs/transfer_composites_done.log
echo "NATIONAL COMPOSITES DONE" > logs/national_composites_done.log
