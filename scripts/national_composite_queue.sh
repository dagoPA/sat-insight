#!/bin/sh
# Compositing for the three-country maps: the Colombian and Brazilian catalogue widened
# to every seat above 0.5 km2 and 20,000 urban residents, and every Mexican municipality
# with urban tracts, eight processes on the Planetary Computer, each index walked from
# both ends. Resumable, and each process is supervised twice over.
#
# A remote tile that fails to read raises through the compositing and kills the run, so a
# process that stops without printing its END line is started again, which costs seconds
# because every composite already on disk is skipped.
#
# An outage of the catalogue service is worse than a crash: every box in turn fails on a
# connection error, the run reaches its END line having composited nothing, and the list
# is consumed. So the queue waits for the service before each attempt, and a run that
# ends with more than a handful of API failures is treated as interrupted and run again.
cd "$(dirname "$0")/.." || exit 1
ATTEMPTS=${ATTEMPTS:-40}
API_TOLERANCE=${API_TOLERANCE:-20}
STAC=https://planetarycomputer.microsoft.com/api/stac/v1

wait_for_service() {
  waited=0
  while [ "$waited" -lt 288 ]; do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$STAC")" = "200" ]; then
      return 0
    fi
    waited=$((waited + 1))
    sleep 300
  done
  return 1
}

api_failures() {
  grep -c "APIError" "$1" 2>/dev/null || echo 0
}

supervise() {
  log=$1
  shift
  attempt=0
  while [ "$attempt" -lt "$ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    wait_for_service || { echo "GAVE UP waiting for the catalogue service" >>"$log"; return 1; }
    before=$(api_failures "$log")
    uv run python "$@" >>"$log" 2>&1
    after=$(api_failures "$log")
    if tail -20 "$log" | grep -q "^END" && [ $((after - before)) -le "$API_TOLERANCE" ]; then
      return 0
    fi
    echo "RESTART $attempt after an interrupted run" >>"$log"
    sleep 60
  done
  echo "GAVE UP after $ATTEMPTS attempts" >>"$log"
  return 1
}

wait_for_service || exit 1
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
