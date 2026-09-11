#!/bin/sh
# Compositing for the three-country maps: the Colombian and Brazilian catalogue widened
# to every seat above 0.5 km2 and 20,000 urban residents, and every Mexican municipality
# with urban tracts, eight processes on the Planetary Computer, each index walked from both ends. Resumable.
cd "$(dirname "$0")/.." || exit 1
uv run python scripts/transfer_catalogue.py 0.5 20000 > logs/national_catalogue.log 2>&1
for i in 0 1; do
  uv run python scripts/transfer_composites.py $i 2 > "logs/national_transfer_composites_$i.log" 2>&1 &
  uv run python scripts/national_composites.py $i 2 > "logs/national_composites_$i.log" 2>&1 &
  uv run python scripts/transfer_composites.py $i 2 reverse > "logs/national_transfer_composites_${i}_reverse.log" 2>&1 &
  uv run python scripts/national_composites.py $i 2 reverse > "logs/national_composites_${i}_reverse.log" 2>&1 &
done
wait
echo "TRANSFER COMPOSITES DONE" > logs/transfer_composites_done.log
echo "NATIONAL COMPOSITES DONE" > logs/national_composites_done.log
