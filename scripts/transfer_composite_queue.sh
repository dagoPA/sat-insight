#!/bin/sh
# Composites every catalogued seat of Colombia and Brazil, resumably: the catalogue is
# rebuilt on each pass so states whose tract geometries finished downloading join in,
# and ensure_composite skips boxes already on disk. Loops until a pass adds nothing.
cd "$(dirname "$0")/.." || exit 1
pass=0
while [ $pass -lt 6 ]; do
  pass=$((pass + 1))
  uv run python scripts/transfer_catalogue.py 1.5 50000 > "logs/transfer_catalogue_$pass.log" 2>&1
  uv run python scripts/transfer_composites.py > "logs/transfer_composites_$pass.log" 2>&1
  if ! grep -q "^FAIL" "logs/transfer_composites_$pass.log" && ! grep -q "skipped" "logs/transfer_catalogue_$pass.log"; then
    break
  fi
  sleep 600
done
echo "TRANSFER COMPOSITES DONE" > logs/transfer_composites_done.log
