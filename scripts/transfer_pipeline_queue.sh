#!/bin/sh
# Encodes and bags the catalogued seats as their composites land, then trains the
# transfer curves once compositing is done. Resumable: each tool skips what exists.
cd "$(dirname "$0")/.." || exit 1
while [ ! -f logs/transfer_composites_done.log ]; do
  uv run python scripts/transfer_encode.py >> logs/transfer_encode_queue.log 2>&1
  uv run python scripts/transfer_bags.py >> logs/transfer_bags_queue.log 2>&1
  sleep 1800
done
uv run python scripts/transfer_encode.py >> logs/transfer_encode_queue.log 2>&1
uv run python scripts/transfer_bags.py >> logs/transfer_bags_queue.log 2>&1
uv run python scripts/transfer_train.py 40 5 > logs/transfer_train_full.log 2>&1
echo "TRANSFER PIPELINE DONE" > logs/transfer_pipeline_done.log
