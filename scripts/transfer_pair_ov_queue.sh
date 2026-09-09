#!/bin/sh
# Re-trains the seam-free DOFA-B transfer on exactly the boxes the seam-free DOFA-L run
# saw (more Brazilian seats had been composited in between), so both rest on the same bags.
cd "$(dirname "$0")/.." || exit 1
export SATINSIGHT_BACKBONE=ov
uv run python scripts/transfer_encode.py > logs/transfer_encode_ov2.log 2>&1
uv run python scripts/transfer_bags.py > logs/transfer_bags_ov2.log 2>&1
uv run python scripts/transfer_train.py 40 5 data/transfer_training_dofalov_keys.txt > logs/transfer_train_ov_paired.log 2>&1
echo "TRANSFER PAIR OV DONE" > logs/transfer_pair_ov_done.log
