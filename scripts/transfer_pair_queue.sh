#!/bin/sh
# After the DOFA large protocol has trained the transfer rows on the boxes encoded so far,
# trains DOFA base on exactly those boxes, so the two backbones' transfer tables rest on
# the same bags. The key set is the one the DOFA large run wrote beside its results.
cd "$(dirname "$0")/.." || exit 1
until [ -f logs/backbone_full_dofal_done.log ]; do sleep 120; done
uv run python scripts/transfer_train.py 40 5 data/transfer_training_dofal_keys.txt > logs/transfer_train_paired_base.log 2>&1
echo "TRANSFER PAIR DONE" > logs/transfer_pair_done.log
