#!/bin/sh
# Copernicus-FM through the validation-selected protocol once the paired transfer run
# has released the GPU: full-pool seeds with saved weights, the test column from those
# weights, the oracle on both splits, and the municipality-paired comparison.
cd "$(dirname "$0")/.." || exit 1
until [ -f logs/transfer_pair_done.log ]; do sleep 120; done
sh scripts/backbone_protocol_queue.sh s2_cfm
