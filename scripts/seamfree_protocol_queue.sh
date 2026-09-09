#!/bin/sh
# The complete protocol on the seam-free vectors of both extractors, after their
# cross-validation has finished: the validation-selected core (curve, oracle, saved
# seeds, test column) and then every other experiment, tag by tag.
cd "$(dirname "$0")/.." || exit 1
until [ -f logs/overlap_cv_done.log ]; do sleep 300; done
for tag in ov dofalov; do
  sh scripts/backbone_protocol_queue.sh "s2_$tag"
  sh scripts/backbone_full_queue.sh "$tag"
done
echo "SEAM-FREE PROTOCOL DONE" > logs/seamfree_protocol_done.log
