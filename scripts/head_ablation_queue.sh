#!/bin/sh
# Head ablation for one feature extractor: the CORAL head (shared weights, one bias per
# threshold) and the five-class softmax head trained with cross-entropy, each under the
# protocol of the reference head: three seeds on the full supply with saved weights and
# persisted validation scores, the test column from those weights, and grouped
# cross-validation over the 138 cities. Results carry the head name in their suffix.
# Usage: head_ablation_queue.sh [tag]   (tag ov or dofalov)
cd "$(dirname "$0")/.." || exit 1
tag=${1:-ov}
for head in coral softmax; do
  export SATINSIGHT_BACKBONE=$tag SATINSIGHT_HEAD=$head
  uv run python scripts/predictions_val.py 30 "s2_$tag" > "logs/head_${head}_predictions_$tag.log" 2>&1
  uv run python scripts/backbone_test.py "s2_$tag" > "logs/head_${head}_test_$tag.log" 2>&1
  uv run python scripts/backbone_cv.py "s2_$tag" > "logs/head_${head}_cv_$tag.log" 2>&1
done
echo "HEAD ABLATION DONE $tag" > "logs/head_ablation_${tag}_done.log"
