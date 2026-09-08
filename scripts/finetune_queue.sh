#!/bin/sh
# Partial fine-tuning of the base backbone: three seeds at the conservative backbone
# learning rate, then three at a ten-times larger one, sequential on the one GPU.
cd "$(dirname "$0")/.." || exit 1
for lr in 1e-5 1e-4; do
  for s in 0 1 2; do
    uv run python scripts/finetune_train.py 30 $s 10 $lr > "logs/finetune_b10_lr${lr}_s${s}.log" 2>&1
    mv "data/finetune_ft_b10_s${s}.csv" "data/finetune_ft_b10_lr${lr}_s${s}.csv"
    mv "data/predictions_ft_b10_s${s}.parquet" "data/predictions_ft_b10_lr${lr}_s${s}.parquet"
    mv "data/weights/llp_final_ft_b10_s${s}.pt" "data/weights/llp_final_ft_b10_lr${lr}_s${s}.pt"
  done
done
echo "FINETUNE DONE" > logs/finetune_done.log
