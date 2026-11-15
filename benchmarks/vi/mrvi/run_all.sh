#!/bin/bash
# Sequential (one GPU process at a time) ours-vs-scvi-tools MrVI validation, 3 seeds.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
for seed in 0 1 2; do
  echo "=== ours seed=$seed ==="
  python run_ours.py "$seed" "run_ours_s${seed}" cuda
  echo "=== ref seed=$seed ==="
  python run_ref.py "$seed" "run_ref_s${seed}" cuda
done
echo "=== compare ==="
python compare.py run > ../results_mrvi_table.md
cat ../results_mrvi_table.md
