#!/bin/bash
#SBATCH --job-name=attnrct_smoke
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/smoke_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#
# Plumbing only. stampede is a GTX 1060 (sm_61) and CANNOT run FA-2, so the flash arm
# takes the SDPA fallback here. That is fine: this proves the env loads, the data reads,
# and a checkpoint writes and resumes. It proves NOTHING about efficiency.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
SMOKE_MAX_LEN=512

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct
if [ -f "$HOME/.bashrc_secrets" ]; then
    source "$HOME/.bashrc_secrets"
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "FATAL: WANDB_API_KEY not set and WANDB_MODE is not offline."
    echo "  -> create ~/.bashrc_secrets with: export WANDB_API_KEY=<key>"
    echo "  -> or export WANDB_MODE=offline in this script to log locally."
    exit 1
fi

cd "$PROJECT"

echo "--- unit tests ---"
python -m pytest attn_rct/tests/ -v

# The loader FILTERS sequences longer than max_len, and the generated data is far longer
# than this smoke setting. We check the survivor count first, because an empty dataset
# trains zero batches and the run would otherwise "pass" having done nothing.
echo "--- how much data survives max_len=$SMOKE_MAX_LEN ---"
python - "$SMOKE_MAX_LEN" <<'PY'
import sys
from attn_rct.data import ListOpsDataset, build_vocab, read_tsv

max_len = int(sys.argv[1])
rows = read_tsv("/datasets/fmnisi/lra/listops/basic_train.tsv")
kept = len(ListOpsDataset(rows, build_vocab(rows), max_len))
print(f"{kept} of {len(rows)} train examples survive max_len={max_len}")
if kept < 100:
    print("FATAL: too few examples to smoke-test. Raise --max-len for this job.")
    sys.exit(1)
PY

echo "--- 2-epoch run, checkpoint written ---"
# Tiny settings: stampede has 6GB, and vanilla at the design-space size needs ~4.1GB for
# the score matrix alone. The numbers here are meaningless, which is fine.
python -m attn_rct.train \
    --manifest cluster/runs.csv --row 1 \
    --data-dir /datasets/fmnisi/lra/listops \
    --checkpoint-dir /tmp/smoke_ckpt \
    --result-file /tmp/smoke_result.json \
    --phase smoke \
    --epochs 2 --limit-batches 20 \
    --batch-size 4 --max-len "$SMOKE_MAX_LEN" --d-model 64 --depth 2

echo "--- resume from that checkpoint (must NOT restart from epoch 0) ---"
python -m attn_rct.train \
    --manifest cluster/runs.csv --row 1 \
    --data-dir /datasets/fmnisi/lra/listops \
    --checkpoint-dir /tmp/smoke_ckpt \
    --result-file /tmp/smoke_result.json \
    --phase smoke \
    --epochs 4 --limit-batches 20 \
    --batch-size 4 --max-len "$SMOKE_MAX_LEN" --d-model 64 --depth 2

echo "smoke test done"
