#!/bin/bash
#SBATCH --job-name=attnrct
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/run_%A_%a.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/run_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --signal=B:USR1@300
# #SBATCH --gres=gpu:1     <- uncomment only if `sinfo -o "%P %G"` shows gpu GRES
#
# Submit with an explicit throttle. bigbatch allows 6 running jobs per user, so:
#   sbatch --array=1-120%6 cluster/train_array.slurm
#
# We ask for SIGUSR1 five minutes before walltime so the training script can checkpoint
# before it is killed. Load shedding can still kill a job with no warning, which is why
# the script also checkpoints periodically -- the signal is a courtesy, not a guarantee.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
MANIFEST=$PROJECT/cluster/runs.csv

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}

# Sequence lengths vary every batch, so the caching allocator is asked for a differently
# sized score matrix each time and fragments badly -- the failed pilot runs showed
# 6.5 GB reserved but unallocated. Set for EVERY arm, not just the ones that were
# failing, or the allocator becomes an arm-dependent variable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.bashrc_secrets ] && source ~/.bashrc_secrets

export WANDB_PROJECT=attn-rct-pilot
export WANDB_ENTITY=1903697-wits-university
export WANDB_DIR=$PROJECT/logs
# If the probe found no outbound network from compute nodes:
#   export WANDB_MODE=offline    (then `wandb sync $PROJECT/logs/wandb/*` from login)

# Map array index -> manifest row. We add 1 because row 1 of the file is the header.
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$ROW" ]; then
    echo "FATAL: no manifest row for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi
RUN_ID=$(echo "$ROW" | cut -d, -f1)

CKPT_DIR=$PROJECT/checkpoints/$RUN_ID
RESULT=$PROJECT/results/$RUN_ID.json
mkdir -p "$CKPT_DIR" "$PROJECT/results" "$PROJECT/logs"

# Partial array failures are normal under load shedding, so we make finished tasks
# no-op instead of hand-picking indices to resubmit.
if [ -f "$RESULT" ]; then
    echo "$RUN_ID already complete ($RESULT) -- skipping"
    exit 0
fi

echo "=== $RUN_ID | task $SLURM_ARRAY_TASK_ID | $(hostname) | $(date) ==="
echo "row: $ROW"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

srun python -m attn_rct.train \
    --manifest "$MANIFEST" \
    --row "$SLURM_ARRAY_TASK_ID" \
    --data-dir /datasets/fmnisi/lra/listops \
    --checkpoint-dir "$CKPT_DIR" \
    --result-file "$RESULT" \
    --checkpoint-every-min 30 \
    --require-flash &

wait
echo "=== finished $RUN_ID: $(date) ==="
