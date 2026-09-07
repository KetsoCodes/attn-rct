#!/bin/bash
#SBATCH --job-name=attnrct
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/run_%A_%a.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/run_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --signal=B:USR1@300

# Submit using an array throttle like `sbatch --array=1-120%6 cluster/train_array.slurm`
# to respect the cluster's concurrent job limit. 
# A SIGUSR1 signal is requested 5 minutes before walltime to trigger a clean checkpoint.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
MANIFEST=$PROJECT/cluster/runs.csv

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}

# Prevents PyTorch CUDA memory fragmentation caused by dynamic sequence lengths across batches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.bashrc_secrets ] && source ~/.bashrc_secrets

export WANDB_PROJECT=attn-rct-pilot
export WANDB_ENTITY=1903697-wits-university
export WANDB_DIR=$PROJECT/logs

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$ROW" ]; then
    echo "FATAL: no manifest row for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi
RUN_ID=$(echo "$ROW" | cut -d, -f1)

CKPT_DIR=$PROJECT/checkpoints/$RUN_ID
RESULT=$PROJECT/results/$RUN_ID.json
mkdir -p "$CKPT_DIR" "$PROJECT/results" "$PROJECT/logs"

# Skips already completed runs to safely handle array resubmissions after cluster load-shedding
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
    --checkpoint-dir "$CKPT_DIR" \
    --result-file "$RESULT" \
    --checkpoint-every-min 30 \
    --require-flash &

wait
echo "=== finished $RUN_ID: $(date) ==="