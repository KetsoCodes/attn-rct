#!/bin/bash
#SBATCH --job-name=attnrct_precision
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/precision_%A_%a.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/precision_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --array=1-4%4
#SBATCH --exclude=mscluster44,mscluster45,mscluster47,mscluster51,mscluster57,mscluster58,mscluster61,mscluster62,mscluster64,mscluster66,mscluster67,mscluster68,mscluster69,mscluster70,mscluster74,mscluster78,mscluster83
#
# Is the linear arm's accuracy advantage actually about the mechanism, or is it just a bf16 artifact?
#

set -uo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
MANIFEST=$PROJECT/cluster/runs.csv

. "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.bashrc_secrets ] && . ~/.bashrc_secrets
export WANDB_PROJECT=attn-rct-pilot
export WANDB_ENTITY=1903697-wits-university
export WANDB_DIR=$PROJECT/logs

cd "$PROJECT"
mkdir -p logs results checkpoints

# Index 0 unused so the array index maps straight onto my test cells.
ROWS=(0 1 1 10 10)
DTYPES=(unused bf16 fp32 bf16 fp32)
NAMES=(unused vanilla_bf16 vanilla_fp32 linear_bf16 linear_fp32)

ROW=${ROWS[$SLURM_ARRAY_TASK_ID]}
DTYPE=${DTYPES[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
TAG="precision_${NAME}"

echo "=== $TAG | row $ROW | dtype $DTYPE | $(hostname) | $(date) ==="

# Pre-flight check to catch dead GPUs before they ruin my cell.
gpu_ok=1
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || gpu_ok=0
if [ "$gpu_ok" = "1" ]; then
    python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda')" 2>/dev/null || gpu_ok=0
fi
if [ "$gpu_ok" = "0" ]; then
    echo "GPU pre-flight FAILED on $(hostname); requeuing."
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Force a fresh start per cell. If I resume a checkpoint here, it carries the optimizer
# state from the other dtype and ruins the comparison.
rm -rf "checkpoints/$TAG"

srun python -u -m attn_rct.train \
    --manifest "$MANIFEST" \
    --row "$ROW" \
    --checkpoint-dir "$PROJECT/checkpoints/$TAG" \
    --result-file "$PROJECT/results/$TAG.json" \
    --checkpoint-every-min 20 \
    --compute-dtype "$DTYPE" \
    --phase smoke &

wait
echo "=== finished $TAG: $(date) ==="