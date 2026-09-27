#!/bin/bash
#SBATCH --job-name=attnrct_pfprobe
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/pfprobe_%A_%a.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/pfprobe_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --array=1-4%4
#
# Diagnostic probe to determine if Pathfinder requires learning-rate warmup.
# 
# Initial grid results showed all arms stuck at chance (0.50). This probe tests 
# whether the task is genuinely unlearnable within our budget, or if it simply 
# requires the LR warmup standard in LRA literature to prevent early collapse.
#
# Runs ONE design/arm (d000, flash) under 4 recipes varying warmup and LR.
# Executes under --phase smoke to prevent contaminating the main analysis.
#
# Expected outcomes:
#   * Any recipe climbs > 0.50: Frame requires warmup. Rerun full Pathfinder grid.
#   * All recipes flat at 0.50: Task unlearnable. Report null, keep efficiency data.

set -uo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
MANIFEST=$PROJECT/cluster/runs.csv

# Row 244 maps exactly to pathfinder_d000_flash_s0 in the manifest.
ROW=244

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

# Recipe table (1-indexed to match SLURM_ARRAY_TASK_ID directly).
RECIPES=(
    "unused"
    "--warmup-steps 4000 --lr 0.001"
    "--warmup-steps 4000 --lr 0.0003"
    "--warmup-steps 8000 --lr 0.002"
    "--warmup-steps 0    --lr 0.001"
)
NAMES=(
    "unused"
    "warm4000_lr1e-3"
    "warm4000_lr3e-4"
    "warm8000_lr2e-3"
    "nowarm_lr1e-3_control"
)

RECIPE=${RECIPES[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
TAG="pfprobe_${NAME}"

echo "=== $TAG | $(hostname) | $(date) ==="
echo "recipe: $RECIPE"

# GPU pre-flight check: requeue if the assigned node has a wedged/unresponsive GPU.
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
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Force a fresh start: resuming would inherit the optimizer state from a previous recipe.
rm -rf "checkpoints/$TAG"

# Run training probe (--phase smoke ensures these logs are ignored by the Demsar pipeline).
srun python -m attn_rct.train \
    --manifest "$MANIFEST" \
    --row "$ROW" \
    --checkpoint-dir "$PROJECT/checkpoints/$TAG" \
    --result-file "$PROJECT/results/$TAG.json" \
    --checkpoint-every-min 30 \
    --epochs 20 \
    --phase smoke \
    --require-flash \
    $RECIPE &

wait
echo "=== finished $TAG: $(date) ==="