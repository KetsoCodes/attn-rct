#!/bin/bash
#SBATCH --job-name=attnrct_pfrecipe
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/pfrecipe_%A_%a.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/pfrecipe_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --array=1-3%3
#SBATCH --exclude=mscluster44,mscluster45,mscluster47,mscluster51,mscluster57,mscluster58,mscluster61,mscluster62,mscluster64,mscluster66,mscluster67,mscluster68,mscluster69,mscluster70,mscluster74,mscluster78,mscluster83
#
# Determines if Pathfinder is learnable under a recipe closer to the LRA reference.
#
# Previous checks confirmed the model's architecture, data pipeline, and token handling
# (padding_idx) are structurally sound. The remaining suspect is the training recipe:
# The original LRA paper achieved 71.4% using an LR of 0.01 for 200 epochs. Our initial 
# grid ran much shorter (20 epochs) with much lower LRs (3e-4 to 1e-3).
#
# This array runs three specific configurations to isolate the bottleneck:
#   1. lra_faithful : Reference recipe (d128, LR 0.01, 200 epochs)
#   2. lra_lr_ourdim: High LR applied to our wider design (d256, LR 0.01, 200 epochs)
#   3. ourlr_long   : Our lower LR but with extended training (d128, LR 0.001, 200 epochs)
#
# If all three fail, the recipe is not the issue, and the problem likely stems from our 
# mean-pooling strategy over the constant background tokens.

set -uo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
MANIFEST=$PROJECT/cluster/runs.csv
ROW=244
EPOCHS=200

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

# Index 0 is unused so the array index directly maps to the recipe.
RECIPES=(
    "unused"
    "--d-model 128 --depth 4 --batch-size 32 --lr 0.01  --warmup-steps 1000"
    "--d-model 256 --depth 4 --batch-size 16 --lr 0.01  --warmup-steps 1000"
    "--d-model 128 --depth 4 --batch-size 32 --lr 0.001 --warmup-steps 1000"
)
NAMES=(
    "unused"
    "lra_faithful"
    "lra_lr_ourdim"
    "ourlr_long"
)

RECIPE=${RECIPES[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
TAG="pfrecipe_${NAME}"

echo "=== $TAG | $(hostname) | $(date) ==="
echo "recipe: $RECIPE --epochs $EPOCHS"

# GPU pre-flight check to prevent silent fallbacks to CPU on dead nodes.
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
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Run training probe (--phase smoke ensures these logs are ignored by the main analysis pipeline).
srun python -u -m attn_rct.train \
    --manifest "$MANIFEST" \
    --row "$ROW" \
    --checkpoint-dir "$PROJECT/checkpoints/$TAG" \
    --result-file "$PROJECT/results/$TAG.json" \
    --checkpoint-every-min 20 \
    --epochs "$EPOCHS" \
    --phase smoke \
    --require-flash \
    $RECIPE &

wait
echo "=== finished $TAG: $(date) ==="