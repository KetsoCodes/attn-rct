#!/bin/bash
#SBATCH --job-name=attnrct_pfcollapse
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/pfcollapse_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/pfcollapse_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --exclude=mscluster44,mscluster45,mscluster47,mscluster51,mscluster57,mscluster58,mscluster61,mscluster62,mscluster64,mscluster66,mscluster67,mscluster68,mscluster69,mscluster70,mscluster74,mscluster78,mscluster83
#
# Reads the three recipe-probe checkpoints and measures the state they are stuck in.
# Trains nothing; a few minutes on a GPU.

set -uo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct

. "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT"
mkdir -p logs

echo "=== pfcollapse | $(hostname) | $(date) ==="

gpu_ok=1
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || gpu_ok=0
if [ "$gpu_ok" = "1" ]; then
    python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda')" 2>/dev/null || gpu_ok=0
fi
if [ "$gpu_ok" = "0" ]; then
    echo "GPU pre-flight FAILED on $(hostname); requeuing."
    scontrol requeue "$SLURM_JOB_ID"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python -u cluster/pathfinder_collapse_probe.py

echo "=== finished: $(date) ==="
