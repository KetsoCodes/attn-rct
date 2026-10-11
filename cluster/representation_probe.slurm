```bash
#!/bin/bash
#SBATCH --job-name=attnrct_repprobe
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/repprobe_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/repprobe_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --exclude=mscluster44,mscluster45,mscluster47,mscluster51,mscluster57,mscluster58,mscluster61,mscluster62,mscluster64,mscluster66,mscluster67,mscluster68,mscluster69,mscluster70,mscluster74,mscluster78,mscluster83
#
# I use this to compare what each arm's encoder actually learned by looking at the 
# checkpoints already sitting on disk. It doesn't train anything new.
#
# I'm testing four specific scopes to figure out if my results are actually about 
# the attention mechanism itself, or just a specific hyperparameter configuration:
#   cifar d000   -- Flash's STRONGEST CIFAR design (0.3937), so I'm comparing them 
#                   where the softmax arms look their absolute best.
#   cifar d005   -- Flash's WEAKEST CIFAR design (0.2159), where linear still gets 0.4077.
#   listops d000 -- The strongest design for the other learnable task.
#   listops d005 -- The weakest design for ListOps (flash 0.2215, linear 0.3641).
#
# If the gap in gamma and effective rank tracks with the accuracy gap across all four, 
# then the mechanism story holds up. If the gap only shows up in the weak designs, 
# then it's actually about configuration sensitivity rather than the mechanisms 
# themselves -- which is a different, but still highly reportable finding.

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

echo "=== repprobe | $(hostname) | $(date) ==="

# Pre-flight check to catch dead GPUs before they ruin my run.
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

# Run the probe across my four chosen scopes.
for scope in "cifar 0" "cifar 5" "listops 0" "listops 5"; do
    set -- $scope
    echo ""
    srun python -u cluster/representation_probe.py --task "$1" --design-id "$2" --seed 0
done

echo "=== finished: $(date) ==="

```