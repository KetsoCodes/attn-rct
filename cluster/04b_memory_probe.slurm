#!/bin/bash
#SBATCH --job-name=attnrct_mem
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/mem_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/mem_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#
# The memory gate. Run this before the grid, and before re-running the pilot.
#
# The pilot lost cells to OOM, and it lost them arm-selectively: vanilla died at depth 6
# and sparse died at depth 4, while flash and linear never did. Those failures are not
# recoverable at analysis time -- the Friedman test needs a complete design x variant
# matrix, so a single OOM cell drops the entire design, and because OOM correlates with
# the arm, what survives is biased toward the efficient arms.
#
# So the question this answers is not "does it usually fit" but "does EVERY cell fit at
# the worst case the loader can produce". It trains real steps on a full batch of
# max_len sequences with no padding to shorten them.
#
# What to check in the output:
#   1. The verdict line. PASS means every cell fits and the grid can be submitted.
#      FAIL lists the cells that OOM -- lower batch_size or depth until it is empty.
#   2. The gap between alloc and reserved. That gap is fragmentation. It should be
#      small with expandable_segments set; if it is not, the allocator is still
#      churning and something is allocating attention-sized tensors per batch.
#   3. reserved for vanilla should be roughly depth x (batch x heads x len^2 x 2 bytes)
#      plus a couple more of the same. If it is far above that, an arm is materialising
#      more than its score matrix.
#
# Exit status is 0 only if no cell OOM'd, so it can gate a submission script.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct

. "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}

# Sequence lengths vary every batch, so the caching allocator is asked for a differently
# sized score matrix each time and fragments badly -- the failed pilot runs showed 6.5 GB
# reserved but unallocated. This must be set for EVERY arm, not just the ones that were
# failing, or the allocator itself becomes an arm-dependent variable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT"
mkdir -p logs results

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

python cluster/memory_probe.py \
    --steps 3 \
    --out "results/memory_probe_$(hostname)_${SLURM_JOB_ID:-local}.json" \
    "$@"
