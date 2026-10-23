#!/bin/bash
#SBATCH --job-name=attnrct
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/run_%A_%a.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/run_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --signal=B:USR1@600
#SBATCH --exclude=mscluster44,mscluster45,mscluster47,mscluster51,mscluster57,mscluster58,mscluster61,mscluster62,mscluster64,mscluster66,mscluster67,mscluster68,mscluster69,mscluster70,mscluster74,mscluster78,mscluster83

# Submit using an array throttle (e.g., `sbatch --array=1-120%6 cluster/train_array.slurm`)
# to respect the cluster's concurrent job limit.
#
# Walltime handling. Ten minutes before the 8-hour limit Slurm sends SIGUSR1 to THIS batch
# shell (the `B:` prefix). The shell then has to do three things, in order:
#   1. forward the signal to training, which checkpoints at its next batch and exits;
#   2. wait for that checkpoint to finish being written;
#   3. requeue this array task, which resumes from the checkpoint when it restarts.
# An earlier version requested the signal but never trapped it. Bash's default action for
# SIGUSR1 is to terminate, so the batch shell died on the spot: the job ended FAILED with
# exit 0:10, training never received the signal, and nothing requeued. Twelve Pathfinder
# cells were lost this way at 7h55m; they survived only because of the 30-minute and
# per-epoch checkpoints. The lead time is 10 minutes, not 5, so even a slow batch and the
# checkpoint write fit inside it.
#
# Node exclusion. These nodes have repeatedly shown dead GPUs ("No devices were found").
# The pre-flight check below catches a dead card anyway, but excluding the known-bad nodes
# stops a job landing on one in the first place.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct
MANIFEST=$PROJECT/cluster/runs.csv

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}

# Prevents PyTorch CUDA memory fragmentation caused by dynamic sequence lengths across batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.bashrc_secrets ] && source ~/.bashrc_secrets

export WANDB_PROJECT=attn-rct-pilot
export WANDB_ENTITY=1903697-wits-university
export WANDB_DIR=$PROJECT/logs

# Maps the Slurm array index to the corresponding manifest row.
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$ROW" ]; then
    echo "FATAL: no manifest row for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi
RUN_ID=$(echo "$ROW" | cut -d, -f1)

CKPT_DIR=$PROJECT/checkpoints/$RUN_ID
RESULT=$PROJECT/results/$RUN_ID.json
mkdir -p "$CKPT_DIR" "$PROJECT/results" "$PROJECT/logs"

# Skips already completed runs to safely handle array resubmissions.
if [ -f "$RESULT" ]; then
    echo "$RUN_ID already complete ($RESULT) -- skipping"
    exit 0
fi

echo "=== $RUN_ID | task $SLURM_ARRAY_TASK_ID | $(hostname) | $(date) ==="
echo "row: $ROW"

# GPU Pre-flight Check: Validates GPU functionality before initiating training.
# If the node has a wedged or unresponsive GPU, the script requeues the task to try a
# different node. It then exits 0, not 1: a task that requeued itself and then exited with
# an error was observed to come back in a HELD state ("job requeued in held state"), where
# it never runs again until someone releases it by hand. Three Text cells sat held for two
# days that way. The walltime path below already exits 0 after requeuing, and those resume
# normally.
gpu_ok=1
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || gpu_ok=0
if [ "$gpu_ok" = "1" ]; then
    python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda')" 2>/dev/null || gpu_ok=0
fi

if [ "$gpu_ok" = "0" ]; then
    echo "GPU pre-flight FAILED on $(hostname): CUDA device unusable."
    nvidia-smi 2>&1 | head -3 || true
    echo "  requeuing ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} to try another node."
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

TRAIN_PID=""

on_walltime_warning() {
    echo "=== walltime warning at $(date): checkpointing $RUN_ID and requeuing ==="
    if [ -n "$TRAIN_PID" ]; then
        # srun forwards SIGUSR1 to the training process, whose handler checkpoints at the
        # next batch and exits with status 42 (EXIT_REQUEUE in train.py).
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "=== requeued ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
    exit 0
}
trap on_walltime_warning USR1

# Executes the training run. Data directories are dynamically sourced from the manifest.
# Run in the background so the shell stays free to receive the walltime signal; a shell
# blocked on a foreground command does not run its trap until that command returns.
srun python -m attn_rct.train \
    --manifest "$MANIFEST" \
    --row "$SLURM_ARRAY_TASK_ID" \
    --checkpoint-dir "$CKPT_DIR" \
    --result-file "$RESULT" \
    --checkpoint-every-min 30 \
    --phase full \
    --require-flash &
TRAIN_PID=$!

wait "$TRAIN_PID"
echo "=== finished $RUN_ID: $(date) ==="
