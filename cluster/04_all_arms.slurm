#!/bin/bash
#SBATCH --job-name=attnrct_allarms
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/allarms_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/allarms_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#
# All five arms, one design, one seed. The last check before committing to the grid.
#
# Each arm has been unit-tested, but never all five on the same design, same data, same
# hardware. This is where cross-arm problems surface: an arm that OOMs at the
# design-space size, an arm far slower than expected, or the flash arm quietly failing
# to take the FA-2 path. It runs on bigbatch because that is the only partition where
# FA-2 works, and the point is to exercise the real configuration.
#
# What to check in the output:
#   1. Every arm completes, with no OOM.
#   2. The flash arm prints an FA-2 backend line. It runs with --require-flash, so it
#      fails loudly rather than silently measuring a different kernel.
#   3. Parameter counts: vanilla/flash/linear/sparse identical; linformer larger by
#      exactly k * max_len (512,000 at k=256, max_len=2000) from the shared E.
#   4. train_seconds per arm -- the first real efficiency signal. The sparse arm is a
#      masked DENSE implementation, so it should be no faster than vanilla. Expected.
#   5. vanilla and flash should reach near-identical loss: same function, shared
#      initialisation. Divergence means something is wrong.

set -uo pipefail          # not -e: one arm failing should not hide the others

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct

. "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}

# Sequence lengths vary every batch, so the caching allocator is asked for a differently
# sized score matrix each time and fragments badly -- the failed pilot runs showed
# 6.5 GB reserved but unallocated. Set for EVERY arm, not just the ones that were
# failing, or the allocator becomes an arm-dependent variable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f "$HOME/.bashrc_secrets" ]; then
    . "$HOME/.bashrc_secrets"
fi
export WANDB_PROJECT=attn-rct-pilot
export WANDB_ENTITY=1903697-wits-university
export WANDB_DIR=$PROJECT/logs

cd "$PROJECT"
mkdir -p logs results checkpoints

echo "=== host: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Design 0, seed 0, one row per arm. The manifest orders design > variant > seed, so
# these land three apart. Regenerate with make_manifest.py if the design space changes:
# these indices are NOT stable across manifests.
ROWS="1 4 7 10 13"

for ROW in $ROWS; do
    RUN_ID=$(sed -n "$((ROW + 1))p" cluster/runs.csv | cut -d, -f1)
    echo
    echo "############ row $ROW : $RUN_ID ############"

    EXTRA=""
    if [[ "$RUN_ID" == *flash* ]]; then
        EXTRA="--require-flash"
    fi

    python -m attn_rct.train \
        --manifest cluster/runs.csv --row "$ROW" \
        --data-dir /datasets/fmnisi/lra/listops \
        --checkpoint-dir "checkpoints/allarms_$RUN_ID" \
        --result-file "results/allarms_$RUN_ID.json" \
        --phase pilot \
        --epochs 1 --limit-batches 50 \
        --batch-size 8 \
        $EXTRA
    STATUS=$?

    echo "exit status: $STATUS"
done

echo
echo "=== summary ==="
python - <<'PY'
import glob, json

rows = []
for path in sorted(glob.glob("results/allarms_*.json")):
    with open(path) as handle:
        rows.append(json.load(handle))

if not rows:
    print("no results written -- every arm failed, check the log above")
else:
    print(f"{'arm':<12}{'params':>12}{'val acc':>10}{'sec/epoch':>12}{'peak MB':>10}  backend")
    for r in rows:
        # We fall back to best_val_accuracy when this session trained nothing (a resume
        # that was already complete), and flag it so the two are not confused.
        accuracy = r["final_val_accuracy"]
        flag = "" if accuracy is not None else "  (restored, not measured this run)"
        accuracy = accuracy if accuracy is not None else r["best_val_accuracy"]
        print(f"{r['variant']:<12}{r['n_params']:>12,}"
              f"{(accuracy or 0):>10.3f}"
              f"{(r['mean_train_seconds_per_epoch'] or 0):>12.1f}"
              f"{(r['peak_memory_mb'] or 0):>10.0f}  {r.get('backend') or ''}{flag}")
PY

echo "=== done: $(date) ==="
