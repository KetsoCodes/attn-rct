#!/bin/bash
#SBATCH --job-name=attnrct_data
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/data_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/data_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#
# ListOps, generated with the LRA authors' own generator.
#
# We generate rather than download because the released archive
# (storage.googleapis.com/long-range-arena/lra_release.gz) returns 403 AccessDenied --
# public bucket access was revoked, verified 2026-08-16 from two independent networks.
# Generating with lra_benchmarks/data/listops.py is the fallback the LRA README itself
# documents.
#
# The generated split is NOT byte-identical to the released one, so absolute accuracies
# are not comparable to published LRA numbers. Internal validity is unaffected: every
# arm sees exactly the same generated data, which is all the paired design needs. State
# this plainly in the methodology.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
DATA_ROOT=/datasets/fmnisi/lra
REPO=$HOME_DIR/long-range-arena

# The commit is pinned so the dataset can be regenerated exactly. Record it in the
# write-up alongside the results.
LRA_COMMIT=cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"

# A separate, disposable env. The generator is TF/JAX-era code needing tensorflow, and
# TF and PyTorch pin conflicting numpy versions -- installing TF into `attnrct` would
# risk breaking the training environment to produce a dataset exactly once.
#   conda env remove -n lradata      # once the data exists
if ! conda env list | grep -q "^lradata "; then
    conda create -y -n lradata python=3.11
fi
conda activate lradata

echo "=== host: $(hostname) | $(date) ==="

# /datasets/ may not exist yet, so fall back to home. ListOps is tens of MB; guideline
# 7.4 targets 50GB+ datasets, so the home quota is not a concern here.
if ! mkdir -p "$DATA_ROOT" 2>/dev/null; then
    echo "NOTE: cannot write /datasets/fmnisi -- using home dir instead"
    DATA_ROOT=$HOME_DIR/data/lra
    mkdir -p "$DATA_ROOT"
fi
echo "data root: $DATA_ROOT"

if [ ! -d "$REPO" ]; then
    git clone https://github.com/google-research/long-range-arena.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin "$LRA_COMMIT"
git checkout --detach "$LRA_COMMIT"
echo "LRA repo commit: $(git rev-parse HEAD)"

# tensorflow-cpu: the generator only uses tf.io.gfile to write files, so the GPU build
# would be a wasted download. LRA's jax/flax deps are for their training code.
python -c "import tensorflow" 2>/dev/null || pip install -q "tensorflow-cpu" absl-py numpy
python -c "import tensorflow as tf; print('tensorflow:', tf.__version__)"

OUT=$DATA_ROOT/listops
mkdir -p "$OUT"

echo "--- generating ListOps ---"
PYTHONPATH="$(pwd):${PYTHONPATH:-}" python -m lra_benchmarks.data.listops \
    --output_dir="$OUT" || {
        echo
        echo "Generator failed. Things to check:"
        echo "  * ls lra_benchmarks/data/listops.py   (module path correct?)"
        echo "  * pip install absl-py                 (flag parsing)"
        echo "  * available flags:"
        PYTHONPATH="$(pwd)" python -m lra_benchmarks.data.listops --help 2>&1 | head -40
        exit 1
    }

echo "--- produced ---"
ls -la "$OUT"

echo "--- schema and sizes ---"
for f in "$OUT"/*.tsv; do
    [ -e "$f" ] || continue
    printf '%s: %s rows | header: ' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
    head -1 "$f"
done

# The generator caps on TREE NODE COUNT but writes a rendered string with brackets, so
# --max_length 2000 yields sequences averaging ~3103 whitespace tokens. We filter rather
# than truncate downstream, so expect a large share to be dropped at load time. See the
# open issue in the handover.
echo
echo "--- rendered token lengths (this is what the loader sees) ---"
for f in "$OUT"/*.tsv; do
    [ -e "$f" ] || continue
    awk -F'\t' 'NR>1 {n=split($1, a, " "); s+=n; if (n>m) m=n; c++}
                END {if (c) printf "%s: mean %.0f, max %d tokens over %d rows\n",
                     FILENAME, s/c, m, c}' "$f"
done

echo
echo "Expect ~96k train / 2k val / 2k test with Source/Target columns."
echo "If filenames differ from basic_*.tsv, update --data-dir in train_array.slurm."
echo "=== done: $(date) ==="
