#!/bin/bash
#SBATCH --job-name=attnrct_flash
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/flash_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/flash_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
# #SBATCH --gres=gpu:1     <- uncomment if `sinfo -p bigbatch -o "%P %G"` shows gpu GRES
#
# flash-attn only, and only on bigbatch. Do NOT submit this with -p stampede (GTX 1060,
# sm_61) or -p biggpu (RTX 8000, sm_75) -- neither can run FA-2.
#
# We install from a PREBUILT WHEEL, not from source. bigbatch has no nvcc, so a source
# build fails during metadata preparation before it compiles anything. find_flash_wheel.py
# reads cuda/torch/abi/python/platform from the live interpreter and installs the wheel
# that matches, because a mismatch surfaces as an ImportError at runtime rather than at
# install time.
#
# No `set -e` around the install: a failure should print diagnostics, not kill the job
# before it can say why.

set -uo pipefail

HOME_DIR=/home-mscluster/fmnisi
PROJECT=$HOME_DIR/attn_rct

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

echo "=== host: $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv

CAP=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability(0))" 2>/dev/null || echo "00")
echo "compute capability: sm_${CAP}"
if [ "${CAP:0:1}" -lt 8 ] 2>/dev/null; then
    echo "FATAL: sm_${CAP} is below sm_80. FA-2 cannot run here."
    echo "  -> submit to bigbatch, without overriding -p."
    exit 1
fi

echo "--- matching and installing a prebuilt wheel ---"
python "$PROJECT/cluster/find_flash_wheel.py" --install

if python -c "import flash_attn" 2>/dev/null; then
    python -c "import flash_attn; print('SUCCESS flash_attn', flash_attn.__version__)"
else
    echo "FAILED. Diagnostics:"
    python -c "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda)"
    echo
    echo "Next options:"
    echo " 1. Re-run find_flash_wheel.py without --install to see which torch versions"
    echo "    have wheels for this python/abi, then pin torch in 00a and rebuild the env."
    echo " 2. Ticket to support@wits-mss.supportsystem.com, subject 'TWK HPC Query',"
    echo "    asking for the CUDA toolkit on bigbatch nodes."
    exit 1
fi

echo "=== done: $(date) ==="
