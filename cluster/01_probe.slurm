#!/bin/bash
#SBATCH --job-name=attnrct_probe
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/fmnisi/attn_rct/logs/probe_%j.out
#SBATCH --error=/home-mscluster/fmnisi/attn_rct/logs/probe_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
# #SBATCH --gres=gpu:1     <- uncomment if `sinfo -o "%P %G"` shows gpu GRES
#
# The FA-2 gate. If this does not print VERIFIED, no efficiency number from the
# baseline arm is trustworthy -- stop and fix it before running anything else.

set -euo pipefail

HOME_DIR=/home-mscluster/fmnisi
source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate attnrct

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
[ -f ~/.bashrc_secrets ] && source ~/.bashrc_secrets

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

python "$HOME_DIR/attn_rct/cluster/probe_backend.py"
