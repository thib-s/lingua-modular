#!/bin/bash
# Jean-Zay lingua environment creation
#SBATCH --job-name=lingua_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# ───────────────── timers ─────────────────────────────────────────────────────
start_time=$(date +%s)
current_date=$(date +%y%m%d)
env_name="lingua_25_${current_date}"

# ───────────────── protect $HOME quota ────────────────────────────────────────
for d in .local .cache .conda; do
    tgt="$WORK/$d"
    src="$HOME/$d"

    # If $src is a dead link or a real directory, move/replace it
    if [ ! -L "$src" ] || [ ! -e "$src" ]; then
        echo "Relocating $src → $tgt (and linking back)"
        mkdir -p "$tgt"
        if [ -e "$src" ] && [ ! -L "$src" ]; then
            mv "$src"/* "$tgt"/ 2>/dev/null || true
            rmdir "$src" 2>/dev/null || true
        fi
        ln -sfn "$tgt" "$src"
    fi
done

# ───────────────── load Jean-Zay stack ────────────────────────────────────────
module purge
module load miniforge/24.9.0      # lightweight Conda
# module load pytorch-gpu/py3/2.6.0  # CUDA-12.1 PyTorch bundle
module load cuda/12.1.0

# ───────────────── create Conda env ───────────────────────────────────────────
conda create -y -n "$env_name" python=3.11

conda activate "$env_name"

echo "Using Python: $(which python)"

# ───────────────── PIP installs ───────────────────────────────────────────────
python -m pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
python -m pip install ninja
# python -m pip install --upgrade pip
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install --no-cache-dir ninja
# pip install --no-cache-dir xformers
pip install --no-cache-dir -r requirements.txt
conda clean -a -y

# ───────────────── timing ─────────────────────────────────────────────────────
end_time=$(date +%s)
echo "Environment $env_name built in $(( (end_time-start_time)/60 )) min."

