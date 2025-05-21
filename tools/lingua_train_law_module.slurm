#!/bin/bash
#SBATCH --job-name=lingua_train            # job name
#SBATCH --partition=gpu_p6                 # H100 queue (‐C h100 is implied)
#SBATCH --nodes=8                          # change to the number of nodes you booked
#SBATCH --ntasks-per-node=1               # one launcher task per node
#SBATCH --cpus-per-task=96                # 96 physical cores per H100 node
#SBATCH --gres=gpu:4                       # 4 GPUs / node on gpu_p6
#SBATCH --hint=nomultithread               # disable SMT
#SBATCH -C h100                            # mandatory feature flag
#SBATCH -A reh@h100                        # your project REPLACE BY CORRECT PROJECT NAME
#SBATCH --time=2:00:00                    # wall-time (≤100 h on gpu_p6)
#SBATCH --output=logs/%x_%j.out            # STDOUT/ERR
#SBATCH --error=logs/%x_%j.err

#####################
# software stack
#####################
module purge
module load arch/h100                      # H100-compatible compiler & libs :contentReference[oaicite:0]{index=0}
module load miniforge/24.9.0
module load cuda/12.8.0
conda activate lingua_25_250520 		   # replace this by env name likely lingua_25_<date>

#####################
# optional: keep W&B local
#####################
export WANDB_MODE=offline                  # local logs only :contentReference[oaicite:1]{index=1}

#####################
# rendez-vous (master) definition
#####################
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500                   # pick any free port

#####################
# distributed launch
#####################
srun --export=ALL --kill-on-bad-exit \
     torchrun \
       --nnodes=$SLURM_NNODES \
       --nproc_per_node=4 \
       --rdzv_backend=c10d \
       --rdzv_id=$SLURM_JOB_ID \
       --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
       -m apps.main.train \
          config=/lustre/fswork/projects/rech/reh/ull45hr/lingua-modular/apps/main/configs/modular_law_1B.yaml

