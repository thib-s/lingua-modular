#!/bin/bash
#SBATCH --job-name=lora_eval_test            # job name
#SBATCH --partition=gpu_p5                 # H100 queue (‐C h100 is implied)
#SBATCH --nodes=1                          # change to the number of nodes you booked
#SBATCH --ntasks-per-node=1               # one launcher task per node
#SBATCH --cpus-per-task=64                 # 96 physical cores per H100 node
#SBATCH --gres=gpu:2                       # 4 GPUs / node on gpu_p6
#SBATCH --hint=nomultithread               # disable SMT
#SBATCH -C a100                            # mandatory feature flag
#SBATCH -A reh@a100                        # your project REPLACE BY CORRECT PROJECT NAME
#SBATCH --time=12:00:00                    # wall-time (≤100 h on gpu_p6)
#SBATCH --output=logs/%x_%j.out            # STDOUT/ERR
#SBATCH --error=logs/%x_%j.err

#####################
# software stack
#####################
module purge
module load arch/a100                      # A100-compatible compiler & libs :contentReference[oaicite:0]{index=0}
module load miniforge/24.9.0
module load cuda/12.8.0
conda activate lingua_25_250520

#####################
# rendez-vous (master) definition
#####################
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500                   # pick any free port


#####################
# optional: keep W&B local
#####################
export MY_CACHE=/lustre/fswork/projects/rech/reh/commun/lingua_cache        # or /scratch/lingua_cache
export HF_DATASETS_CACHE=$MY_CACHE/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline                  # local logs only :contentReference[oaicite:1]{index=1}

#####################
# Path to the model and adapter
#####################
export PEFT_MODULE_CKPT="/lustre/fswork/projects/rech/reh/commun/FOR-sight-ai/SmolLM2-1.7B/Pile-Freelaw"
export BASE_CKPT="/lustre/fswork/projects/rech/reh/commun/models/SmolLM2-1.7B"
export OUTPUT_PATH="/lustre/fswork/projects/rech/reh/commun/evals_results"
export OUTPUT_NAME="pile_law_peft_mmlu_eval_test"

#####################
# distributed launch
#####################
srun --export=ALL --kill-on-bad-exit \
    lighteval accelerate \
        "pretrained=$BASE_CKPT,peft=$PEFT_MODULE_CKPT" \
        "batch_size=1" \
        "mmlu|international_law|0|0" \
        "mmlu:jurisprudence|0|0" \
        "mmlu:professional_law|0|0" \
        "mmlu:public_relations|0|0" \
        "mmlu:us_foreign_policy|0|0" \
        "mmlu:high_school_government_and_politics|0|0" \
        "mmlu:global_facts|0|0" \
        --max-samples 40 \
        --output-dir "$OUTPUT_PATH/$OUTPUT_NAME" \
        --save-details
