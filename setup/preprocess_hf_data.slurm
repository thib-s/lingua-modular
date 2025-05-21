#!/usr/bin/env bash
#
#SBATCH --job-name=hf_data_prep           # Name of the job
#SBATCH --account=<account>@cpu                 # possible to switch to compile node
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=32                # Number of CPU cores per task
#SBATCH --hint=nomultithread              # Disable hyperthreading
#SBATCH --output=logs/%x.%j.out           # STDOUT log (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x.%j.err            # STDERR log
#SBATCH --time=02:00:00                   # Max run time (HH:MM:SS)

# Load or activate your environment
# Adjust module load or conda/venv activation as needed
source env_act.sh

# Create log directory if it doesn't exist
mkdir -p logs

# Run the data download and preparation script
python download_prepare_hf_data.py \
    dclm_baseline_1.0_10prct 40 \
    --data_dir "/lustre/fsn1/projects/rech/<PROJ>/<USER>/data/"

echo "Job completed on $(date)"

