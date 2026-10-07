#!/bin/bash

#SBATCH --account=comp3710
#SBATCH --job-name=data_audit
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Dataset audit (OASIS by default). Read-only; the report goes to the .out log.
# Usage: sbatch slurm/audit.sh [extra data_audit.py args]
#        DATA_ROOT=/other/path sbatch -J hipmri-audit slurm/audit.sh --scan-regex "(\d+_week_\d+)"

# Stop the job when a command fails.
set -eo pipefail

ARGS=("$@")
# Clear the positional arguments so the Conda activate script does not inherit them.
set --

echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"

# Activate the existing PyTorch environment.
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Reject unset variables after activating Conda.
set -u

cd "$HOME/ImprovedUNet_Segmentation_S49800414"
export PYTHONUNBUFFERED=1

srun python -u data_audit.py --root "${DATA_ROOT:-/home/groups/comp3710/OASIS}" "${ARGS[@]}"

echo "Finish: $(date)"
