#!/bin/bash

#SBATCH --account=comp3710
#SBATCH --job-name=train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Train one segmentation model; any extra arguments go to train.py.
# Usage: sbatch -J oasis-improved  slurm/train.sh improved
#        sbatch -J oasis-baseline  slurm/train.sh baseline
#        sbatch -J hipmri-improved slurm/train.sh improved --dataset hipmri
#        sbatch -J oasis-smoke     slurm/train.sh improved --epochs 2 --limit 64 --out outputs/smoke_improved

# Stop the job when a command fails.
set -eo pipefail

MODEL="${1:-}"
EXTRA=("${@:2}")
# Clear the positional arguments so the Conda activate script does not inherit them.
set --

case "$MODEL" in
    baseline|improved) ;;
    *) echo "Usage: sbatch slurm/train.sh baseline|improved [train.py args]" >&2; exit 1 ;;
esac

echo "Job: $SLURM_JOB_ID ($MODEL)"
echo "Host: $(hostname)"
echo "Start: $(date)"

# Activate the existing PyTorch environment.
source "$HOME/miniconda3/bin/activate"
conda activate torch

# Reject unset variables after activating Conda.
set -u

cd "$HOME/ImprovedUNet_Segmentation_S49800414"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

nvidia-smi
srun python -u train.py --model "$MODEL" --workers 8 "${EXTRA[@]}"

echo "Finish: $(date)"
