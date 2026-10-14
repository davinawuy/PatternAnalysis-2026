#!/bin/bash

#SBATCH --account=comp3710
#SBATCH --job-name=eval
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Evaluate a trained model on the test split; the first argument is the run folder
# under outputs/ and any extra arguments go to predict.py.
# Usage: sbatch -J eval-improved        slurm/predict.sh improved
#        sbatch -J eval-hipmri-improved slurm/predict.sh hipmri_improved

# Stop the job when a command fails.
set -eo pipefail

RUN="${1:-}"
EXTRA=("${@:2}")
# Clear the positional arguments so the Conda activate script does not inherit them.
set --

if [ -z "$RUN" ] || [ ! -f "$HOME/ImprovedUNet_Segmentation_S49800414/outputs/$RUN/best.pt" ]; then
    echo "Usage: sbatch slurm/predict.sh <run folder with outputs/<run>/best.pt> [predict.py args]" >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID ($RUN)"
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
srun python -u predict.py --checkpoint "outputs/$RUN/best.pt" "${EXTRA[@]}"

echo "Finish: $(date)"
