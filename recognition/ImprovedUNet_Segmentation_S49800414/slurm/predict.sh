#!/bin/bash

#SBATCH --account=comp3710
#SBATCH --job-name=oasis_eval
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Evaluate a trained OASIS model on the test split; extra arguments go to predict.py.
# Usage: sbatch -J eval-improved slurm/predict.sh improved
#        sbatch -J eval-baseline slurm/predict.sh baseline

# Stop the job when a command fails.
set -eo pipefail

MODEL="${1:-}"
EXTRA=("${@:2}")
# Clear the positional arguments so the Conda activate script does not inherit them.
set --

case "$MODEL" in
    baseline|improved) ;;
    *) echo "Usage: sbatch slurm/predict.sh baseline|improved [predict.py args]" >&2; exit 1 ;;
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
srun python -u predict.py --checkpoint "outputs/$MODEL/best.pt" "${EXTRA[@]}"

echo "Finish: $(date)"
