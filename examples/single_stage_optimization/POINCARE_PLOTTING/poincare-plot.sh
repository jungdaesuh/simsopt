#!/bin/bash
#SBATCH --account=m4680
#SBATCH --job-name=poincare_surfaces
#SBATCH --output=poincare_surfaces_%j.out
#SBATCH --error=poincare_surfaces_%j.err
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rb3736@columbia.edu

# Load Python module
module load python/3.11

# Initialize conda for batch environment
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
CONDA_ENV="${CONDA_ENV:-simsopt}"
conda activate "$CONDA_ENV"

# Match thread counts to SLURM allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Go to working directory
SIMSOPT_ROOT="${SIMSOPT_ROOT:-$HOME/simsopt}"
cd "$SIMSOPT_ROOT/examples/single_stage_optimization/POINCARE_PLOTTING"

# Run the simulation
python3 poincare_surfaces.py
