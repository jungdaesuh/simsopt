#!/bin/bash
#SBATCH --account=m4680
#SBATCH --job-name=poincare_jax_default
#SBATCH --output=poincare_jax_default_%j.out
#SBATCH --error=poincare_jax_default_%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rb3736@columbia.edu

module load python/3.11

source "$(conda info --base)/etc/profile.d/conda.sh"

CONDA_ENV="${CONDA_ENV:-simsopt}"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export POINCARE_JAX_PLATFORM="${POINCARE_JAX_PLATFORM:-gpu}"

SIMSOPT_ROOT="${SIMSOPT_ROOT:-$HOME/simsopt}"
cd "$SIMSOPT_ROOT/examples/single_stage_optimization/POINCARE_PLOTTING"

python3 poincare_surfaces_jax_default.py "$@"
