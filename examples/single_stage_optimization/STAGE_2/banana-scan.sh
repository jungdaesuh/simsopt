#!/bin/bash
#SBATCH --job-name=banana_solver
#SBATCH --account=m4680
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=banana_%j.out

# Load Python module
module load python/3.11

# Initialize conda for batch environment
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
CONDA_ENV="${CONDA_ENV:-simsopt}"
conda activate "$CONDA_ENV"

# Set thread counts to match SLURM allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Recommended thread placement for CPU nodes
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Go to working directory
SIMSOPT_ROOT="${SIMSOPT_ROOT:-$HOME/simsopt}"
cd "$SIMSOPT_ROOT/examples/single_stage_optimization/STAGE_2"

# Run the simulation
srun python banana_coil_solver.py "$@"
