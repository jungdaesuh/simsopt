#!/bin/bash
#SBATCH --account=m4680
#SBATCH --job-name=banana_scan
#SBATCH --output=banana_scan_%j.out
#SBATCH --error=banana_scan_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL

# Load Python module
module load python/3.11

# Activate virtual environment
conda activate simsopt

# Go to working directory
cd ~/simsopt/examples/single_stage_optimization/STAGE_2

# Run the simulation
python3 banana_coil_solver.py
