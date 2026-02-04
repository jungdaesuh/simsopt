#!/bin/bash
#SBATCH --account=m4680
#SBATCH --job-name=poincare_surfaces
#SBATCH --output=poincare_surfaces_%j.out
#SBATCH --error=poincare_surfaces_%j.err
#SBATCH --time=04:00:00
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
cd ~/simsopt/examples/single_stage_optimization/POINCARE_PLOTTING

# Run the simulation
python3 poincare-surfaces.py
