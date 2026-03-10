#!/bin/bash
#SBATCH -A m4680
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1

# Load Python module
module load python/3.11

# Activate virtual environment
conda activate simsopt

# Go to working directory
cd ~/simsopt/examples/single_stage_optimization/SINGLE_STAGE

# Run the simulation
python3 single_stage_banana_example.py
