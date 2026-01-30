#!/bin/bash
#SBATCH --account=apam
#SBATCH --job-name=banana_scan
#SBATCH --output=banana_scan_%j.out
#SBATCH --error=banana_scan_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simsopt

# Navigate to working directory
cd ~/Projects/Banana-Coils/simsopt/examples/

# Run the simulation
python iterable_banana_coil_solver.py

