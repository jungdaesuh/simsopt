#!/bin/bash
#SBATCH --account=apam
#SBATCH --job-name=adaptive_boozer_opt
#SBATCH --output=iota_adaptive_boozer_opt_%j.out
#SBATCH --error=iota_adaptive_boozer_opt_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simsopt

# Navigate to working directory
cd ~/Projects/Banana-Coils/simsopt/examples/

# Run your Python script
python iota_iterable_single_stage.py
