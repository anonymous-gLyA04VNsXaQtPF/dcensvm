#!/bin/bash
# This script is designed to run the real data analysis in R on a SLURM cluster.
#SBATCH --job-name=real_data_analysis
#SBATCH --output=real_data_analysis_%j.out
#SBATCH --error=real_data_analysis_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=600M

# Load necessary modules
module purge
module load slurm
module load R/4.4.0

# Rscript reinstall_loaded_packages.R

# Run the real data analysis script
Rscript real_data/real_data_slurm.R

echo "Real data analysis completed"