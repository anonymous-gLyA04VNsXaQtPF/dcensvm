#!/bin/bash
# Script to run all R simulation scripts in the Sim directory
#SBATCH --job-name=run_all_simulations
#SBATCH --output=run_all_simulations_%A_%a.out
#SBATCH --error=run_all_simulations_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=600M
#SBATCH --array=0-13%3  # Array range (replace with actual count-1), limit 3 concurrent jobs

# Load necessary modules
module purge
module load slurm
module load R/4.4.0

# Rscript reinstall_loaded_packages.R

# Get the list of R simulation scripts that invoke slurm jobs
mapfile -t scripts < <(find Sim -name "*_slurm.R")

# Print which script is being executed
echo "Running ${scripts[$SLURM_ARRAY_TASK_ID]}"

# Run the R script corresponding to the SLURM_ARRAY_TASK_ID
Rscript "${scripts[$SLURM_ARRAY_TASK_ID]}"

echo "Simulation script ${scripts[$SLURM_ARRAY_TASK_ID]} completed"

# End of script