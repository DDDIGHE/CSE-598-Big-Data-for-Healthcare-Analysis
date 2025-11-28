#!/bin/bash
#SBATCH --job-name=mpi_bootstrap
#SBATCH --output=mpi_bootstrap_%j.out
#SBATCH --error=mpi_bootstrap_%j.err
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --partition=short-28core

# Load required modules
module load anaconda3
module load openmpi

# Change to working directory
cd /gpfs/projects/AMS598/class2025/Wei_Lu/project_final

# Run the MPI bootstrap script
mpiexec -n 4 python mpi_bootstrap.py

echo "Job completed at $(date)"
