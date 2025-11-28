#!/bin/bash
#SBATCH --job-name=mpi_boot
#SBATCH --output=mpi_bootstrap_%j.out
#SBATCH --error=mpi_bootstrap_%j.err
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --partition=short-40core

# Load required modules
module load anaconda3
module load mpi4py/latest

# Change to working directory (UPDATE THIS PATH!)
cd /gpfs/home/luwwei/final_project

# Run the MPI bootstrap script
mpirun -np 4 python mpi_bootstrap.py

echo "Job completed at $(date)"
