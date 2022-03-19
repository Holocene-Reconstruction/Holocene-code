#!/bin/bash
#SBATCH --job-name=da_Holocene       # Name of the job
#SBATCH --output=output_logfile.txt  # File for output and errors
#SBATCH --time=6:00:00               # Maximum time for job to run
#SBATCH --mem=50000                  # Memory (MB)
#SBATCH --cpus-per-task=1            # Number of processors

# Run this with the command: sbatch run_DA.sh.
srun python -u da_main_code.py config.yml

