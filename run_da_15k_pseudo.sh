#!/bin/bash
#SBATCH --job-name=da_15k_pseudo_Holocene        # Name of the job
#SBATCH --output=output_logfile_15k_pseudo.out   # File for output and errors
#SBATCH --time=3-00:00:00             # Maximum time for job to run
#SBATCH --mem=30000                   # Memory (MB)
#SBATCH --cpus-per-task=1             # Number of processors

# Run this with the command: sbatch run_DA.sh.
srun python -u da_main_code.py config_15k_pseudo.yml

