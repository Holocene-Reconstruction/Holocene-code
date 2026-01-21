#!/bin/bash
#SBATCH --job-name=da_process  # Name of the job
#SBATCH --output=output.txt    # File for output and errors
#SBATCH --time=10:00           # Maximum time for job to run
#SBATCH --mem=20000            # Memory (MB)

# Run this with the command: sbatch run_script.sh.
#srun python -u make_combined_file_iters.py 'locrad_none'
#srun python -u make_combined_file_iters.py 'locrad_10k'
#srun python -u make_combined_file_iters.py 'locrad_5k'
#srun python -u make_combined_file_iters.py 'locrad_5k_plus_rankbased'
#srun python -u make_combined_file_iters.py 'locrad_4k'
#srun python -u make_combined_file_iters.py 'locrad_3k'
#srun python -u make_combined_file_iters.py 'locrad_1k'
srun python -u make_combined_file_iters.py 'locrad_5k_temp_precip'

