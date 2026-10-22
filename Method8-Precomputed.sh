#!/bin/bash
#SBATCH --job-name=kathydata
#SBATCH --nodes=1
#SBATCH --partition=ascher
#SBATCH --time=72:00:00
#SBATCH --output=job.%A_%a.out
#SBATCH --error=job.%A_%a.err

#SBATCH --array=1-55


radius1=$(head -n $SLURM_ARRAY_TASK_ID PhenotypeParameters.txt | tail -n 1)
radius2=$(head -n $SLURM_ARRAY_TASK_ID FoldParameters.txt | tail -n 1)

python Method8-Precomputed.py $radius1 $radius2





