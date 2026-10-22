#!/bin/bash
#SBATCH --job-name=kathydata
#SBATCH --nodes=1
#SBATCH --partition=ascher
#SBATCH --time=72:00:00
#SBATCH --output=job.%A_%a.out
#SBATCH --error=job.%A_%a.err
#SBATCH --array=1-11

radius=$(head -n $SLURM_ARRAY_TASK_ID All_Phenotypes2.txt | tail -n 1)

python Method8-Calculated.py $radius 0
python Method8-Calculated.py $radius 1
python Method8-Calculated.py $radius 2
python Method8-Calculated.py $radius 3
python Method8-Calculated.py $radius 4


#python Method8-Precomputed.py $radius 0
#python Method8-Precomputed.py $radius 1
#python Method8-Precomputed.py $radius 2
#python Method8-Precomputed.py $radius 3
#python Method8-Precomputed.py $radius 4





