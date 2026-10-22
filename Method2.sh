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

python Method2.py $radius1 $radius2
python Method5.py depression 0
python Method5.py depression 1
python Method5.py depression 2
python Method5.py depression 3
python Method5.py depression 4
python Method5.py high_cholesterol 0
python Method5.py high_cholesterol 1
python Method5.py high_cholesterol 2
python Method5.py high_cholesterol 3
python Method5.py high_cholesterol 4
python Method5.py irritable_bowel_syndrome 0
python Method5.py irritable_bowel_syndrome 1
python Method5.py irritable_bowel_syndrome 2
python Method5.py irritable_bowel_syndrome 3
python Method5.py irritable_bowel_syndrome 4
python Method5.py migraine 0
python Method5.py migraine 1
python Method5.py migraine 2
python Method5.py migraine 3
python Method5.py migraine 4



