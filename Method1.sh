#!/bin/bash
#SBATCH --job-name=kathydata
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=job.%A_%a.out
#SBATCH --error=job.%A_%a.err
#SBATCH --array=1-50
#SBATCH --mem=50G

radius1=$(head -n $SLURM_ARRAY_TASK_ID PhenotypeParametersReal.txt | tail -n 1)
radius2=$(head -n $SLURM_ARRAY_TASK_ID FoldParametersReal.txt | tail -n 1)

#conda activate leopard-sdm
#
#python Method1.py "$radius1" "$radius2"
#python Method2.py "$radius1" "$radius2"
#python Method3.py "$radius1" "$radius2"
#python Method4.py "$radius1" "$radius2"
#python Method6.py "$radius1" "$radius2"
#python Method8-Calculated.py "$radius1" "$radius2"
#python Method8-Precomputed.py "$radius1" "$radius2"

# =========================================================
# LDSC environment methods
# =========================================================
conda activate ldscc

#python Method5.py "$radius1" "$radius2"
python Method7.py "$radius1" "$radius2"
python Method10.py "$radius1" "$radius2"

# Return to main environment
#conda activate leopard-sdm



# 
#python Method8-Precomputed.py depression 0  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py depression 1  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py depression 2  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py depression 3  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py depression 4  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py high_cholesterol 0  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py high_cholesterol 1  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py high_cholesterol 2  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py high_cholesterol 3  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py high_cholesterol 4  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py hypertension 0  # partial NULL h2 (4/6 NaN)
#python Method8-Precomputed.py hypertension 1  # partial NULL h2 (4/6 NaN)
#python Method8-Precomputed.py hypertension 2  # partial NULL h2 (4/6 NaN)
#python Method8-Precomputed.py hypertension 3  # partial NULL h2 (4/6 NaN)
#python Method8-Precomputed.py hypertension 4  # partial NULL h2 (4/6 NaN)
#python Method8-Precomputed.py irritable_bowel_syndrome 0  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py irritable_bowel_syndrome 1  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py irritable_bowel_syndrome 2  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py irritable_bowel_syndrome 3  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py irritable_bowel_syndrome 4  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py migraine 0  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py migraine 1  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py migraine 2  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py migraine 3  # partial NULL h2 (1/6 NaN)
#python Method8-Precomputed.py migraine 4  # partial NULL h2 (1/6 NaN))