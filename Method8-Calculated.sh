#!/bin/bash
#SBATCH --job-name=kathydata
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=job.%A_%a.out
#SBATCH --error=job.%A_%a.err
#SBATCH --array=1-1

python Method8-Calculated.py body_mass_index_bmi 0
python Method8-Calculated.py body_mass_index_bmi 1
python Method8-Calculated.py body_mass_index_bmi 2
python Method8-Calculated.py body_mass_index_bmi 3
python Method8-Calculated.py body_mass_index_bmi 4
python Method8-Calculated.py hypertension 0
python Method8-Calculated.py hypertension 1
python Method8-Calculated.py hypertension 2
python Method8-Calculated.py hypertension 3
python Method8-Calculated.py hypertension 4
python Method8-Calculated.py depression 0
python Method8-Calculated.py depression 1
python Method8-Calculated.py depression 2
python Method8-Calculated.py depression 3
python Method8-Calculated.py depression 4
python Method8-Calculated.py asthma 0
python Method8-Calculated.py asthma 1
python Method8-Calculated.py asthma 2
python Method8-Calculated.py asthma 3
python Method8-Calculated.py asthma 4
python Method8-Calculated.py osteoarthritis 0
python Method8-Calculated.py osteoarthritis 1
python Method8-Calculated.py osteoarthritis 2
python Method8-Calculated.py osteoarthritis 3
python Method8-Calculated.py osteoarthritis 4
python Method8-Calculated.py high_cholesterol 0
python Method8-Calculated.py high_cholesterol 1
python Method8-Calculated.py high_cholesterol 2
python Method8-Calculated.py high_cholesterol 3
python Method8-Calculated.py high_cholesterol 4
python Method8-Calculated.py irritable_bowel_syndrome 0
python Method8-Calculated.py irritable_bowel_syndrome 1
python Method8-Calculated.py irritable_bowel_syndrome 2
python Method8-Calculated.py irritable_bowel_syndrome 3
python Method8-Calculated.py irritable_bowel_syndrome 4
python Method8-Calculated.py hypothyroidism_myxoedema 0
python Method8-Calculated.py hypothyroidism_myxoedema 1
python Method8-Calculated.py hypothyroidism_myxoedema 2
python Method8-Calculated.py hypothyroidism_myxoedema 3
python Method8-Calculated.py hypothyroidism_myxoedema 4
python Method8-Calculated.py gastro_oesophageal_reflux_gord_gastric_reflux 0
python Method8-Calculated.py gastro_oesophageal_reflux_gord_gastric_reflux 1
python Method8-Calculated.py gastro_oesophageal_reflux_gord_gastric_reflux 2
python Method8-Calculated.py gastro_oesophageal_reflux_gord_gastric_reflux 3
python Method8-Calculated.py gastro_oesophageal_reflux_gord_gastric_reflux 4
python Method8-Calculated.py migraine 0
python Method8-Calculated.py migraine 1
python Method8-Calculated.py migraine 2
python Method8-Calculated.py migraine 3
python Method8-Calculated.py migraine 4
python Method8-Calculated.py SampleData1 0
python Method8-Calculated.py SampleData1 1
python Method8-Calculated.py SampleData1 2
python Method8-Calculated.py SampleData1 3
python Method8-Calculated.py SampleData1 4



