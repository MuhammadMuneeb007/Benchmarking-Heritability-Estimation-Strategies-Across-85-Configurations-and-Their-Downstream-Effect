#!/usr/bin/env python
# coding: utf-8

# # Download and Install Anaconda:
# 
# ### Windows:
# 1. Visit the [Anaconda website](https://www.anaconda.com/download).
# 2. Choose the appropriate installer (64-bit recommended for most systems).
# 3. Run the installer and follow the on-screen instructions.
# 
# ### Linux:
# 1. Open a terminal window and run these commands:
#     ```bash
#     wget https://repo.anaconda.com/archive/Anaconda3-2024.01-Linux-x86_64.sh
#     chmod +x Anaconda3-2024.01-Linux-x86_64.sh
#     bash Anaconda3-2024.01-Linux-x86_64.sh
#     source ~/.bashrc
#     ```
# 
# ### Create a Dedicated Environment for Your Project:
# 
# 1. Open a terminal window (or Anaconda Prompt on Windows).
# 2. Create a new environment named "prstools" with Python 3.10:
#     ```bash
#     conda create -n prstools python=3.10
#     ```
# 
# 3. Activate the environment:
#     ```bash
#     conda activate prstools
#     ```
# 
# ### Install R within the Environment:
# 
# Install R version 4.3.2 from the conda-forge channel:
# ```bash
# conda install -c conda-forge r-base=4.3.2
# conda install -c conda-forge r-essentials

# ## Data Preparation
# 
# ### Download Sample Data
# 
# The first step is to download the sample data. You can find the base data [here](https://drive.google.com/file/d/1RWjk49QNZj9zvJHc9X_wyZ51fdy6xQjv/view) and the target data [here](https://drive.google.com/file/d/1uhJR_3sn7RA8U5iYQbcmTp6vFdQiF4F2/view).
# 
# ### Extracting Target Data
# 
# Extract the target data and organize it with the following files in a directory named `SampleData1`:
# 
# - `SampleData1.bed`: Plink Files (bed, bim, fam) containing genotype information.
# - `SampleData1.bim`
# - `SampleData1.fam`
# - `SampleData1.cov`: Contains covariate information.
# - `SampleData1.height`: Contains the actual height phenotype.
# - `SampleData1.gz`: GWAS summary statistic file.
# 
# **Ensure to rename all files to the `SampleData1` prefix.**
# 
# ### View Bed Files
# 
# To view Bed files, visit [Plink Binary Files Documentation](https://zzz.bwh.harvard.edu/plink/binary.shtml).
# 
# 
# ### Analyzing Quantitative Trait (Height)
# 
# We will begin by analyzing the quantitative trait - height. The dataset used for the continuous trait analysis is sourced from the tutorial by [choishingwan](https://choishingwan.github.io/PRS-Tutorial/base/), and we express our gratitude for making it publicly available.
# 
# ### Polygenic Risk Score (PRS) Tools Overview
# 
# PRS tools can be categorized into three types:
# 
# 1. **Type 1: Summary Statistic File Only**
#    - These tools use a summary statistic file (e.g., gwas.txt) and estimate betas or posterior probabilities. They may require population-specific linkage disequilibrium calculations.
# 
# 2. **Type 2: Summary Statistic File and Individual Data Set**
#    - This category of PRS tools requires both a summary statistic file and an individual dataset. They optimize various hyperparameters for beta estimation or snp weight estimation.
# 
# 3. **Type 3: Individual Data Set Only**
#    - These tools focus on polygenic risk score calculation for individuals, without relying on a summary statistic file. They incorporate linkage equilibrium for betas or snp weight estimation.
# 
# "In these tutorials, our focus is exclusively on tools falling under the Type 2 category. This category is chosen because it enables the optimization of hyperparameters using both Genome-Wide Association Study (GWAS) data and individual datasets. This optimization can be further applied to test datasets. The pipeline is specifically designed for a cross-validation design, enhancing the robustness of our analyses."
# 
# ### Data Splitting and Hyperparameter Optimization
# 
# The target data is split into a 75% training or hyperparameter optimization set and a 25% testing set for final performance evaluation.
# 
# Hyperparameter optimization can be categorized into two types:
# 
# 1. **Quality of the Dataset**
#    - This includes considerations such as pruning and clumping during quality controls on the target data.
# 
# 2. **Tools-Specific Hyperparameters**
#    - When using the GWAS File and an individual dataset, specific hyperparameters, such as lambdas for lasso, may be considered.
# 
# 
# ### Full Credit to Shing Wan Choi
# 
# - GitHub: [Shing Wan Choi](https://github.com/choishingwan)
# 
# Tutorial Link from where the code has been inferred: [Shing Wan Choi's PRS Tutorial](https://choishingwan.github.io/PRS-Tutorial/base/)
# 

# In[3]:

 
import os
import pandas as pd
import subprocess
import sys

# Set the directory where the files are located
filedirec = sys.argv[1]

# Define file paths for different data files
BED = filedirec + os.sep + filedirec
BIM = filedirec + os.sep + filedirec+".bim"
FAM = filedirec + os.sep + filedirec+".fam"
COV = filedirec + os.sep + filedirec+".cov"
Height = filedirec + os.sep + filedirec+".height"
GWAS = filedirec + os.sep + filedirec+".gz"
 
 
bimfile = pd.read_csv(BIM, sep="\s+", header=None)
print("Columns of BIM file:")
print(bimfile.columns)
print("First 10 rows of BIM file:")

bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
 
 

temp = pd.read_csv(FAM, sep="\s+", header=None, nrows=10)
print("Columns of FAM file:")
print(temp.columns)
print("First 10 rows of FAM file:")
print(temp.head())

 
temp = pd.read_csv(COV, sep="\s+", nrows=10)
print("Columns of COV file:")
print(temp.columns)
print("First 10 rows of COV file:")
print(temp.head())

# Read the first 10 rows of the Height file (file containing actual height phenotypes)
temp = pd.read_csv(Height, sep="\s+", nrows=10)
print("Columns of Height file:")
print(temp.columns)
print("First 10 rows of Height file:")
print(temp.head())

# Read the first 10 rows of the GWAS file (GWAS summary statistic file)
temp = pd.read_csv(GWAS, sep="\s+", nrows=10)
print("Columns of GWAS file:")
print(temp.columns)
print("First 10 rows of GWAS file:")
print(temp.head())

 

temp = pd.read_csv(GWAS,compression= "gzip",sep="\s+",nrows=10)
print(temp.head())


# Read GWAS data from a compressed file using pandas
 
df = pd.read_csv(GWAS, compression="gzip",sep="\s+",on_bad_lines='skip')
# Display the initial number of rows in the dataframe
print("Initial number of SNPs:", len(df))
# Apply quality control steps: Filter SNPs based on Minor Allele Frequency (MAF) and Imputation Information Score (INFO)
#df = df.loc[(df['MAF'] > 0.01) & (df['INFO'] > 0.8)]


# Display the number of rows after applying the filters
print("Number of SNPs after quality control:", len(df))

# Remove duplicate SNPs based on the 'SNP' column
#df = df.drop_duplicates(subset='SNP')

# Display the number of rows after removing duplicate SNPs
#print("SNPs in GWAS after removing duplicate SNPs:", len(df))

# Remove ambiguous SNPs with complementary alleles (C/G or A/T) to avoid potential errors
df = df[~((df['A1'] == 'A') & (df['A2'] == 'T') |
          (df['A1'] == 'T') & (df['A2'] == 'A') |
          (df['A1'] == 'G') & (df['A2'] == 'C') |
          (df['A1'] == 'C') & (df['A2'] == 'G'))]

# Display the final number of SNPs after removing ambiguous SNPs
print("Final number of SNPs after removing ambiguous SNPs:", len(df))

 
 

df["match"] = df["CHR"].astype(str)+"_"+df["BP"].astype(str)+"_"+df["A2"].astype(str)+"_"+df["A1"].astype(str)
df.drop_duplicates(subset='match')
del df["match"]
 

print("Removing SNPs for which even a single row does not contain the required value:", len(df))

if (df['SNP'] == 'X').all():
    print("RSIDs are missing!")
    bimfile = pd.read_csv(filedirec+os.sep+filedirec+".bim", sep="\s+", header=None)
    bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str)+"_"+df["BP"].astype(str)+"_"+df["A2"].astype(str)+"_"+df["A1"].astype(str)
    
    df.drop_duplicates(subset='match', inplace=True)
    bimfile.drop_duplicates(subset='match', inplace=True)

    df = df[df['match'].isin(bimfile['match'].values)]
    bimfile = bimfile[bimfile['match'].isin(df['match'].values)]
    df = df[df['match'].isin(bimfile['match'].values)]
    bimfile = bimfile[bimfile['match'].isin(df['match'].values)]

    df.drop_duplicates(subset='match', inplace=True)
    bimfile.drop_duplicates(subset='match', inplace=True)    

    df["SNP"] = bimfile[1].values
    print("match",len(df))
    del df["match"]
    df.to_csv(GWAS,compression="gzip",sep="\t",index=None)   


    pass
else:
    df.drop_duplicates(subset='SNP', inplace=True)
    df.to_csv(GWAS,compression="gzip",sep="\t",index=None)
    print("RSID is present!")
    pass
#df.to_csv(GWAS,compression="gzip",sep="\t",index=None)
#exit(0)



"""
if (df['SNP'] == 'X').all():
    print("All entries in the column contain 'X'")
    bimfile = pd.read_csv(BIM, sep="\s+", header=None)
    bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str)+"_"+df["BP"].astype(str)+"_"+df["A1"].astype(str)+"_"+df["A2"].astype(str)
    print(len(df))
    print(len(bimfile))

    common_values = set(df["match"].values).intersection(bimfile["match"].values)
    print("Common SNPs",len(common_values))

    df = df[df['match'].isin(bimfile['match'].values)]
    del df['match']
else:
    print("Not all entries in the column contain 'X'")

print("Length of GWAS file after filtering ",len(df)) 
#df.to_csv(GWAS,compression="gzip",sep="\t",index=None)
#exit(0)


try:
    bimfile = pd.read_csv("ref.bim", sep="\s+", header=None)
    common_values = set(bimfile[1].values).intersection(df["SNP"].values)
    print("Common SNPs",len(common_values))
except:
    pass


try:
    bimfile = pd.read_csv("ref.bim", sep="\s+", header=None)
    bimfile2 = pd.read_csv(BIM, sep="\s+", header=None)
    bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
    bimfile2["match"] = bimfile2[0].astype(str)+"_"+bimfile2[3].astype(str)+"_"+bimfile2[4].astype(str)+"_"+bimfile2[5].astype(str)
     
    common_values = set(bimfile["match"].values).intersection(bimfile2["match"].values)
    print("Common SNPs",len(common_values))
except:
    pass

bimfile = pd.read_csv("ref.bim", sep="\s+", header=None)
print(bimfile.head())
bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
df["match"] = df["CHR"].astype(str)+"_"+df["BP"].astype(str)+"_"+df["A2"].astype(str)+"_"+df["A1"].astype(str)

df = df[df['match'].isin(bimfile['match'].values)]
bimfile = bimfile[bimfile['match'].isin(df['match'].values)]
df = df[df['match'].isin(bimfile['match'].values)]
print("match",len(df))
df["SNP"] = bimfile[1].values
print("match",len(df))
try:
    #bimfile = pd.read_csv(BIM, sep="\s+", header=None)
    #print(bimfile.head())

    bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str)+"_"+df["BP"].astype(str)+"_"+df["A2"].astype(str)+"_"+df["A1"].astype(str)
    df = df[df['match'].isin(bimfile['match'].values)]
    bimfile = bimfile[bimfile['match'].isin(df['match'].values)]
    df["SNP"] = bimfile[1].values
    print("match",len(df))
except:
    pass

exit(0)





if (df['SNP'] == 'X').all():
    print("All entries in the column contain 'X'")
    bimfile = pd.read_csv("ref.bim", sep="\s+", header=None)
    bimfile["match"] = bimfile[0].astype(str)+"_"+bimfile[3].astype(str)+"_"+bimfile[4].astype(str)+"_"+bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str)+"_"+df["BP"].astype(str)+"_"+df["A1"].astype(str)+"_"+df["A2"].astype(str)
    df = df[df['match'].isin(bimfile['match'].values)]
    del df['match']
else:
    print("Not all entries in the column contain 'X'")

print("Length of GWAS file after filtering ",len(df)) 
#
#exit(0)
"""
 
 


# ## Individual genotype data (Target Data) Processing
# 
# 
# Ensure that the phenotype file, FAM file, and covariate file contain an identical number of samples. Remove any missing samples based on your data. Note that the extent of missingness in phenotypes and covariates may vary.
# 
# 
# **Note:** Plink needs to be installed or placed in the same directory as this notebook.
# 
# [Download Plink](https://www.cog-genomics.org/plink/)
# 
# We recommend using Linux. In cases where Windows is required due to package installation issues on Linux, we provide the following guidance:
# 
# 1. For Windows, use `plink`.
# 2. For Linux, use `./plink`.
# 

# In[16]:


# New files to be saved with QC suffix
newfilename = filedirec + "_QC"

# Read information from FAM file
f = pd.read_csv(FAM, header=None, sep="\s+", names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"])
print("FAM file contents:")
print(f.head())
print("Total number of people in FAM file:", len(f))

 
h = pd.read_csv(Height, sep="\t")
h = h.dropna(subset=["Height"])
print("Phenotype information is available for:", len(h), "people")
print(len(h))
result = pd.merge(f, h, on=['FID', 'IID'])

# Replace 'Phenotype' column with 'Height' and save to a new PeopleWithPhenotype.txt file
result["Phenotype"] = result["Height"].values
del result["Height"]
print(result)
result.to_csv(filedirec + os.sep + "PeopleWithPhenotype.txt", index=False, header=False, sep="\t")
#result.to_csv(FAM, index=False, header=False, sep="\t")

# Use plink to keep only the people with phenotype present
plink_command = [
    './plink',
    '--bfile', filedirec + os.sep + filedirec,
    '--keep', filedirec + os.sep + "PeopleWithPhenotype.txt",
    '--make-bed',
    '--out', filedirec + os.sep + newfilename
]
subprocess.run(plink_command)

# Update the phenotype information in the new FAM file
f = pd.read_csv(filedirec + os.sep + newfilename + ".fam", header=None, sep="\s+",
                names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"])
f["Phenotype"] = result["Phenotype"].values
f.to_csv(filedirec + os.sep + newfilename + ".fam", index=False, header=False, sep="\t")

# Update the covariate file as well
covfile = filedirec + os.sep + filedirec + '.cov'
covfile = pd.read_csv(covfile, sep="\s+")

print("Covariate file contents:")
print(covfile.head())
print("Total number of people in Covariate file:", len(covfile))

# Match the FID and IID from covariate and height file
covfile = covfile[covfile['FID'].isin(f["FID"].values) & covfile['IID'].isin(f["IID"].values)]
print("Covariate file contents after matching with FAM file:")
print(covfile.head())
print("Total number of people in Covariate file after matching:", len(covfile))
covfile.to_csv(filedirec + os.sep + newfilename + ".cov", index=None, sep="\t")

 



import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import subprocess
from sklearn.model_selection import KFold, cross_val_score

# Step 1: Read the Fam file

input_file_path = filedirec+os.sep+newfilename+'.fam'
df = pd.read_csv(input_file_path,sep="\s+",header=None)

# Step 2: Create 5 directories for storing fold information
output_directory_base = filedirec
os.makedirs(output_directory_base, exist_ok=True)

# Step 3: Split the data into 5 folds using Stratified K-Folds
fold_column = 5  # fifth column contains phenotypes
kf = KFold(n_splits=5, shuffle=True, random_state=42)
phenotype_col = 5

# The following code is for continous phenotype.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
column_values = df[phenotype_col].unique()

if set(column_values) == {1, 2}:
    print("The column contains only 0 and 1.")
    #exit(0)
    for fold_id, (train_index, test_index) in enumerate(skf.split(df, df[phenotype_col])):
        fold_directory = os.path.join(output_directory_base, f'Fold_{fold_id}')

        train_file_name = "train_data"
        test_file_name = "test_data"

        new_train_file_name = "train_data.QC"
        new_test_file_name = "test_data.QC"

        os.makedirs(fold_directory, exist_ok=True)
        #"""
        # Save train and test data to separate CSV files
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        
        train_data.to_csv(os.path.join(fold_directory, 'train_data.fam'),sep="\t",header=False,index=False)
        test_data.to_csv(os.path.join(fold_directory, 'test_data.fam'),sep="\t",header=False,index=False)
        #print(train_data)
        #exit(0)
        
        #exit(0)
        # Step 4: Use PLINK to extract test and train samples for each fold
        plink_train_command = [
            './plink',
            '--bfile', filedirec+os.sep+newfilename,
            '--keep', os.path.join(fold_directory, train_file_name+'.fam'),
            '--make-bed',
            '--out', os.path.join(fold_directory, train_file_name)
        ]

        plink_test_command = [
            './plink',
            '--bfile', filedirec+os.sep+newfilename,
            '--keep', os.path.join(fold_directory, test_file_name+'.fam'),
            '--make-bed',
            '--out', os.path.join(fold_directory, test_file_name)
        ]

        subprocess.run(plink_train_command)
        subprocess.run(plink_test_command)
        
        covfile = filedirec+os.sep+newfilename+'.cov'
        covfile = pd.read_csv(covfile)

        cov_train_data = covfile.iloc[train_index]
        cov_test_data = covfile.iloc[test_index]

        cov_train_data.to_csv(os.path.join(fold_directory, train_file_name+'.cov'),sep=",",index=False)
        cov_test_data.to_csv(os.path.join(fold_directory, test_file_name+'.cov'),sep=",",index=False)

        #exit(0)
        ### perform Quality controls on the training data only.
        print(os.path.join(fold_directory, train_file_name))
        plink_command_1 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            '--maf', '0.01',
            '--hwe', '1e-6',
            '--geno', '0.1',
            '--mind', '0.1',
            '--write-snplist',
            '--make-just-fam',
            '--out', os.path.join(fold_directory, new_train_file_name)
        ]

        subprocess.run(plink_command_1)
        
        # Command 2
        plink_command_2 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            '--keep', os.path.join(fold_directory, new_train_file_name+'.fam'),
            '--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
            '--indep-pairwise', '200', '50', '0.25',
            '--out', os.path.join(fold_directory, new_train_file_name)
        ]

        #subprocess.run(plink_command_2)
        
        # Command 3
        plink_command_3 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            #'--extract', os.path.join(fold_directory, new_train_file_name+'.prune.in'),
            '--keep', os.path.join(fold_directory, new_train_file_name+'.fam'),
            #'--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
            #'--keep', os.path.join(fold_directory, new_train_file_name+'.fam'),
            '--het',
            '--out', os.path.join(fold_directory, new_train_file_name)
        ]

        subprocess.run(plink_command_3)
        # Invoked R functions.
        
        os.system("Rscript Module1.R "+os.path.join(fold_directory)+"  "+train_file_name+" "+new_train_file_name+ " "+"1")
        
        
        #exit(0)
        
        # Code for sex check: The sample data have 22 chromosome so this operation is skipped.
        #"""
        plink_command = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        #'--extract', os.path.join(fold_directory, 'train_data.QC.prune.in'),
        '--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
        '--keep', os.path.join(fold_directory, train_file_name+'.valid.sample'),
        '--check-sex',
        '--out', os.path.join(fold_directory, new_train_file_name)
        ]
        
        # Invoke the PLINK command using subprocess for sex check
        #subprocess.run(plink_command)
        #os.system("Rscript Module1.R "+os.path.join(fold_directory)+"  "+train_file_name+" "+new_train_file_name+ " "+"2")
        
        #"""

        plink_command_1 = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        #'--extract', os.path.join(fold_directory, new_train_file_name+'.prune.in'),
        # If you use sex check then use the following line. Otherwise it is not required.
        '--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
        #'--keep', os.path.join(fold_directory, 'train_data.QC.valid'),
        '--rel-cutoff', '0.125',
        '--out', os.path.join(fold_directory, new_train_file_name)
        ]
        
        subprocess.run(plink_command_1)
   
        plink_command_2 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            '--make-bed',
            '--keep', os.path.join(fold_directory, new_train_file_name+'.rel.id'),
            '--out', os.path.join(fold_directory, new_train_file_name),
            #'--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
            #'--exclude', os.path.join(fold_directory, train_file_name+'.mismatch'),
            '--a1-allele', os.path.join(fold_directory, train_file_name+'.a1')
        ]

        subprocess.run(plink_command_2)
        #exit(0)
else:
    print("The column does not contain only 0 and 1.")
    for fold_id, (train_index, test_index) in enumerate(kf.split(df, df[phenotype_col])):
        fold_directory = os.path.join(output_directory_base, f'Fold_{fold_id}')

        train_file_name = "train_data"
        test_file_name = "test_data"

        new_train_file_name = "train_data.QC"
        new_test_file_name = "test_data.QC"

        os.makedirs(fold_directory, exist_ok=True)
        #"""
        # Save train and test data to separate CSV files
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        
        train_data.to_csv(os.path.join(fold_directory, 'train_data.fam'),sep="\t",header=False,index=False)
        test_data.to_csv(os.path.join(fold_directory, 'test_data.fam'),sep="\t",header=False,index=False)
        #print(train_data)
        #exit(0)
        
        #exit(0)
        # Step 4: Use PLINK to extract test and train samples for each fold
        plink_train_command = [
            './plink',
            '--bfile', filedirec+os.sep+newfilename,
            '--keep', os.path.join(fold_directory, train_file_name+'.fam'),
            '--make-bed',
            '--out', os.path.join(fold_directory, train_file_name)
        ]

        plink_test_command = [
            './plink',
            '--bfile', filedirec+os.sep+newfilename,
            '--keep', os.path.join(fold_directory, test_file_name+'.fam'),
            '--make-bed',
            '--out', os.path.join(fold_directory, test_file_name)
        ]

        subprocess.run(plink_train_command)
        subprocess.run(plink_test_command)
        
        covfile = filedirec+os.sep+newfilename+'.cov'
        covfile = pd.read_csv(covfile)

        cov_train_data = covfile.iloc[train_index]
        cov_test_data = covfile.iloc[test_index]

        cov_train_data.to_csv(os.path.join(fold_directory, train_file_name+'.cov'),sep=",",index=False)
        cov_test_data.to_csv(os.path.join(fold_directory, test_file_name+'.cov'),sep=",",index=False)

        #exit(0)
        ### perform Quality controls on the training data only.
        print(os.path.join(fold_directory, train_file_name))
        plink_command_1 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            '--maf', '0.01',
            '--hwe', '1e-6',
            '--geno', '0.1',
            '--mind', '0.1',
            '--write-snplist',
            '--make-just-fam',
            '--out', os.path.join(fold_directory, new_train_file_name)
        ]

        subprocess.run(plink_command_1)
        
        # Command 2
        plink_command_2 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            '--keep', os.path.join(fold_directory, new_train_file_name+'.fam'),
            '--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
            '--indep-pairwise', '200', '50', '0.25',
            '--out', os.path.join(fold_directory, new_train_file_name)
        ]

        #subprocess.run(plink_command_2)
        
        # Command 3
        plink_command_3 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            #'--extract', os.path.join(fold_directory, new_train_file_name+'.prune.in'),
            '--keep', os.path.join(fold_directory, new_train_file_name+'.fam'),
            #'--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
            #'--keep', os.path.join(fold_directory, new_train_file_name+'.fam'),
            '--het',
            '--out', os.path.join(fold_directory, new_train_file_name)
        ]

        subprocess.run(plink_command_3)
        # Invoked R functions.
        
        os.system("Rscript Module1.R "+os.path.join(fold_directory)+"  "+train_file_name+" "+new_train_file_name+ " "+"1")
        
        
        #exit(0)
        
        # Code for sex check: The sample data have 22 chromosome so this operation is skipped.
        #"""
        plink_command = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        #'--extract', os.path.join(fold_directory, 'train_data.QC.prune.in'),
        '--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
        '--keep', os.path.join(fold_directory, train_file_name+'.valid.sample'),
        '--check-sex',
        '--out', os.path.join(fold_directory, new_train_file_name)
        ]
        
        # Invoke the PLINK command using subprocess for sex check
        #subprocess.run(plink_command)
        #os.system("Rscript Module1.R "+os.path.join(fold_directory)+"  "+train_file_name+" "+new_train_file_name+ " "+"2")
        
        #"""

        plink_command_1 = [
        './plink',
        '--bfile', os.path.join(fold_directory, train_file_name),
        #'--extract', os.path.join(fold_directory, new_train_file_name+'.prune.in'),
        # If you use sex check then use the following line. Otherwise it is not required.
        '--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
        #'--keep', os.path.join(fold_directory, 'train_data.QC.valid'),
        '--rel-cutoff', '0.125',
        '--out', os.path.join(fold_directory, new_train_file_name)
        ]
        
        subprocess.run(plink_command_1)
   
        plink_command_2 = [
            './plink',
            '--bfile', os.path.join(fold_directory, train_file_name),
            '--make-bed',
            '--keep', os.path.join(fold_directory, new_train_file_name+'.rel.id'),
            '--out', os.path.join(fold_directory, new_train_file_name),
            #'--extract', os.path.join(fold_directory, new_train_file_name+'.snplist'),
            #'--exclude', os.path.join(fold_directory, train_file_name+'.mismatch'),
            '--a1-allele', os.path.join(fold_directory, train_file_name+'.a1')
        ]

        subprocess.run(plink_command_2)
        #exit(0)
 


# ### Check final files.

# In[15]:


import os

# List of file names to check for existence
f = [
    "train_data.QC.bed",
    "train_data.QC.bim",
    "train_data.QC.fam",
    "train_data.cov",
    
    "test_data.bed",
    "test_data.bim",
    "test_data.fam",
    "test_data.cov"
]

# Specify the fold number
fold_number = 1

# Loop through each file name in the list
for loop in f:
    # Check if the file exists in the specified directory for the given fold
    if os.path.exists(filedirec + os.sep + "Fold_" + str(fold_number) + os.sep + loop):
        # Print a message indicating that the file exists
        print(loop, "Yes, the file exists.")
    else:
        # Print a message indicating that the file does not exist
        print(loop, "No, the file does not exist.")


# ../bolt \ --bfile=EUR_subset \ --remove=EUR_subset.remove \ --exclude=EUR_subset.exclude \ --exclude=EUR_subset.exclude2 \ --phenoFile=EUR_subset.pheno2.covars \ --phenoCol=PHENO   \ --lmm \ --LDscoresFile=../tables/LDSCORE.1000G_EUR.tab.gz \ --numThreads=2 \ --statsFile=example.stats \  

# python polyfun.py     --compute-h2-L2     --no-partitions     --output-prefix output/testrun     --sumstats example_data/sumstats.parquet     --ref-ld-chr /data/ascher01/uqmmune1/PGS/kathy_data/pgs_input/LDSC_FILES/baselineLD.1.l2.ldscore.gz     --w-ld-chr /data/ascher01/uqmmune1/PGS/kathy_data/pgs_input/LDSC_FILES/weights.

# python extract_snpvar.py --sumstats ../SampleData1/SampleData1.gz --out output/snps_with_var.gz

# Rscript npsR/nps_val.R ../SampleData1/Fold_0/npsdat --val-dataset change.test

# python3 /data/ascher01/uqmmune1/PGS/finalprstools/polyfun/finemapper.py --chr 1 --start 1 --end 3000001 --out SampleData1/polyfun_output.chr1.1_3000001.gz --ld /UKBB_LD/chr1_1_3000001 --method susie --sumstats SampleData1/SampleData1.temppolyfun.parquet --n 40000 --memory 1 --max-num-causal 5 --non-funct

# python3 /data/ascher01/uqmmune1/PGS/finalprstools/polyfun/finemapper.py --chr 13 --start 96000001 --end 99000001 --out SampleData1/polyfun_output.chr13.96000001_99000001.gz --ld UKBB_LD/temp/chr13_96000001_99000001 --method susie --sumstats SampleData1/SampleData1.temppolyfun.parquet --n 40000 --memory 1 --max-num-causal 5 --non-funct

# 

# 
# 
