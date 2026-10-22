#!/usr/bin/env python
# coding: utf-8

# # LDpred-2 
# 
# In this notebook, we will use LDpred-2 to calculate the Polygenic Risk Score (PRS).
# 
# **Note:** LDpred-2 needs to be installed using R.
# 
# We will use the same flow of the code, and when we have to invoke LDpred-2 functions, we will call the script using Python. When LDpred-2 finishes executions, results will be stored in the specific directories for each fold and will be retrieved in Python for further calculations.
# 
# ## Install LDpred-2
# 
# Install LDpred-2 using the following commands:
# 
# 1. Activate the conda Environment.
# 2. Type R
# 3. Run the following commands:
# 
# ```R
# install.packages("remotes")
# library(remotes)
# remotes::install_github("https://github.com/privefl/bigsnpr.git")
# ```
# 
# The content in this notebook has been taken from the [PRS Tutorial](https://choishingwan.github.io/PRS-Tutorial/ldpred/), [LDpred-2 Documentation](https://privefl.github.io/bigsnpr/articles/LDpred2.html), and [Polygenic Scores (PGS)](https://privefl.github.io/bigsnpr-extdoc/polygenic-scores-pgs.html).
# 
# Thanks to Florian Privé for creating one of the best documentations explaining each aspect of the calculation. We will be using LDpred-2 for cross-validation design and hyper-parameter optimization, and this tutorial complements the existing one.
# 
# 

# ## LDpred-2  Hyperparameters
# 
# ### Hyperparameters for LDpred-2  performed using PLINK
# 
# #### Pruning Parameters
# 
# Informs Plink that we wish to perform pruning with a window size of 200 variants, sliding across the genome with a step size of 50 variants at a time, and filter out any SNPs with LD \( r^2 \) higher than 0.25.
# 
# 
# ```python
# 1. p_window_size = [200]
# 2. p_slide_size = [50]
# 3. p_LD_threshold = [0.25]
# ```
# 
# #### Clumping Parameters
# 
# The P-value threshold for an SNP to be included. 1 means to include all SNPs for clumping. SNPs having \( r^2 \) higher than 0.1 with the index SNPs will be removed. SNPs within 200k of the index SNP are considered for clumping.
# 
# ```python
# 1. clump_p1 = [1]
# 2. clump_r2 = [0.1]
# 3. clump_kb = [200]
# ```
# #### PCA
# Pca also affects the results evident from the initial analysis; however, including more PCA overfits the model. 
# 
# ### Hyperparameters for LDpred-2 
#  
# 
# 
# #### Heritability
# 
# To calculate h2 using LDpred-2, we followed the [LDpred-2 tutorial](https://choishingwan.github.io/PRS-Tutorial/ldpred/).
# The code for this part is in R, as LDpred-2 is in R.
# 
# 1. Using SNPs from the HapMap as preferred by the authors. The name in the code is `LDpred-2_hapmap`.
# 2. Using all the SNPs. The name in the code is `LDpred-2_full`.
# 
# This approach is computationally expensive, as the correlation between all the SNPs in a specific chromosome is being calculated.
# 
# ## LDpred-2 Models
# 
# Each model has its own set of hyperparameters, so first, we will generate those hyperparameters for those models and call them when invoking LDpred-2.
# 
# Use the following R code to know more about the hyperparameters:
# - `help(snp_ldpred2_inf)`
# - `help(snp_ldpred2_grid)`
# - `help(snp_ldpred2_auto)`
# - `help(snp_lassosum2)`
# 
# ### 1. LDpred2-inf: Infinitesimal Model
# 
# No Hyperparameters
# 
# ### 2. LDpred2(-grid): Grid of Models
# 
# Each model has its own set of parameters:
# - `burn_in = 100`
# - `num_iter = 10`
# - `p = c(1e-4, 1e-5)`
# - `h2 = c(0.1, 0.2)`
# - `sparse = c(FALSE, TRUE)`
# 
# ### 3. LDpred2-auto: Automatic Model
# 
# - `burn_in = 100`
# - `num_iter = 100`
# - `sparse = c(FALSE, TRUE)`
# - `alpha_bounds = c(1, 0.5)`
# - `use_MLE = c(FALSE, TRUE)`
# - `p = c(1e-4, 1e-5)`
# - `shrink_corr = 0.7`
# - `allow_jump_sign = c(FALSE, TRUE)`
# 
# ### 4. lassosum2: Grid of Models
# 
# - `lambda = c(0.001, 0.01, 0.1, 1)`
# - `delta = 30`
# 
# We will execute the code for each mkodel seperatly.
# 

# ### GWAS file processing for Plink.
# When the effect size relates to disease risk and is thus given as an odds ratio (OR) rather than BETA (for continuous traits), the PRS is computed as a product of ORs. To simplify this calculation, take the natural logarithm of the OR so that the PRS can be computed using summation instead.

# In[ ]:

import os
import sys
import pandas as pd
import numpy as np

filedirec = sys.argv[1]

GWAS = filedirec + os.sep + filedirec+".gz"
df = pd.read_csv(GWAS,compression= "gzip",sep="\s+")

import pandas as pd
if "BETA" in df.columns.to_list():
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    df['Z'] = df['BETA'] / df['SE']
    #df = df[['SNP','N','Z','A1', 'A2']]


else:
    df["BETA"] = np.log(df["OR"])
    df = df[['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF']]
    df['Z'] = df['BETA'] / df['SE']
    #df = df[['SNP','N','Z','A1', 'A2']]

df.to_csv(filedirec + os.sep +filedirec+".txt",sep="\t",index=False)










import subprocess

command = [
    "python ldsc.py",
    "--h2", filedirec + os.sep +filedirec+".txt",
    "--ref-ld-chr", "eur_w_ld_chr/",
    "--w-ld-chr", "eur_w_ld_chr/",
    "--out", "scz_h2"
]

print(" ".join(command))
# Execute the command
os.system(" ".join(command))


exit(0)


# ### Define Hyperparameters
# 
# Define hyperparameters to be optimized and set initial values.
# 
# ### Extract Valid SNPs from Clumped File
# 
# For Windows, download `gwak`, and for Linux, the `awk` command is sufficient. For Windows, `GWAK` is required. You can download it from [here](https://sourceforge.net/projects/gnuwin32/). Get it and place it in the same directory.
# 
#     

# ## 1. LDpred2-inf: Infinitesimal Model

# In[7]:


from operator import index
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import pandas as pd
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

def create_directory(directory):
    """Function to create a directory if it doesn't exist."""
    if not os.path.exists(directory):  # Checking if the directory doesn't exist
        os.makedirs(directory)  # Creating the directory if it doesn't exist
    return directory  # Returning the created or existing directory

filedirec = sys.argv[1]  # Setting the base directory path

foldnumber = sys.argv[2]
#foldnumber = "1"  # Setting 'foldnumber' to "0"

folddirec = filedirec + os.sep + "Fold_" + foldnumber  # Creating a directory path for the specific fold
trainfilename = "train_data"  # Setting the name of the training data file
newtrainfilename = "train_data.QC"  # Setting the name of the new training data file

testfilename = "test_data"  # Setting the name of the test data file
newtestfilename = "test_data.QC"  # Setting the name of the new test data file

# Number of PCA to be included as a covariate.
numberofpca = ["6"]  # Setting the number of PCA components to be included

# Clumping parameters.
clump_p1 = [1]  # List containing clump parameter 'p1'
clump_r2 = [0.1]  # List containing clump parameter 'r2'
clump_kb = [200]  # List containing clump parameter 'kb'

# Pruning parameters.
p_window_size = [200]  # List containing pruning parameter 'window_size'
p_slide_size = [50]  # List containing pruning parameter 'slide_size'
p_LD_threshold = [0.25]  # List containing pruning parameter 'LD_threshold'

# Kindly note that the number of p-values to be considered varies, and the actual p-value depends on the dataset as well.
# We will specify the range list here.

minimumpvalue = 4  # Minimum p-value in exponent
numberofintervals = 10  # Number of intervals to be considered
allpvalues = np.logspace(-minimumpvalue, 0, numberofintervals, endpoint=True)  # Generating an array of logarithmically spaced p-values
count = 1
with open(folddirec + os.sep + 'range_list', 'w') as file:
    for value in allpvalues:
        #file.write(f'pv_{value} 0 {value}\n')  # Writing range information to the 'range_list' file
        count = count + 1

pvaluefile = folddirec + os.sep + 'range_list'

# Initializing an empty DataFrame with specified column names
prs_result = pd.DataFrame(columns=["clump_p1", "clump_r2", "clump_kb", "p_window_size", "p_slide_size", "p_LD_threshold",
                                "numberofpca","h2model","h2","numberofvariants","clumpprune","relatedmatrix","data"])

 
def transform_gemma_data(traindirec, newtrainfilename,models,p, clumpprune,relatedmatrix,data,p1_val, p2_val, p3_val, c1_val, c2_val, c3_val,Name,pvaluefile):

    ### First perform clumping on the file and save the clumpled file.
    
    #clupmedfile = traindirec+os.sep+newtrainfilename+".clump"
    #prunedfile = traindirec+os.sep+newtrainfilename+".clump.prune"
 

    command = [
    "./plink",
    "--bfile", traindirec+os.sep+newtrainfilename,
    "--indep-pairwise", p1_val, p2_val, p3_val,
    "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)
    # First perform pruning and then clumping and the pruning.
 
    command = [
    "./plink",
    "--bfile", traindirec+os.sep+newtrainfilename,
    "--clump-p1", c1_val,
    "--extract", traindirec+os.sep+trainfilename+".prune.in",
    "--clump-r2", c2_val,
    "--clump-kb", c3_val,
    "--clump", filedirec+os.sep+filedirec+".txt",
    "--clump-snp-field", "SNP",
    "--clump-field", "P",
    "--out", traindirec+os.sep+trainfilename
    ]    
    subprocess.run(command)

    # Extract the valid SNPs from th clumped file.
    # For windows download gwak for linux awk commmand is sufficient.
    ### For windows require GWAK.
    ### https://sourceforge.net/projects/gnuwin32/
    ##3 Get it and place it in the same direc.
    os.system("gawk "+"\""+"NR!=1{print $3}"+"\"  "+ traindirec+os.sep+trainfilename+".clumped >  "+traindirec+os.sep+trainfilename+".valid.snp")
    #print("gawk "+"\""+"NR!=1{print $3}"+"\"  "+ traindirec+os.sep+trainfilename+".clumped >  "+traindirec+os.sep+trainfilename+".valid.snp")

    #Linux:
    #os.system("awk "+"\""+"NR!=1{print $3}"+"\"  "+ traindirec+os.sep+trainfilename+".clumped >  "+traindirec+os.sep+trainfilename+".valid.snp")
    #print("awk "+"\""+"NR!=1{print $3}"+"\"  "+ traindirec+os.sep+trainfilename+".clumped >  "+traindirec+os.sep+trainfilename+".valid.snp")


    command = [
    "./plink",
    "--make-bed",
    "--bfile", traindirec+os.sep+newtrainfilename,
    "--indep-pairwise", p1_val, p2_val, p3_val,
    "--extract", traindirec+os.sep+trainfilename+".valid.snp",
    "--out", traindirec+os.sep+newtrainfilename+".clumped.pruned"
    ]
    subprocess.run(command)
    
    # Also extract the PCA at this point.
    command = [
        "./plink",
        "--bfile", traindirec+os.sep+newtrainfilename+".clumped.pruned",
        "--extract", traindirec+os.sep+trainfilename+".valid.snp",
        "--pca", p,
        "--out", traindirec+os.sep+trainfilename
    ]
    subprocess.run(command)
    
    # At this stage, we will merge the PCA and COV file. 
    tempphenotype_train = pd.read_table(traindirec+os.sep+newtrainfilename+".clumped.pruned"+".fam", sep="\s+",header=None)
    phenotype = pd.DataFrame()
    phenotype = tempphenotype_train[[0,1,5]]
    phenotype.to_csv(traindirec+os.sep+trainfilename+".PHENO",sep="\t",header=['FID', 'IID', 'PHENO'],index=False)
 
    #pcs_train = pd.read_table(traindirec+os.sep+trainfilename+".eigenvec", sep="\s+",header=None, names=["FID", "IID"] + [f"PC{str(i)}" for i in range(1, int(p)+1)])
    covariate_train = pd.read_table(traindirec+os.sep+trainfilename+".cov",sep="\s+")
    covariate_train.iloc[:, 2:].to_csv(traindirec+os.sep+trainfilename+".covgemma", header=False, index=False,sep="\t")
 
    covariate_train.fillna(0, inplace=True)
    print(covariate_train.head())
    print(len(covariate_train))
    covariate_train = covariate_train[covariate_train["FID"].isin(pcs_train["FID"].values) & covariate_train["IID"].isin(pcs_train["IID"].values)]
    print(len(covariate_train))
 
    covariate_train['FID'] = covariate_train['FID'].astype(str)
    pcs_train['FID'] = pcs_train['FID'].astype(str)
    covariate_train['IID'] = covariate_train['IID'].astype(str)
    pcs_train['IID'] = pcs_train['IID'].astype(str)
    covandpcs_train = pd.merge(covariate_train, pcs_train, on=["FID","IID"])
    covandpcs_train.to_csv(traindirec+os.sep+trainfilename+".COV_PCA",sep="\t",index=False)
    covandpcs_train.iloc[:, 2:].to_csv(traindirec+os.sep+trainfilename+".COV_PCAgemma", header=False, index=False,sep="\t")
 
 
    relatedmatrixname = ""
    if relatedmatrix=="1":
        relatedmatrixname = "centered"
    else: 
        relatedmatrixname = "standardized"
    
    h2modelname = ""

    if models=="1":
        h2modelname = "HE regression"
    else: 
        h2modelname = "REML AI algorithm"

    temppve = ""

    if clumpprune=="yes":
        BFILE = traindirec+os.sep+newtrainfilename+".clumped.pruned"
    else:
        BFILE = traindirec+os.sep+newtrainfilename
    
    import shutil
    shutil.rmtree("output", ignore_errors=True)

    def readpve():
        file_path = "output"+os.sep+"output.log.txt"  
        with open(file_path, 'r') as file:
            for line in file:
                if "pve estimates" in line:
                    pve_estimate = line.split("=")[-1].strip()
                    print("PVE Estimate:", pve_estimate)
                    return pve_estimate

    if data == "genotype":
         
        print("Processing genotype data")
        subprocess.run(["./DPR",
                "--bfile", BFILE,
                "-gk", relatedmatrix,
                "-o", "output"])
 

        # Command 2: ./gemma -p train_data.fam -k output/output.cXX.txt -n 6 -vc 1 -o output
        subprocess.run(["./gemma",
                        "-p", BFILE+".fam",
                        "-k", "output/output.cXX.txt",
                        "-n", "6",
                        "-vc", model,
                        "-o", "output"])
 
        temppve =  readpve()             
        print("PVE with only genotype data",readpve())

        subprocess.run(["./DPR",
                "--bfile", BFILE,
                "-dpr", "1",
                "-o", "output"])

    exit(0)

    if data == "genotype_covariate":
        subprocess.run(["./gemma",
                "--bfile", BFILE,
                "-gk", relatedmatrix,
                "-o", "output"])      

        subprocess.run([
                './gemma',
                '-p', BFILE+".fam",
                '-k', 'output/output.cXX.txt',
                '-n', '6',
                '-vc', model,
                '-c', traindirec+os.sep+trainfilename+".covgemma",
                '-o', 'output'
            ])
        print(" ".join([
                './gemma',
                '-p', BFILE+".fam",
                '-k', 'output/output.cXX.txt',
                '-n', '6',
                '-vc', model,
                '-c', traindirec+os.sep+trainfilename+".covgemma",
                '-o', 'output'
            ]))
        temppve =  readpve()             
        print("PVE with only genotype data",readpve())


    if data == "genotype_covariate_pca":
        subprocess.run(["./gemma",
                "--bfile", BFILE,
                "-gk", relatedmatrix,
                "-o", "output"])         
         
        subprocess.run([
                './gemma',
                '-p', BFILE+".fam",
                '-k', 'output/output.cXX.txt',
                '-n', '6',
                '-vc', model,
                '-c', traindirec+os.sep+trainfilename+".COV_PCAgemma",
                '-o', 'output'
            ])

        temppve =  readpve()             
        print("PVE with only genotype data",readpve())




    global prs_result 
    prs_result = prs_result._append({
        "clump_p1": c1_val,
        "clump_r2": c2_val,
        "clump_kb": c3_val,
        "p_window_size": p1_val,
        "p_slide_size": p2_val,
        "p_LD_threshold": p3_val,
        "h2":temppve,
        "h2model":h2modelname,
        "clumpprune":clumpprune,
        #"numberofvariants":tempdata2,
        "numberofpca":p,
        "relatedmatrix":relatedmatrixname,
        "data":data,
         
    }, ignore_index=True)

    prs_result.to_csv(traindirec+os.sep+Name+os.sep+"Results.csv",index=False)
    
 
    return

 
#-gk 1 calculates the centered relatedness matrix while “-gk 2” calculates the standardized relatedness matrix; “ 
relatedmatrixs = ["1","2"]
#”-vc 1” (default) uses HE regression and ”-vc 2” uses REML AI algorithm;
h2models = ["1","2"]
datas = ["genotype","genotype_covariate","genotype_covariate_pca"]
#datas = ["genotype_covariate","genotype_covariate_pca"]

clumpprunes = ["yes","no"]
# Nested loops to iterate over different parameter values
create_directory(folddirec+os.sep+"gemma")
for p1_val in p_window_size:
 for p2_val in p_slide_size: 
  for p3_val in p_LD_threshold:
   for c1_val in clump_p1:
    for c2_val in clump_r2:
     for c3_val in clump_kb:
      for p in numberofpca:
       for model in  h2models:
        for clumpprune in clumpprunes:
         for relatedmatrix in relatedmatrixs:
          for data  in datas:
            transform_gemma_data(folddirec, newtrainfilename,model, p, clumpprune,relatedmatrix,data,str(p1_val), str(p2_val), str(p3_val), str(c1_val), str(c2_val), str(c3_val), "gemma", pvaluefile)
            #exit(0)
exit(0)




