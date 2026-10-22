import sys
import os
import pandas as pd

# Process each phenotype.
phenotype = sys.argv[1]

# Function to detect the genomic build of the GWAS and update the genotype .bed file.
def detect_genome_build(phenotype):
    # Define the path to the GWAS file for the given phenotype
    GWAS = phenotype + os.sep + phenotype + ".gz"
    
    # Read the GWAS file
    df = pd.read_csv(GWAS, compression="gzip", sep="\s+", on_bad_lines='skip')
    
    # Read the bim file for hg19 build
    bimfile = pd.read_csv("genotypes.bim19", sep="\s+", header=None)
    
    # Create a matching column in bimfile and df based on chromosome, base pair location, and alleles
    bimfile["match"] = bimfile[0].astype(str) + "_" + bimfile[3].astype(str) + "_" + bimfile[4].astype(str) + "_" + bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str) + "_" + df["BP"].astype(str) + "_" + df["A1"].astype(str) + "_" + df["A2"].astype(str)
    
    # Remove duplicates in the match columns
    df.drop_duplicates(subset='match', inplace=True)
    bimfile.drop_duplicates(subset='match', inplace=True)
    
    # Count the number of variants matching the hg19 build
    hg19variants = len(df[df['match'].isin(bimfile['match'].values)])
    print(hg19variants)

    # Re-read the GWAS file to reset the dataframe
    df = pd.read_csv(GWAS, compression="gzip", sep="\s+", on_bad_lines='skip')
    
    # Read the bim file for hg38 build
    bimfile = pd.read_csv("genotypes.bim38", sep="\s+", header=None)
    
    # Create a matching column in bimfile and df based on chromosome, base pair location, and alleles
    bimfile["match"] = bimfile[0].astype(str) + "_" + bimfile[3].astype(str) + "_" + bimfile[4].astype(str) + "_" + bimfile[5].astype(str)
    df["match"] = df["CHR"].astype(str) + "_" + df["BP"].astype(str) + "_" + df["A1"].astype(str) + "_" + df["A2"].astype(str)
    
    # Remove duplicates in the match columns
    df.drop_duplicates(subset='match', inplace=True)
    bimfile.drop_duplicates(subset='match', inplace=True)
    
    # Count the number of variants matching the hg38 build
    hg38variants = len(df[df['match'].isin(bimfile['match'].values)])
    print(hg38variants)

    # Return the build with more matching variants
    if hg19variants > hg38variants:
        return "19"
    if hg38variants > hg19variants:
        return "38"

# Check the detected genome build and copy the corresponding bim file to the phenotype directory
if detect_genome_build(phenotype) == "19":
    os.system("cp genotypes.bim19 " + phenotype + os.sep + phenotype + ".bim")
     
if detect_genome_build(phenotype) == "38":
    os.system("cp genotypes.bim38 " + phenotype + os.sep + phenotype + ".bim")
     
#exit(0)
import os
import pandas as pd

# Copy the genotype .bed and .fam files to the phenotype directory
os.system("cp genotypes.bed " + phenotype + os.sep + phenotype + ".bed")
os.system("cp genotypes.fam " + phenotype + os.sep + phenotype + ".fam")

def create_directory(directory):
    """Function to create a directory if it doesn't exist."""
    if not os.path.exists(directory):  # Check if the directory doesn't exist
        os.makedirs(directory)  # Create the directory if it doesn't exist
    return directory  # Return the created or existing directory

"""
# Function to extract and save all phenotype columns to a text file (commented out)
def getmeallphenotypes():
    data = pd.read_csv("phenotype_file.txt", sep="\s+")
    print(data.columns.to_list()[19:])
    my_list = data.columns.to_list()[19:]
    file_path = 'All_Phenotypes.txt'
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(f"{item}\n")
"""

# Load the phenotype file
data = pd.read_csv("phenotype_file.txt", sep="\s+")
print(data.columns.to_list()[4:19])
my_list = data.columns.to_list()[4:19]
for m in my_list:
    print(m)

# Define the path to save the list of phenotypes
file_path = 'All_Phenotypes.txt'

# Load the actual phenotype data
actualphenotype = pd.read_csv("phenotype_file.txt", sep="\t")
# Select the covariate columns (first 152 columns)
covariatecolumns = actualphenotype.columns.to_list()[0:152]
cov = actualphenotype[covariatecolumns]
del cov['fasting_time']  # Remove the 'fasting_time' column

# Remove the actual phenotype column
del cov[sys.argv[1]]

# Rename columns to match PLINK requirements
cov.rename(columns={'ID': 'IID'}, inplace=True)
cov.rename(columns={'sex': 'Sex'}, inplace=True)

# Update the Sex column for PLINK
cov['Sex'] = cov['Sex'].replace({1: 2, 0: 1})

# Create the directory for the phenotype if it doesn't exist
create_directory(phenotype)

# Add 'FID' column with the same values as 'IID'
cov.insert(loc=1, column='FID', value=cov["IID"].values)
# Save the covariate file
cov.to_csv(phenotype + os.sep + phenotype + ".cov", sep="\t", index=False)

# Load the actual phenotype data
actualphenotype = pd.read_csv("phenotype_file.txt", sep="\t")
print(actualphenotype.head())

# Create a temporary dataframe for the phenotype data
tempframe = pd.DataFrame()
tempframe["IID"] = actualphenotype["ID"].values
tempframe["FID"] = actualphenotype["ID"].values
tempframe["Height"] = actualphenotype[phenotype].values
column_values = tempframe["Height"].unique()

# If the phenotype is binary, replace values 0 and 1 with 1 and 2 as required by PLINK
if set(column_values) == {0, 1}:
    # Change the cases and controls
    print("The column contains only 0 and 1.")
    tempframe.replace({0: 1, 1: 2}, inplace=True)
else:
    print("The column does not contain only 0 and 1.")

# Save the temporary phenotype file
tempframe.to_csv(phenotype + os.sep + phenotype + ".heightOLD", sep="\t", index=False)

# Reindex the temporary frame to match the order in the .fam file
tempframe.set_index('IID', inplace=True)
custom_order = pd.read_csv(phenotype + os.sep + phenotype + ".fam", sep="\s+", header=None)[0].values
df_reindexed = tempframe.reindex(custom_order)
df_reindexed = df_reindexed.reset_index()
df_reindexed.to_csv(phenotype + os.sep + phenotype + ".height", sep="\t", index=False)

# Load and process the covariate file to match the order in the .fam file
cov = pd.read_csv(phenotype + os.sep + phenotype + ".cov", sep="\s+")
cov.set_index('IID', inplace=True)
cov.to_csv(phenotype + os.sep + phenotype + ".covOLD", sep="\t", index=False)
custom_order = pd.read_csv(phenotype + os.sep + phenotype + ".fam", sep="\s+", header=None)[0].values
df_reindexed = cov.reindex(custom_order)
df_reindexed = df_reindexed.reset_index()
df_reindexed.to_csv(phenotype + os.sep + phenotype + ".cov", sep="\t", index=False)






