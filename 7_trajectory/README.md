# 7_trajectory: analysis on disease trajectory  

## Step 1 - Creating the dataset used for model training and analysis  
from create_dataset import create_dataset  
create_dataset(source_paths, target_path)  
source_paths is a list of h5ad files (e.g. ["dataset1.h5ad", "dataset2.h5ad"])  
target_path is the directory where the gene data (.dat file) and metadata (.pkl file) will be saved

