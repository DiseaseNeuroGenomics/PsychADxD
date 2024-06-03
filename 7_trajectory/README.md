# 7_trajectory: analysis on disease trajectory  

### Step 1 - Create the dataset used for model training and analysis  
from create_dataset import create_dataset  
create_dataset(source_paths, target_path)  
"""source_paths is a list of h5ad files (e.g. ["dataset1.h5ad", "dataset2.h5ad"])  
target_path is the directory where the gene data (.dat file) and metadata (.pkl file) will be saved"""  

### Step 2 - Create train test splits
from create_train_test_splits import create_splits
create_splits(metadata_fn, save_fn)
"""metadata_fn is the name of the saved metadata file created above"""

