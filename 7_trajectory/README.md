# 7_trajectory: analysis on disease trajectory  

### Requirements  
PyTorch and PyTorch Lightning (https://lightning.ai/docs/pytorch/stable/).  
Conda environment used for model training given in env.yaml

### Step 1 - Create the dataset used for model training and analysis 
```
from create_dataset import create_dataset  
create_dataset(source_paths, target_path)  
```
source_paths is a list of h5ad files (e.g. ["dataset1.h5ad", "dataset2.h5ad"])  
target_path is the directory where the gene data (numpy memmap .dat file) and metadata (.pkl file) will be saved 

### Step 2 - Create train test splits
```
from create_train_test_splits import create_splits  
create_splits(metadata_fn, save_fn)  
```
metadata_fn is the name of the saved metadata file created above

### Step 3 - Modify config.py
Add path names for gene data (.dat file), metadata (.pkl file) and train/test splits (.pkl) to config.py.  
Modify cell_restrictions to change which cell class to train.
For the paper, we train separate models for each of the eight cell classes (EN, IN, Astro, Immune, Oligo, OPC, Mural and Endo).

### Step 4 - Train model and run inference  
```
python train.py  
```
Model will be trained on all train/test splits. Model inference (e.g. predicted Braak and dementia scores, cell index) will be saved in lightning_logs/version_X/test_results_epX.pkl after each epoch.  
  
### Step 5 - Create AnnData structure (h5ad) with model predictions
In process_model.ipynb, we walk through the step required to create the h5ad needed for downstream analysis 

### Step 6 - Analysis
Analysis code is found in analysis.py. Worksheet showing how the figures from the paper were generated are shown in generate_figures.ipynb.





