# 7_trajectory: analysis on disease trajectory  

Note: we borrowed heavily from https://github.com/tabdelaal/scVI/blob/master/scvi/models/  

### Requirements  
PyTorch and PyTorch Lightning (https://lightning.ai/docs/pytorch/stable/).  
Conda environment used for model training given in vae.yaml

### Step 1 - Create the dataset used for model training and analysis 
```
from create_dataset import create_dataset  
create_dataset(source_paths, target_path)  
```
source_paths is a list of h5ad files (e.g. ["dataset1.h5ad", "dataset2.h5ad"])  
target_path is the directory where the gene data (.dat file) and metadata (.pkl file) will be saved 

### Step 2 - Create train test splits
```
from create_train_test_splits import create_splits  
create_splits(metadata_fn, save_fn)  
```
metadata_fn is the name of the saved metadata file created above

### Step 3 - Modify config.py
Add path names for gene data (.dat file), metadata (.pkl file) and train/test splits (.pkl) to config.py.  
Modify cell_restrictions to change which cell class to train.

### Step 4 - Train model and run inference  
python train.py  
Model will be trained on all train/test splits. Model inference (e.g. predicted Braak and dementia scores, cell index) will be saved in lightning_logs/version_X/test_results_epX.pkl after each epoch.  
  
### Step 5 - Create AnnData structure with model predictions
In the example below, we will create the AnnData structure with the donor-averaged model predictions and gene expression values. The model predictions are usually saved in the lightning_logs directory, where the results of each of the ten train/test splits are saved in the subdirectory version_XX.   

```
import process_data
mr = process_data.ModelResults(    
    data_fn=[Gene data filename, .dat],  
    meta_fn = [Metadata data filename, .pkl],  
    include_analysis_only=True,  
    normalize_gene_counts=True,  
    log_gene_counts=True,  
    add_gene_scores=True,  
)  
```
For this example, we assume that the model predictions have been saved in the folders version_0 through version_9.
We will average model predictions created after the 4th and 5th epoch (keep in mind 0-indexing in Python).

```
data_path = "[Path where logs are saved]/lightning_logs/"  
start_epoch = 3   
fns = []  
for m in range(start_epoch, start_epoch + 2):  
    fns0 = []  
    for n in range(0, 10): # assuming results have been saved in the folders version_0 through version_9  
        fns0.append(base_path + f"version_{n}/test_results_ep{m}.pkl")  
    fns.append(fns0)  

adata = mr.create_data(fns, model_average=True)  
```






