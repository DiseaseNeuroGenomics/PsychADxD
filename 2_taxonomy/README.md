# 2_taxonomy: preprocessing and hierarchical cellular taxonomy of the snRNA-seq data

### Requirements  
A collection of custom pegasus functions was adopted from https://github.com/leelaboratory/leetools/blob/main/pegasus/pge.py

### Step 1 - Preprocessing
Performs QCs at gene-level, cell-level, and library-level to filter out low-quality cells using pegasus. Performs shifted log-normalization and signature score required in subsequent processing.

Example Usage
```
python PsychAD_pegasus_end2end_1_qc.py --input <Input h5ad file> --output <Output h5ad file>
```

### Step 2 - Highly variable feature
Highly variable feature selection. scanpy's `highly_variable_genes` function was extended to select from protein-coding genes in autosome.

Example Usage
```
PsychAD_pegasus_end2end_2_hvf.py --input <Input h5ad file> --output <Output h5ad file> --flavor cell_ranger --n_top_genes 6000 --batch_key poolID
```

### Step 3 - First pass clustering
First pass clustering analysis using pegasus. This step is required to prepare for the doublet removal step.

Example Usage
```
python PsychAD_pegasus_end2end_3_pass.py --input <Input h5ad file> --output <Output h5ad file> --n_pcs 50 --batch poolID --res 1.2 --n_neighbors 15 --tsne False
```

### Step 4 - Doublet removal
Removal of doublets using scrublet.

Example Usage
```
python PsychAD_pegasus_end2end_4_scrublet.py --input <Input h5ad file> --output <Output h5ad file>
```

### Step 5 - Iterative clustering
Iterative clustering to define the hierarchical structure of clustering.

Example Usage
```
python PsychAD_pegasus_end2end_5_iclust.py --input <Input h5ad file> --output <Output h5ad file>
```
