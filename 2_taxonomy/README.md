# 2_taxonomy: preprocessing and hierarchical cellular taxonomy of the snRNA-seq data

A collection of custom pegasus functions was adopted from https://github.com/leelaboratory/leetools/blob/main/pegasus/pge.py

1. `PsychAD_pegasus_end2end_1_qc.py` Performs QCs at gene-level, cell-level, and library-level to filter out low-quality cells using pegasus. Performs shifted log-normalization and signature score required in subsequent processing.
2. `PsychAD_pegasus_end2end_2_hvf.py` Highly variable feature selection. scanpy's `highly_variable_genes` function was extended to select from protein-coding genes in autosome.
3. `PsychAD_pegasus_end2end_3_pass.py` First pass clustering analysis using pegasus.
4. `PsychAD_pegasus_end2end_4_scrublet.py` Removal of doublets using scrublet.
5. `PsychAD_pegasus_end2end_5_iclust.py` Iterative clustering.
