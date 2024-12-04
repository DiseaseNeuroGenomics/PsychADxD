# PsychADxD
[PsychAD Cross-Disorder Capstone] Single-cell atlas of transcriptomic vulnerability across multiple neuropsychiatric and neurodegenerative diseases

# Contents
```
PsychADxD
├──1_metadata: metadata for the snRNA-seq data
├──2_taxonomy: preprocessing and hierarchical cellular taxonomy of the snRNA-seq data
├──3_transcriptomic-variation: analysis on variation of transcriptome
├──4_cross-disorder_composition: analysis on compositional variation across multiple disorders
├──5_cross-disorder_expression: analysis on gene expression variation across multiple disorders
├──6_AD-pathology: analysis on compositional & gene expression variation along AD pathology
└──7_trajectory: analysis on disease trajectory
```

# System requirements
Python codes require Python >= v3.8. R codes require BioC v3.17 for R >= v4.3.0.

The following version of dependencies were used when testing for compatibility.
```
anndata v0.9.1
aplot v0.2.0
Biobase v2.60.0
BiocGenerics v0.46.0
BiocParallel v1.34.2
cowplot v1.1.1
crumblr v0.99.8
dplyr v1.1.2
dreamlet v0.99.25
GenomeInfoDb v1.36.1
GenomicRanges v1.52.0
ggplot2 v3.4.3
ggtree v3.8.2
IRanges v2.34.1
limma v3.56.2
louvain v0.7.1
MatrixGenerics v1.12.3
matrixStats v1.0.0
numpy v1.24.4
pandas v1.5.0
pynndescent v0.5.6
python-igraph v0.9.10
RColorBrewer v1.1-3
S4Vectors v0.38.1
scanpy v1.9.3
scikit-learn v1.3.0
scipyv1.10.1
SingleCellExperiment v1.22.0
statsmodels v0.14.0
SummarizedExperiment v1.30.2
tidyr v1.3.0
umap v0.5.3
variancePartition v1.31.15
zellkonverter v1.10.1
zenith v1.2.0
```
- For more information about the installation, demo workflow, and use cases of Dreamlet, please visit https://diseaseneurogenomics.github.io/dreamlet/.
- For more information about the installation, demo workflow, and use cases of Crumblr, please visit https://diseaseneurogenomics.github.io/crumblr/.

# Citation
Donghoon Lee, Mikaela Koutrouli, Nicolas Y. Masse, Gabriel E. Hoffman, Seon Kinrot, Xinyi Wang, Prashant N.M., Milos Pjanic, Tereza Clarence, Fotios Tsetsos, Deepika Mathur, David Burstein, Karen Therrien, Aram Hong, Clara Casey, Zhiping Shao, Marcela Alvia, Stathis Argyriou, Jennifer Monteiro Fortes, Pavel Katsel, Pavan K. Auluck, Lisa L. Barnes, Stefano Marenco, David A. Bennett, PsychAD Consortium, Lars Juhl Jensen, Kiran Girdhar, Georgios Voloudakis, Vahram Haroutunian, Jaroslav Bendl, John F. Fullard, Panos Roussos. Single-cell atlas of transcriptomic vulnerability across multiple neurodegenerative and neuropsychiatric diseases. medRxiv, 2024. doi: https://doi.org/10.1101/2024.10.31.24316513

# License
MIT License
