# 5_cross-disorder_expression: Cross-Disorder Transcriptomic Correlation

This repository contains code to generate a cross-disorder correlation heatmap for brain cell types (subclass) across multiple neurodegenerative and neuropsychiatric diseases.

## Overview

This script performs the following steps:

- Loads and filters meta-analysis results across disorders.
- Removes genes with shared effects across conditions (using mashr).
- Computes Spearman correlation of gene effects across disorders within each cell subtype.
- Annotates cell types and disease pairs.
- Generates a complex heatmap of cross-disorder transcriptomic correlations.

## Requirements

### R Packages

Ensure the following R libraries are installed:

```r
# Dreamlet ecosystem
dreamlet, crumblr, zenith

# Data IO
SingleCellExperiment, zellkonverter, arrow, tidyr, tidyverse

# Plotting and visualization
ggplot2, dplyr, cowplot, ggtree, aplot, purrr, circlize, ComplexHeatmap, RColorBrewer

# Meta-analysis
muscat, metafor, broom

# Utilities
ape, dendextend
```

## Input Files
- res_meta.parquet: Meta-analysis results
- topTable_combined.parquet: Differential expression data from one brain bank
- PsychAD_rowData_34890.csv: Gene annotations
- PsychAD_mashr_shared.csv: Shared genes across cell types/disorders
- PsychAD_color_palette.csv: Cell type and class color annotations
- tree_subclass_um.nwk: Dendrogram of brain cell subclasses

## Key Steps

### 1. Filter Protein-Coding Genes
```
pcg = read.csv('/path/to/PsychAD_rowData_34890.csv')
pcg = pcg[pcg$gene_type == 'protein_coding', 'gene_name']
```

### 2. Load and Filter Meta-Analysis by Disorder
For each disorder (AD, SCZ, DLBD, etc.), filter by FDR < 0.05, retain only protein-coding genes, and compute FDR-adjusted p-values.

### 3. Remove Shared Genes (from mashr)
```
excl_genes = read.csv("/path/to/PsychAD_mashr_shared.csv")
# Exclude genes with shared effects across subclasses
```

### 4. Merge Datasets and Compute Correlation
- Compute Spearman correlations of logFC estimates across shared genes for each disorder pair.
- Also calculate the number of overlapping significant genes (FDR < 0.05) for each disorder pair.

### 5. Tree and Assay Ordering
- Use tree_subclass_um.nwk to extract the cell type (assay) order.
- Map subclasses to higher-level classes using the color palette CSV.

### 6. Build ComplexHeatmap
- Heatmap of Spearman correlations per assay × disorder pair
- Annotations:
	•	Number of shared FDR-significant genes (rows and columns)
	•	Disease-pair group (NDD, NPD, NDD-NPD)
	•	Cell class for each subtype

### 7. Save Plot
```
pdf(file="PsychAD_Figure5B_crossDis_dreamlet_heatmap.pdf", width=9.5, height=7)
draw(ht_list)
dev.off()
```
