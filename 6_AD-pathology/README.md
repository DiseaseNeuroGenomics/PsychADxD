# 6_AD-pathology: Differential Meta-analysis and Visualization of AD Risk Genes

This repository contains an R-based analysis pipeline for identifying and visualizing consistently dysregulated genes across cell types in Alzheimer's disease (AD), using meta-analysis of single-cell transcriptomic data. The output corresponds to Figure 6D of the manuscript.

## Overview

The workflow includes:

- Loading and preprocessing meta-analysis results
- Filtering protein-coding genes
- Identifying genes with consistent effects across phenotypes and assays
- Visualizing selected genes and their association strength via a heatmap-style dot plot overlaid on a cell type dendrogram

## Setup

### Required R Packages

```r
# Core analysis packages
library(dreamlet)
library(crumblr)
library(zenith)

# Data I/O
library(SingleCellExperiment)
library(zellkonverter)
library(tidyr)
library(tidyverse)
library(arrow)

# Plotting
library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(cowplot)
library(ggtree)
library(aplot)

# Meta-analysis
library(muscat)
library(metafor)
library(broom)

# Tree plotting
library(ape)
library(dendextend)
```

## Input Files
- res_meta.parquet: Meta-analysis results from multiple phenotypes
- PsychAD_rowData_34890.csv: Row metadata containing gene types
- tree_subclass_um.nwk: Newick-formatted cell type tree

## Key Steps

### 1. Load and Merge Meta-analysis Results

Filter for each phenotype of interest (AD, CERAD, Braak, Dementia) and compute FDR values:

```
meta = read_parquet("/path/to/res_meta.parquet")

# Filter by phenotype and compute FDR
meta.dx = ...
meta.ce = ...
meta.br = ...
meta.de = ...

merged = rbind(meta.dx, meta.ce, meta.br, meta.de)
```

### 2. Filter for Protein-coding Genes
```
pcg = read.csv("/path/to/PsychAD_rowData_34890.csv")
merged_pcg = merged %>% filter(ID %in% pcg$gene_name[pcg$gene_type == "protein_coding"])
```

### 3. Load Tree and Visualize Cell Type Relationships
```
tree = read.tree(file="/path/to/tree_subclass_um.nwk")
fig.tree = plot_tree_simple(as.phylo(tree), xmax.scale=1.5)
```

### 4. Identify Genes with Consistent Signals
```
# Criteria
FDR_threshold = 0.01
logFC_threshold = 0.5

# Subset significant genes found in ≥2 studies and ≥4 cell types
df_subset <- merged_pcg %>% filter(FDR < FDR_threshold, abs(estimate) >= logFC_threshold, n.studies >= 2)
g <- names(table(df_subset$ID)[table(df_subset$ID) >= 4])
```

### 5. Prepare Data for Plotting
```
df_top <- rbind(...) %>% 
  filter(assay %in% assay_order, ID %in% g) %>% 
  mutate(...) %>%
  arrange(...)
```

### 6. Generate the Dot Plot Heatmap
```
fig.hm <- df_top %>%
  ggplot(...) +
  geom_point(...) +
  geom_text(...) +
  scale_color_gradient2(...) +
  scale_size_area(...) +
  scale_x_continuous(...) +
  scale_y_continuous(...) +
  labs(...) +
  theme(...)
```

### 7. Save the Plot
```
ggsave(paste0(prefix, "_AD.svg"), width = 26, height = 12)
```
