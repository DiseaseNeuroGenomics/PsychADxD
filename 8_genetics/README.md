# Genetic Correlation and scDRS Analysis Pipeline

This repository contains code and instructions for performing genetic correlation analysis using LDSC and cell-type-specific enrichment analysis using scDRS.

## Overview

- **LDSC (Linkage Disequilibrium Score Regression)** is used to compute genetic correlations between neurodegenerative and psychiatric disorders using GWAS summary statistics.
- **scDRS (single-cell Disease Relevance Score)** is used to assess cell-type enrichment of disease-associated gene sets in single-cell transcriptomic data.

## 1. LDSC: Genetic Correlation Analysis

### Installation

Follow instructions to install LDSC:  
🔗 [https://github.com/bulik/ldsc](https://github.com/bulik/ldsc)

### Data Preparation

Download LD scores and SNP lists:

```bash
wget https://data.broadinstitute.org/alkesgroup/LDSCORE/eur_w_ld_chr.tar.bz2
wget https://data.broadinstitute.org/alkesgroup/LDSCORE/w_hm3.snplist.bz2
tar -jxvf eur_w_ld_chr.tar.bz2
bunzip2 w_hm3.snplist.bz2
```
## 2. scDRS: Cell-type Enrichment Analysis

### Required Libraries

Install the following packages:
- scdrs, pegasus, scanpy, anndata
- numpy, pandas, seaborn, matplotlib
- synapseclient, scipy, pynndescent
