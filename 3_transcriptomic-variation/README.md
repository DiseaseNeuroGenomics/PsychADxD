# 3_transcriptomic-variation: Analysis of Sources of Transcriptomic Variation

This repository contains scripts used to perform variance partition analysis with stacked assays, supporting the generation of Figure 3 in our study.

## Overview

To quantify transcriptomic variation across different cell types, we applied the `stackAssays` approach on pseudobulk expression data followed by variance partition analysis. This enabled us to assess the contribution of various factors to gene expression variability across conditions and cell types.

## Methods

- **Stacked Assays**: Multiple pseudobulk assays were combined using the [`stackAssays`](https://diseaseneurogenomics.github.io/dreamlet/reference/stackAssays.html) function from the [`dreamlet`](https://diseaseneurogenomics.github.io/dreamlet/) R package.
- **Variance Partitioning**: The combined data were analyzed to estimate the proportion of expression variance attributable to each factor of interest.

## Usage

Clone this repository and follow the scripts provided to reproduce the variance partition analysis.

```bash
git clone https://github.com/<your-username>/3_transcriptomic-variation.git
cd 3_transcriptomic-variation
