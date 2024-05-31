#!/usr/bin/env python
# coding: utf-8

import pegasus as pg
import scanpy as sc
import scanpy.external as sce
import anndata as ad
from anndata.tests.helpers import assert_equal
from anndata._core.sparse_dataset import SparseDataset
from anndata.experimental import read_elem, write_elem

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# data
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
import h5py

# sys
import gc
from pathlib import Path
import sys
import os
import random
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist, squareform

# Set random seed for reproducibility
np.random.seed(42)

# Disorder names
disorder_names = ["AD", "DLBD", "SCZ", "BD", "Vasc", "Tau", "PD", "FTD"]

# Load dataframes
dataframes = {}
for disorder in disorder_names:
    filename = f"{disorder}_crumblr_meta.tsv"
    if os.path.exists(filename):
        dataframes[disorder] = pd.read_csv(filename, index_col=[0], sep='\t')
    else:
        print(f"File not found: {filename}")

# Generate random colors for categories
categories = dataframes['AD']['assay']
random_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(categories))]
pal = dict(zip(categories, random_colors))

def scatter_crumblr(crumblr_x, crumblr_y, name_x='logFC_x', name_y='logFC_y'):
    """
    Generates a scatter plot of two crumblr datasets and calculates their Spearman correlation coefficient.
    
    Parameters:
    crumblr_x (DataFrame): First crumblr dataset.
    crumblr_y (DataFrame): Second crumblr dataset.
    name_x (str): Column name for x-axis values.
    name_y (str): Column name for y-axis values.
    
    Returns:
    float: Spearman correlation coefficient between the datasets.
    """
    crumblr_x.index = crumblr_x.assay
    crumblr_x = crumblr_x[['estimate', 'std.error']]
    crumblr_x.columns = [name_x, 'SE_x']

    crumblr_y.index = crumblr_y.assay
    crumblr_y = crumblr_y[['estimate', 'std.error']]
    crumblr_y.columns = [name_y, 'SE_y']
    
    merged = pd.merge(crumblr_x, crumblr_y, left_index=True, right_index=True)
    merged['color'] = [pal[x] for x in merged.index]
    merged = merged.reset_index()
    
    corcoef, _ = spearmanr(merged[name_x], merged[name_y])
    corcoef = round(corcoef, 2)
    
    model = LinearRegression()
    model.fit(merged[name_x].values.reshape(-1, 1), merged[name_y].values.reshape(-1, 1))
    X_new = np.linspace(min(merged[name_x]), max(merged[name_x]), 100).reshape(-1, 1)
    y_pred = model.predict(X_new)
    
    return corcoef

def create_correlation_matrix(data, neuron_filter=None):
    """
    Creates a correlation matrix for the provided data.

    Parameters:
    data (dict): Dictionary of dataframes for each disorder.
    neuron_filter (tuple): Tuple of strings to filter neuron types, e.g., ('EN_', 'IN_').

    Returns:
    DataFrame: Ordered correlation matrix.
    """
    if neuron_filter:
        data = {key: df[df.index.str.startswith(neuron_filter)] for key, df in data.items()}
    else:
        data = {key: df[~df.index.str.startswith(('EN_', 'IN_'))] for key, df in data.items()}

    disorder_names = list(data.keys())
    scatter_results = {}

    for disorder1, disorder2 in itertools.combinations(disorder_names, 2):
        result = scatter_crumblr(crumblr_x=data[disorder1],
                                 crumblr_y=data[disorder2],
                                 name_x=f'logFC_{disorder1}',
                                 name_y=f'logFC_{disorder2}')
        scatter_results[f'{disorder1}_{disorder2}'] = result

    df = pd.DataFrame(index=disorder_names, columns=disorder_names).fillna(1.0)

    for key, result in scatter_results.items():
        disorder1, disorder2 = key.split('_')
        df.loc[disorder1, disorder2] = result
        df.loc[disorder2, disorder1] = result

    row_linkage = linkage(pdist(df, metric='euclidean'), method='average')
    ordered_index = leaves_list(row_linkage)
    df_ordered = df.iloc[ordered_index, :].iloc[:, ordered_index]

    return df_ordered

def plot_heatmap(df_ordered, title, mask=True):
    """
    Plots a heatmap of the ordered correlation matrix.

    Parameters:
    df_ordered (DataFrame): Ordered correlation matrix.
    title (str): Title of the heatmap.
    mask (bool): Whether to mask the upper triangle of the heatmap.
    """
    mask = np.triu(np.ones_like(df_ordered, dtype=bool)) if mask else None

    plt.figure(figsize=(10, 10))
    sns.clustermap(df_ordered, mask=mask, cmap='PiYG',
                   annot=True,
                   row_cluster=False,  # Disable row clustering
                   col_cluster=False,  # Disable column clustering
                   vmin=-1, 
                   vmax=1, 
                   cbar_kws={'label': title},
                   linewidths=3.5,
                   figsize=(10, 8))
    plt.show()

# Create and plot heatmaps for neurons only and non-neurons only
df_neurons = create_correlation_matrix(dataframes, neuron_filter=('EN_', 'IN_'))
plot_heatmap(df_neurons, 'Neuronal Correlation Coefficient')

df_non_neurons = create_correlation_matrix(dataframes)
plot_heatmap(df_non_neurons, 'Non-Neuronal Correlation Coefficient', mask=False)

