#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import os
import random
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import itertools

np.random.seed(42)

def load_data(disorder_names, data_dir="."):
    """
    Load crumblr data for each disorder from the specified directory.

    Parameters:
    disorder_names (list): List of disorder names.
    data_dir (str): Directory where the data files are located. Default is current directory.

    Returns:
    dict: Dictionary of DataFrames, one for each disorder.
    """
    dataframes = {}
    for disorder in disorder_names:
        filename = os.path.join(data_dir, f"{disorder}_crumblr_meta.tsv")
        if os.path.exists(filename):
            dataframes[disorder] = pd.read_csv(filename, index_col=[0], sep='\t')
        else:
            print(f"File not found: {filename}")
    return dataframes

disorder_names = ["AD", "DLBD", "SCZ", "BD", "Vasc", "Tau", "PD", "FTD"]
data_dir = "."  # Use current directory or specify the path to your data directory

dataframes = load_data(disorder_names, data_dir)

categories = dataframes['AD']['assay']
random_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(categories))]
pal = dict(zip(categories, random_colors))

def scatter_crumblr(crumblr_x, crumblr_y, name_x='logFC_x', name_y='logFC_y'):
    """
    Generate a scatter plot between two crumblr datasets and calculate the Spearman correlation coefficient.

    Parameters:
    crumblr_x (pd.DataFrame): DataFrame containing data for the x-axis.
    crumblr_y (pd.DataFrame): DataFrame containing data for the y-axis.
    name_x (str): Name for the x-axis data. Default is 'logFC_x'.
    name_y (str): Name for the y-axis data. Default is 'logFC_y'.

    Returns:
    float: Spearman correlation coefficient between the two datasets.
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

def calculate_scatter_results(disorder_names, dataframes):
    """
    Calculate scatter plot results for all combinations of disorders.

    Parameters:
    disorder_names (list): List of disorder names.
    dataframes (dict): Dictionary of DataFrames, one for each disorder.

    Returns:
    dict: Dictionary with keys as disorder combinations and values as correlation coefficients.
    """
    scatter_results = {}
    for disorder1, disorder2 in itertools.combinations(disorder_names, 2):
        result = scatter_crumblr(crumblr_x=dataframes[disorder1],
                                 crumblr_y=dataframes[disorder2],
                                 name_x=f'logFC_{disorder1}',
                                 name_y=f'logFC_{disorder2}')
        scatter_results[f'{disorder1}_{disorder2}'] = result
    return scatter_results

scatter_results = calculate_scatter_results(disorder_names, dataframes)

def create_correlation_matrix(disorder_names, scatter_results):
    """
    Create a correlation matrix from scatter plot results.

    Parameters:
    disorder_names (list): List of disorder names.
    scatter_results (dict): Dictionary with keys as disorder combinations and values as correlation coefficients.

    Returns:
    pd.DataFrame: Correlation matrix.
    """
    df = pd.DataFrame(index=disorder_names, columns=disorder_names).fillna(1.0)
    for key, result in scatter_results.items():
        disorder1, disorder2 = key.split('_')
        df.loc[disorder1, disorder2] = result
        df.loc[disorder2, disorder1] = result
    return df

correlation_matrix = create_correlation_matrix(disorder_names, scatter_results)

def hierarchical_clustering(data):
    """
    Perform hierarchical clustering on the given data.

    Parameters:
    data (dict): Dictionary of DataFrames, one for each trait.

    Returns:
    tuple: (ordered_data, traits) where ordered_data is the reordered dictionary of data and traits is the list of trait names.
    """
    corr_matrix = pd.DataFrame({key: df['estimate'] for key, df in data.items()}).corr(method='spearman')
    link = linkage(squareform(1 - corr_matrix), method='average')
    ordered_indices = leaves_list(link)
    ordered_traits = corr_matrix.columns[ordered_indices]
    ordered_data = {trait: data[trait] for trait in ordered_traits}
    return ordered_data, list(ordered_traits)

data = {key: df[df.index.str.startswith(('EN_', 'IN_'))] for key, df in dataframes.items()}
ordered_data, traits = hierarchical_clustering(data)

def plot_heatmap(ordered_data, traits):
    """
    Plot a heatmap of the given ordered data.

    Parameters:
    ordered_data (dict): Dictionary of ordered DataFrames.
    traits (list): List of trait names.
    """
    n = len(traits)
    plt.figure(figsize=(10, 10))
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(10, 8), sharex='col', sharey='row')

    for i, trait1 in enumerate(traits):
        for j, trait2 in enumerate(traits):
            ax = axes[i, j]
            if j > i:  # Only fill upper triangle
                merged_df = pd.merge(ordered_data[trait1], ordered_data[trait2], left_index=True, right_index=True, suffixes=('_' + trait1, '_' + trait2))
                sns.kdeplot(data=merged_df, x='estimate_' + trait1, y='estimate_' + trait2, ax=ax, fill=True)
            elif i == j:  # Diagonal: place trait names
                ax.text(0.5, 0.5, trait1, transform=ax.transAxes, fontsize=12, ha='center', va='center')
                ax.axis('off')

    # Add column and row labels
    for i, trait in enumerate(traits):
        axes[i, -1].annotate(trait, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10, ha='left', va='center', rotation=0)
        axes[-1, i].annotate(trait, xy=(0.5, -0.35), xycoords='axes fraction', fontsize=10, ha='center', va='top', rotation=90)

    # Hide the lower triangle
    for i in range(n):
        for j in range(i):
            axes[i, j].axis('off')

    # Adjust layout
    plt.tight_layout()

plot_heatmap(ordered_data, traits)

