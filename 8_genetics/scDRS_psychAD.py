# Required Libraries
# ------------------
# Scientific analysis
import scdrs
import pegasus as pg
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import seaborn as sns

# File I/O and utilities
import os
import csv
import re
import glob
from scipy.stats import zscore
import pynndescent

# Synapse API
from synapseclient import Synapse

# Initialize Synapse Client
# -------------------------
syn = Synapse()
syn.login()  # Make sure your Synapse credentials are configured

# Configuration
# -------------
DATA_DIR = "scDRS"
GENESET_PATH = "./custom_geneset.gs"
ENTITY_ID = "" # syn number

# Ensure directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download and Load Data
# ----------------------
print("Downloading data from Synapse...")
syn_entity = syn.get(entity=ENTITY_ID, downloadLocation=DATA_DIR)
adata = sc.read_h5ad(syn_entity.path)

# Preprocess AnnData Object
# -------------------------
print("Preprocessing AnnData object...")
scdrs.preprocess(adata, n_mean_bin=20, n_var_bin=20, copy=False)

# Load Custom Gene Set
# --------------------
print("Loading custom gene sets...")
dict_gs = scdrs.util.load_gs(
    GENESET_PATH, src_species="human", dst_species="human", to_intersect=adata.var_names
)
dict_you_want = dict_gs  # Modify if specific studies are required

# Assign Connectivity Slot
# -------------------------
adata.obsp['connectivities'] = adata.obsp.get('W_pca_regressed_harmony', None)

# Functions
# ---------
def calculate_scdrs_scores(adata, gene_sets, ctrl_match_key="mean_var", n_ctrl=200):
    """Calculate scDRS scores for given gene sets."""
    scores = {}
    for trait, (gene_list, gene_weights) in gene_sets.items():
        print(f"Calculating scDRS scores for: {trait}")
        scores[trait] = scdrs.score_cell(
            data=adata,
            gene_list=gene_list,
            gene_weight=gene_weights,
            ctrl_match_key=ctrl_match_key,
            n_ctrl=n_ctrl,
            weight_opt="vs",
            return_ctrl_raw_score=False,
            return_ctrl_norm_score=True,
            verbose=False
        )
    return scores

def perform_downstream_analysis(adata, scores, group_col, output_prefix):
    """Perform downstream analysis and save results."""
    for trait, df_full_score in scores.items():
        print(f"Performing downstream analysis for: {trait} ({group_col})")
        df_stats = scdrs.method.downstream_group_analysis(
            adata=adata, df_full_score=df_full_score, group_cols=[group_col]
        )[group_col]
        
        # Save results
        df_full_score.to_csv(
            f"{output_prefix}_{group_col}_{trait}.csv", sep="\t", quoting=csv.QUOTE_NONE
        )
        df_stats.to_csv(
            f"{output_prefix}_{group_col}_{trait}_summary.csv", sep="\t", quoting=csv.QUOTE_NONE
        )

# Main Workflow
# -------------
# Calculate scDRS Scores
print("Calculating scDRS scores...")
dict_df_score = calculate_scdrs_scores(adata, dict_you_want)

# Perform Downstream Analysis
print("Performing downstream analysis...")
perform_downstream_analysis(adata, dict_df_score, "class", "./psychAD_class")
perform_downstream_analysis(adata, dict_df_score, "subclass", "./psychAD_subclass")

print("Analysis complete.")

