# sc
import pegasus as pg
import scanpy as sc
import anndata as ad
from anndata.tests.helpers import assert_equal
from anndata._core.sparse_dataset import SparseDataset
from anndata.experimental import read_elem, write_elem

# plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
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

# pge
import sys
sys.path.append("/path/to/leetools/pegasus/")
import pge
pge.info()

# etc
import argparse

################################################################################

parser = argparse.ArgumentParser(description='pegasus end2end wrapper script')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

parser.add_argument('--flavor', help='HVF selection method', required=False, default='cell_ranger')
parser.add_argument('--n_top_genes', help='n_top_genes', required=False, type=int, default=None)
parser.add_argument('--batch_key', help='batch_key', required=False, default='poolID')

args = parser.parse_args()

################################################################################

### hvf ########################################################################

hvg = pge.scanpy_hvf_h5ad(h5ad_file=args.input, flavor=args.flavor, batch_key=args.batch_key, n_top_genes=args.n_top_genes, min_mean=0.0125, max_mean=3, min_disp=0.5, robust_protein_coding=True)

### load data
data = pg.read_input(args.input)
print(data)

### set scanpy hvg as hvf
data.var.highly_variable_features = False
data.var.loc[data.var.index.isin(hvg),'highly_variable_features'] = True

### final value counts
print(data.var.highly_variable_features.value_counts())

### save
pge.save(data, args.output)

################################################################################
