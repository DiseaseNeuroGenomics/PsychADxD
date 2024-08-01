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

parser.add_argument('--res', help='Cluster resolution', required=False, type=float, default=3.0)

args = parser.parse_args()

################################################################################

class_label='leiden_labels_res'+str(int(args.res*10))

### scrublet ###################################################################

### load data
data = pg.read_input(args.input)
print(data)

### find doublets
pg.infer_doublets(data, channel_attr = 'Channel', clust_attr = class_label, plot_hist=None)
pg.mark_doublets(data)
pge.save(data, args.output.replace('.h5ad','.mark_doublets.h5ad'))
pg.scatter(data, attrs='demux_type', basis='umap', dpi=150, return_fig=True).savefig('figures/'+args.output+'.png')
print(data.uns['pred_dbl_cluster'])

### doublet counts
dc = data.obs['demux_type'].value_counts().reset_index()
print(dc)
pct_dbl = dc.loc[dc['index']=='doublet','demux_type'] / np.sum(dc.loc[:,'demux_type']) * 100
print('Doublets: %.2f%%' % pct_dbl)

### filter doublets
pg.qc_metrics(data, select_singlets=True)
pg.filter_data(data)

### save
pge.save(data, args.output)

################################################################################
