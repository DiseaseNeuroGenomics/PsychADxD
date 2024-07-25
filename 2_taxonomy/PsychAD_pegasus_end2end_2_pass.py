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

parser.add_argument('--n_pcs', help='Number of pcs', required=False, type=int, default=50)
parser.add_argument('--batch', help='Batch for harmony', required=False, default='poolID')
parser.add_argument('--res', help='Cluster resolution', required=False, type=float, default=3.0)
parser.add_argument('--n_neighbors', help='Number of nearest neighbors considered for UMAP', required=False, type=int, default=15)
parser.add_argument('--tsne', default=False, action='store_true')

args = parser.parse_args()

################################################################################

class_label='leiden_labels_res'+str(int(args.res*10))

### pass X #####################################################################

### load data
data = pg.read_input(args.input)
print(data)

### pca/harmony/umap/tsne
pg.pca(data, n_components=args.n_pcs)
pg.elbowplot(data)
npc = min(data.uns["pca_ncomps"], args.n_pcs)
print('Using %i components for PCA' % npc)
pg.regress_out(data, attrs=['n_counts','percent_mito','cycle_diff'])
pg.run_harmony(data, batch=args.batch, rep='pca_regressed', max_iter_harmony=20, n_comps=npc)
pg.neighbors(data, rep='pca_regressed_harmony', use_cache=False, dist='cosine', K=100, n_comps=npc)
pg.leiden(data, rep='pca_regressed_harmony', resolution=args.res, class_label=class_label)
pg.umap(data, rep='pca_regressed_harmony', n_neighbors=args.n_neighbors, rep_ncomps=npc)
if args.tsne:
    pg.tsne(data, rep='pca_regressed_harmony', rep_ncomps=npc)

# figure
sc.pl.umap(data.to_anndata(), color=[class_label], legend_loc='on data', frameon=False, legend_fontsize=5, legend_fontoutline=1, title=[class_label], size=1, wspace=0, ncols=1, save=args.output+'.png')

### save
pge.save(data, args.output)

################################################################################
