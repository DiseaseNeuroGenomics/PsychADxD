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

args = parser.parse_args()

################################################################################

### pass 0 #####################################################################

### load data
data = pg.read_input(args.input, genome='GRCh38', modality='rna')
data = pge.clean_unused_categories(data)
print(data)

### signature scores
pg.calc_signature_score(data, 'cell_cycle_human') ## 'cycle_diff', 'cycling', 'G1/S', 'G2/M' ## cell cycle gene score based on [Tirosh et al. 2015 | https://science.sciencemag.org/content/352/6282/189]
pg.calc_signature_score(data, 'gender_human') # female_score, male_score
pg.calc_signature_score(data, 'mitochondrial_genes_human') # 'mito_genes' contains 13 mitocondrial genes from chrM and 'mito_ribo' contains mitocondrial ribosomal genes that are not from chrM
pg.calc_signature_score(data, 'ribosomal_genes_human') # ribo_genes
pg.calc_signature_score(data, 'apoptosis_human') # apoptosis

pge.save(data, args.output)

################################################################################
