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

def qc(input, output, prefix, args_mad_k = 3, args_pct_mito = 5, args_min_n_cells = 50):

    data = pg.read_input(input, file_type='h5ad', genome='GRCh38', modality='rna')
    
    ### create channel
    data.obs['Channel'] = data.obs.individualID
    
    ##################
    ### QC by gene ###
    ##################
    
    ### identify robust genes
    pg.identify_robust_genes(data, percent_cells=0.05)
    
    ### remove features that are not robust (expressed at least 0.05% of cells) from downstream analysis
    data._inplace_subset_var(data.var['robust'])
    
    ### add ribosomal genes
    data.var['ribo'] = [x.startswith("RP") for x in data.var.gene_name]
    
    ### add mitochondrial genes
    data.var['mito'] = [x.startswith("MT-") for x in data.var.gene_name]
    
    ### add protein_coding genes
    data.var['protein_coding'] = [x=='protein_coding' for x in data.var.gene_type]
    
    ### define mitocarta_genes
    mitocarta = pd.read_csv('annotation/Human.MitoCarta3.0.csv')
    data.var['mitocarta'] = [True if x in list(mitocarta.Symbol) else False for x in data.var.index ]
    
    ### define robust_protein_coding genes (exclude ribosomal (RPL,RPS), mitochondrial, or mitocarta genes
    data.var['robust_protein_coding'] = data.var['robust'] & data.var['protein_coding']
    data.var.loc[data.var.ribo, 'robust_protein_coding'] = False
    data.var.loc[data.var.mito, 'robust_protein_coding'] = False
    data.var.loc[data.var.mitocarta, 'robust_protein_coding'] = False
    
    ### define robust_protein_coding_autosome genes (exclude ribosomal (RPL,RPS), mitochondrial, or mitocarta genes
    data.var['robust_protein_coding_autosome'] = data.var['robust_protein_coding'] & data.var.gene_chrom.isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'])
    
    ####################
    ### QC by counts ###
    ####################
    
    pg.qc_metrics(data, mito_prefix='MT-')
    
    ### nUMI and nGene QCs
    n_counts_lower, n_counts_upper = pge.qc_boundary(data.obs.n_counts, k=args_mad_k)
    print('n_UMIs lower: %d upper: %d' % (n_counts_lower, n_counts_upper))
    
    n_genes_lower, n_genes_upper = pge.qc_boundary(data.obs.n_genes, k=args_mad_k)
    print('n_genes lower: %d upper: %d' % (n_genes_lower, n_genes_upper))
    
    # log(n_counts)
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(np.log10(data.obs.n_counts))
        plt.axvline(np.log10(n_counts_lower), color='red')
        plt.axvline(np.log10(n_counts_upper), color='red')
        plt.xlabel('log10(n_counts)', fontsize=12)
        plt.savefig(prefix+"_histplot_n_counts.png")
    
    # log(n_genes)
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(np.log10(data.obs.n_genes))
        plt.axvline(np.log10(n_genes_lower), color='red')
        plt.axvline(np.log10(n_genes_upper), color='red')
        plt.xlabel('log10(n_genes)', fontsize=12)
        plt.savefig(prefix+"_histplot_n_genes.png")
    
    # scatter
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.scatterplot(x=data.obs.n_genes, y=data.obs.n_counts, alpha=0.5, s=0.1)
        plt.axhline(n_counts_lower, color='red')
        plt.axhline(n_counts_upper, color='red')
        plt.axvline(n_genes_lower, color='red')
        plt.axvline(n_genes_upper, color='red')
        plt.savefig(prefix+"_scatterplot_threshold.png")
    
    # percent_mito
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(data.obs.percent_mito)
        plt.axvline(args_pct_mito, color='red')
        plt.xlabel('percent_mito', fontsize=12)
        plt.savefig(prefix+"_histplot_percent_mito.png")
        
    ## apply QC filter
    pg.qc_metrics(data, 
                  min_genes=n_genes_lower, max_genes=n_genes_upper,
                  min_umis=n_counts_lower, max_umis=n_counts_upper,
                  mito_prefix='MT-', percent_mito=args_pct_mito)
    
    df = pg.get_filter_stats(data)
    df.to_csv(prefix+"_filter_stats.csv")
    
    #####################
    ### QC by n_cells ###
    #####################
    
    n_cells_before_qc = data.obs.Channel.value_counts().rename_axis('Channel').reset_index(name='counts')
    n_cells_after_qc = data.obs[data.obs.passed_qc].Channel.value_counts().rename_axis('Channel').reset_index(name='counts')
    
    print('n_cells before QC:', np.sum(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
    print('n_cells after QC:', np.sum(n_cells_after_qc[n_cells_after_qc.counts>0].counts))
    
    print('mean n_cells before QC', np.mean(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
    print('mean n_cells after QC', np.mean(n_cells_after_qc[n_cells_after_qc.counts>0].counts))
    
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(np.log10(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
        plt.axvline(np.log10(args_min_n_cells), color='red')
        plt.xlabel('log10(n_cells)', fontsize=12)
        plt.savefig(prefix+"_histplot_n_cells.png")
    
    ### n_cells QC
    n_cells_outlier = list(n_cells_after_qc[n_cells_after_qc.counts<args_min_n_cells].Channel)
    data.obs.loc[data.obs.Channel.isin(n_cells_outlier),'passed_qc'] = False
    print('remove %i donors that have cells less than %i: %s' % (len(n_cells_outlier),args_min_n_cells,n_cells_outlier))
    
    ### filter cells
    pg.filter_data(data)
    
    ### clean unused categories
    data.obs['Channel'] = data.obs.Channel.cat.remove_unused_categories()
    
    ### save
    data.to_anndata().write(output)

def log1p_norm(input, output):
    adata = sc.read_h5ad(input)
    adata.layers["counts"] = adata.X
    
    # shifted log1p transform
    scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
    adata.X = sc.pp.log1p(scales_counts["X"], copy=True)

    adata.write(output)

def sig_score(input, output):
    data = pg.read_input(input)
    
    pg.calc_signature_score(data, 'cell_cycle_human') ## 'cycle_diff', 'cycling', 'G1/S', 'G2/M' ## cell cycle gene score based on [Tirosh et al. 2015 | https://science.sciencemag.org/content/352/6282/189]
    pg.calc_signature_score(data, 'gender_human') # female_score, male_score
    pg.calc_signature_score(data, 'mitochondrial_genes_human') # 'mito_genes' contains 13 mitocondrial genes from chrM and 'mito_ribo' contains mitocondrial ribosomal genes that are not from chrM
    pg.calc_signature_score(data, 'ribosomal_genes_human') # ribo_genes
    pg.calc_signature_score(data, 'apoptosis_human') # apoptosis

    pge.save(data, output)

### QC #########################################################################

qc(input = args.input, output = args.input.replace('.h5ad','.qc.h5ad'), prefix = args.input.replace('.h5ad',''))

gc.collect()

log1p_norm(input = args.input.replace('.h5ad','.qc.h5ad'), output = args.input.replace('.h5ad','.qc_norm.h5ad'))

gc.collect()

sig_score(input = args.input.replace('.h5ad','.qc_norm.h5ad'), output = args.output)

################################################################################
