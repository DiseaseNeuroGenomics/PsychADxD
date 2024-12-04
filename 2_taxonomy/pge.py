### pegasus extentions
### collection of custom pegasus functions
### Donghoon Lee

# sc
import pegasus as pg
import scanpy as sc
import anndata as ad
from anndata.tests.helpers import assert_equal
from anndata._core.sparse_dataset import SparseDataset
from anndata.experimental import read_elem, write_elem

# data
import numpy as np
import pandas as pd
import numpy_groupies as npg # required for pseudoMetaCellByGroup
from scipy import stats, sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cross_decomposition import PLSRegression
import h5py
from collections import defaultdict

# plot
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import seaborn as sns
from adjustText import adjust_text

sns.set_style("whitegrid", {'axes.grid' : False})
sc.set_figure_params(scanpy=True, dpi=100, dpi_save=300, fontsize=12, color_map = 'YlOrRd') #'viridis_r'
sc.settings.verbosity = 1
sc.logging.print_header()

# sys
import gc
from pathlib import Path
import time
import logging
logger = logging.getLogger("pegasus")

def info():
    print("pegasus extentions - collection of custom functions for pegasus")
    print('pegasus v%s' % pg.__version__)
    print('scanpy v%s' % sc.__version__)
    print('anndata v%s' % ad.__version__)
    print('numpy v%s' % np.__version__)
    print('pandas v%s' % pd.__version__)

def scanpy_hvf(data, flavor='cell_ranger', batch_key=None, min_mean=0.0125, max_mean=3.0, min_disp=0.5, n_top_genes=None, robust_protein_coding=False, protein_coding=False, autosome=False):
    
    ### scanpy HVF
    adata=data.to_anndata()

    if robust_protein_coding:
        # if adata.uns:
        #     del adata.uns
        # adata._inplace_subset_var(adata.var['robust_protein_coding']) # makes a copy, bad for mem
        adata=adata[:,adata.var.robust_protein_coding]

    if protein_coding:
        adata=adata[:,adata.var.gene_type=='protein_coding']

    if autosome:
        adata=adata[:,~adata.var.gene_chrom.isin(['MT', 'X', 'Y'])]

    # seurat_v3 expects raw counts
    if flavor=='seurat_v3':
        adata.X = adata.raw.X
        if n_top_genes is None:
            raise ValueError('`n_top_genes` is mandatory if `flavor` is `seurat_v3`.')

    # find highly variable genes
    hvg = sc.pp.highly_variable_genes(adata, flavor=flavor, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, batch_key=batch_key, n_top_genes=n_top_genes, inplace=False, subset=False)
    print(hvg.highly_variable.value_counts())
    
    # plot hvf
    sc.pl.highly_variable_genes(hvg)

    # set robust genes and scanpy hvg as hvf
    data.var.highly_variable_features = False
    data.var.loc[hvg[hvg.highly_variable].index,'highly_variable_features'] = True

    # if protein_coding is True, remove non protein_coding genes from hvf
    if protein_coding:
        data.var.loc[~data.var.protein_coding, 'highly_variable_features'] = False
    else:
        # sanity check only if not protein_coding
        if set(data.var[data.var.highly_variable_features].index) != set(hvg[hvg.highly_variable].index):
            raise ValueError('`highly_variable_genes` is not the same as `highly_variable_features`.')    

    # final value counts
    print(data.var.highly_variable_features.value_counts())

def scanpy_hvf_h5ad(h5ad_file, flavor='cell_ranger', batch_key=None, min_mean=0.0125, max_mean=3.0, min_disp=0.5, n_top_genes=None, robust_protein_coding=False, protein_coding=False, autosome=False):
    
    ### scanpy HVF
    adata=sc.read_h5ad(h5ad_file)
    print(adata)
    
    if robust_protein_coding:
        print('subset robust_protein_coding')
        # if adata.uns:
        #     del adata.uns
        # adata._inplace_subset_var(adata.var['robust_protein_coding']) # makes a copy, bad for mem
        adata=adata[:,adata.var.robust_protein_coding]

    if protein_coding:
        adata=adata[:,adata.var.gene_type=='protein_coding']

    if autosome:
        adata=adata[:,~adata.var.gene_chrom.isin(['MT', 'X', 'Y'])]

    # seurat_v3 expects raw counts
    if flavor=='seurat_v3':
        adata.X = adata.raw.X
        if n_top_genes is None:
            raise ValueError('`n_top_genes` is mandatory if `flavor` is `seurat_v3`.')

    # find highly variable genes
    print('scanpy hvg')
    hvg = sc.pp.highly_variable_genes(adata, flavor=flavor, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, batch_key=batch_key, n_top_genes=n_top_genes, inplace=False, subset=False)
    print(hvg.highly_variable.value_counts())
    
    # plot hvf
    sc.pl.highly_variable_genes(hvg)

    return adata.var.index[hvg.highly_variable].tolist()
    
def scanpy_pca(data, n_comps=50, use_highly_variable=True):

    ### scanpy PCA
    adata=data.to_anndata()
    adata.var['highly_variable'] = adata.var.highly_variable_features
    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=use_highly_variable, svd_solver='arpack')

    data.obsm["X_pca"] = adata.obsm['X_pca']
    data.uns["PCs"] = adata.varm['PCs']
    data.uns["pca_variance"] = adata.uns['pca']['variance']
    data.uns["pca_variance_ratio"] = adata.uns['pca']['variance_ratio']
    
def proc_h5ad_v2(filepath):
    adata = ad.read_h5ad(filepath)

    ### obs
    adata.obs['Channel'] = [x[0]+'-'+x[1] for x in zip(adata.obs.SubID_cs,adata.obs.rep)]
    adata.obs = adata.obs[['Channel', 'SubID_cs', 'round_num', 'batch', 'prep', 'rep', 'HTO_n_cs']]
    adata.obs.columns = ['Channel', 'SubID', 'round_num', 'batch', 'prep', 'rep', 'HTO']
    adata.obs.batch = [x.replace('-cDNA','') for x in adata.obs.batch]
    adata.obs['barcodekey'] = [x[0]+'-'+x[1] for x in zip(adata.obs.Channel,adata.obs.index)]
    adata.obs.index = adata.obs.barcodekey
    del adata.obs['barcodekey']
    adata.obs['Source'] = [x[0] for x in adata.obs.SubID.tolist()]

    ### var
    adata.var.index = [x.replace('_index','') for x in adata.var.index]

    return(adata)

# read indiv. h5ad files
def proc_h5ad_v3(filepath, dummy_adata):
    # Guarding against overflow for very large datasets
    dummy_adata.X.indptr = dummy_adata.X.indptr.astype(np.int64)
    dummy_adata.X.indices = dummy_adata.X.indices.astype(np.int64)

    adata = ad.read_h5ad(filepath)
    
    # obs
    adata.obs['Channel'] = [x[0]+'-'+x[1] for x in zip(adata.obs.SubID_vS,adata.obs.rep)]
    adata.obs = adata.obs[['Channel', 'SubID_vS', 'rep', 'poolID_ref', 'round_num', 'prep', 'SubID_cs', 'HTO_n_cs', 'max_prob', 'doublet_prob']]
    adata.obs.columns = ['Channel', 'SubID', 'rep', 'poolID', 'round_num', 'prep', 'SubID_cs', 'HTO_n_cs', 'max_prob', 'doublet_prob']
    adata.obs['Source'] = [x[0] for x in adata.obs.SubID.tolist()]
    
    # obs barcodekey
    adata.obs['barcodekey'] = [x[0]+'-'+x[1] for x in zip(adata.obs.Channel,adata.obs.index)]
    adata.obs.index = adata.obs.barcodekey
    del adata.obs['barcodekey']
    
    # var
    adata.var.index = [x.replace('_index','') for x in adata.var.index]
    
    # expand var
    adata = ad.concat([dummy_adata, adata], join='outer', merge=None)
    adata.X.sort_indices()
    
    # assert_equal var_names order is sorted
    assert_equal(adata.var_names.to_list(), sorted(adata.var_names))
    
    # Guarding against overflow for very large datasets
    adata.X.indptr = adata.X.indptr.astype(np.int64)
    adata.X.indices = adata.X.indices.astype(np.int64)
    
    return(adata)

def proc_manifest(manifest_file, prefix, postfix, chunk_size, dummy_adata):
    list_h5ad_parts = []

    # read manifest
    df = pd.read_csv(manifest_file)

    # split data into chunks
    chunks = [df.Location[i:i+chunk_size] for i in range(0, len(df.Location), chunk_size)]

    for j,list_filepath in enumerate(chunks):
        # make list of adata
        list_adata = []
        for i,filepath in enumerate(list_filepath):
            print(filepath)
            list_adata.append(proc_h5ad_v3(filepath, dummy_adata))

        # make part file name
        outfilename = prefix+'_raw_'+postfix+str((j+1))+'.h5ad'

        # write h5ad
        adata = ad.concat(list_adata, join='outer', merge=None)
        adata.X.sort_indices()

        # Guarding against overflow for very large datasets
        adata.X.indptr = adata.X.indptr.astype(np.int64)
        adata.X.indices = adata.X.indices.astype(np.int64)

        # save
        adata.write(outfilename)

        # store outfilename
        list_h5ad_parts.append(outfilename)

        # free up memory
        del list_adata
        del adata
        gc.collect()
        
    return(list_h5ad_parts)

def qc_boundary(counts, k=3):
    x = np.log1p(counts)
    mad = stats.median_abs_deviation(x)
    return np.exp(np.median(x) - k*mad), np.exp(np.median(x) + k*mad)

### aggregates mean normalized expression per cluster
def agg_by_cluster(h5ad_file, cluster_label):
    
    # read h5ad
    adata = sc.read_h5ad(h5ad_file)

    # copy count data (assuming its available in raw.X)
    adata.X = adata.raw.X
    
    # log normalize and scale
    sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    norm_scaled_counts = sc.pp.scale(adata, copy=True).X # scaling helps to normalize variance across genes

    labels = sorted(list(set(adata.obs[cluster_label])))
    mat = []
    for lab in labels:
        mat.append(np.mean(norm_scaled_counts[adata.obs[cluster_label]==lab,:],axis=0).reshape(1,-1))
    mat = np.concatenate(mat,axis=0)

    mat = pd.DataFrame(mat)
    mat.index=labels
    mat.columns=adata.var.index
    return(mat)

### aggregates pseudobulk per cluster
def pb_agg_by_cluster(h5ad_file, cluster_label, robust_var_label=None, log1p=False, PFlog1pPF=False, mat_key='raw.X'):
    
    # read h5ad
    data = pg.read_input(h5ad_file)

    # create pseudobulk and normalize
    pb = pg.pseudobulk(data, sample=cluster_label, mat_key=mat_key)
    pb.uns['modality'] = 'rna'

    # robust_var_label
    if robust_var_label:
        pb.var['robust'] = data.var[robust_var_label]
    else:
        pb.var['robust'] = np.array(np.mean(data.X, axis=0)).flatten()>0

    # subset
    pb._inplace_subset_var(pb.var.robust)
    
    if(log1p):

        adata = pb.to_anndata()

        # log1pPF, log1p transform of proportional fitting to mean of cell depth
        adata.layers["log1pPF"] = sc.pp.log1p(sc.pp.normalize_total(adata, target_sum=None, inplace=False)["X"])

        mat = pd.DataFrame(adata.layers["log1pPF"])

    elif(PFlog1pPF):
        
        adata = pb.to_anndata()

        # log1pPF, log1p transform of proportional fitting to mean of cell depth
        adata.layers["log1pPF"] = sc.pp.log1p(sc.pp.normalize_total(adata, target_sum=None, inplace=False)["X"])

        # PFlog1pPF, additional proportional fitting
        adata.layers["PFlog1pPF"] = sc.pp.normalize_total(adata, target_sum=None, layer="log1pPF", inplace=False)["X"]
        
        mat = pd.DataFrame(adata.layers["PFlog1pPF"])

    else:   
        
        # log normalize and scale
        pg.log_norm(pb)
        mat = pd.DataFrame(pb.X)
    
    mat.columns = pb.var.index
    mat.index = pb.obs.index
    return(mat)

### compares clusters via cosine similarity
def cos_similarity(mat1, mat2):
    
    # find shared genes
    common_columns = [x for x in mat1.columns if x in mat2.columns]
    print('Shared gene names:', len(common_columns))

    merged = pd.concat([mat1.loc[:,common_columns],
                        mat2.loc[:,common_columns]])

    # annot
    cos_dist = pd.DataFrame(cosine_similarity(merged))
    cos_dist.index = merged.index
    cos_dist.columns = merged.index

    cos_dist = cos_dist.iloc[len(mat1.index):,:len(mat1.index)]
    return(cos_dist)

### compares clusters via L2 distance
def l2_distance(mat1, mat2):
    
    # find shared genes
    common_columns = [x for x in mat1.columns if x in mat2.columns]
    print('Shared gene names:', len(common_columns))

    merged = pd.concat([mat1.loc[:,common_columns],
                        mat2.loc[:,common_columns]])

    # annot
    l2_dist = pd.DataFrame(pairwise_distances(merged))
    l2_dist.index = merged.index
    l2_dist.columns = merged.index

    l2_dist = l2_dist.iloc[len(mat1.index):,:len(mat1.index)]
    return(l2_dist)

### compares clusters via spearman_corr
def spearman_corr(mat1, mat2):
    
    # find shared genes
    common_genes = sorted(set(mat1.columns).intersection(set(mat2.columns)))
    print('Shared gene names:', len(common_genes))

    corr = np.zeros((len(mat2.index),len(mat1.index)))

    for m1 in range(len(mat1.index)):
        for m2 in range(len(mat2.index)):
            corr[m2,m1] = stats.spearmanr(mat1[common_genes].iloc[m1],
                                          mat2[common_genes].iloc[m2]).correlation
    df = pd.DataFrame(corr)
    df.columns = mat1.index
    df.index = mat2.index
    return(df)

### compares clusters via pearson_corr
def pearson_corr(mat1, mat2):
    
    # find shared genes
    common_genes = sorted(set(mat1.columns).intersection(set(mat2.columns)))
    print('Shared gene names:', len(common_genes))

    corr = np.zeros((len(mat2.index),len(mat1.index)))

    for m1 in range(len(mat1.index)):
        for m2 in range(len(mat2.index)):
            corr[m2,m1] = stats.pearsonr(mat1[common_genes].iloc[m1],
                                         mat2[common_genes].iloc[m2]).statistic
    df = pd.DataFrame(corr)
    df.columns = mat1.index
    df.index = mat2.index
    return(df)

### scree plot
def scree_plot(data):
    fig = plt.figure()
    plt.plot(range(1,data.uns['PCs'].shape[1]+1), data.uns['pca']['variance_ratio'], 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    return fig

### calc signature scores
def calc_sig_scores(data):
    pg.calc_signature_score(data, 'cell_cycle_human') ## 'cycle_diff', 'cycling', 'G1/S', 'G2/M' ## cell cycle gene score based on [Tirosh et al. 2015 | https://science.sciencemag.org/content/352/6282/189]
    pg.calc_signature_score(data, 'gender_human') # female_score, male_score
    pg.calc_signature_score(data, 'mitochondrial_genes_human') # 'mito_genes' contains 13 mitocondrial genes from chrM and 'mito_ribo' contains mitocondrial ribosomal genes that are not from chrM
    pg.calc_signature_score(data, 'ribosomal_genes_human') # ribo_genes
    pg.calc_signature_score(data, 'apoptosis_human') # apoptosis
    return data.obs

### find correlated features
def corrFeatures(data, gene, top_n=50, sample_size=5000, vmin=0.0, vmax=1.0):
    # random sample
    obs_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
    # covariate matrix
    cov_mat = np.cov(data.X[obs_indices].T.todense())
    # index of gene
    idx = data.var.index.get_loc(gene)
    # arg sort
    top_cor_idx = np.argsort(cov_mat[idx])
    # subset cov mat
    corr = cov_mat[top_cor_idx[-top_n:]][:,top_cor_idx[-top_n:]]
#     # fill 1 diagonally
#     np.fill_diagonal(corr, 0)
    # mask
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    corr=pd.DataFrame(corr)
    corr.columns = data.var.iloc[top_cor_idx[-top_n:]].index.tolist()
    corr.index = data.var.iloc[top_cor_idx[-top_n:]].index.tolist()

    with rc_context({'figure.figsize': (12, 12)}):
        sns.set(style='white', font_scale=0.75)
        sns.heatmap(corr, mask=mask, vmin=vmin, vmax=vmax, square=True, cmap='YlGnBu', linewidth=0.1)
    return corr.index.tolist()

### new correlated features
def corrMat(data, sample_size=1000, corr_method='spearman'):
    
    # random sample
    obs_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)

    # dataframe
    exp_df = pd.DataFrame(data.X[obs_indices].todense())
    exp_df.columns = data.var_names

    # subset genes with non-zero variance
    exp_df = exp_df.loc[:,exp_df.var(axis=0)!=0].copy()

    # correlation matrix
    corr=exp_df.corr(method=corr_method)
    
    return corr

def corrPlot(corr, gene, top_n=50, vmin=-1.0, vmax=1.0, figsize=15, clustermap=False):
    
    # index of gene
    idx = corr.index.get_loc(gene)

    # arg sort
    top_cor_idx = np.argsort(abs(corr.iloc[idx]))

    # subset corr mat
    subset = corr.iloc[top_cor_idx[-top_n:],top_cor_idx[-top_n:]]

    # plot
    with rc_context({'figure.figsize': (figsize, figsize)}):
        sns.set(style='white', font_scale=0.7)
        if clustermap:
            sns.clustermap(subset, vmin=vmin, vmax=vmax, cmap='vlag', linewidth=0.1)
        else:
            sns.heatmap(subset, vmin=vmin, vmax=vmax, cmap='vlag', linewidth=0.1, square=True)
        
    return subset

def corrFeatures2(data, gene, top_n=50, sample_size=1000, vmin=-1.0, vmax=1.0):
    
    # random sample
    obs_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)

    # dataframe
    exp_df = pd.DataFrame(data.X[obs_indices].todense())
    exp_df.columns = data.var_names

    # subset genes with non-zero variance
    exp_df = exp_df.loc[:,exp_df.var(axis=0)!=0].copy()

    # correlation matrix
    corr=exp_df.corr()

    # index of gene
    idx = corr.index.get_loc(gene)

    # arg sort
    top_cor_idx = np.argsort(abs(corr.iloc[idx]))

    # subset corr mat
    subset = corr.iloc[top_cor_idx[-top_n:],top_cor_idx[-top_n:]]

    # plot
    with rc_context({'figure.figsize': (15, 15)}):
        sns.set(style='white', font_scale=0.75)
        sns.clustermap(subset, vmin=vmin, vmax=vmax, cmap='vlag', linewidth=0.1)
        
    return subset

### QC: mark MitoCarta genes
def mark_MitoCarta(data):
    ### define mitocarta_genes
    mitocarta = pd.read_csv('/sc/arion/projects/CommonMind/leed62/ref/MitoCarta/Human.MitoCarta3.0.csv')
    data.var['mitocarta_genes'] = [True if x in list(mitocarta.Symbol) else False for x in data.var.index ]

    ### set all mitocarta genes as non-robust gene
    data.var.loc[data.var.mitocarta_genes, 'robust'] = False

    return data.var[data.var['robust']]

### PseudoMetaCells By Group
def pseudoMetaCellByGroup(adata, groupby, rep='X_pca_regressed_harmony', n_pcs=25, n_neighbors=15, weighted_dist_thres = 0.25, k=3, metric='cosine', postfix='pmc'):
    list_group = list(sorted(set(adata.obs[groupby])))
    for g in list_group:
        print('Processing:',g)

        # subset
        adata_sub = adata[adata.obs[groupby]==g].copy()

        # calc knn
        sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, use_rep=rep, n_pcs=n_pcs, knn=True, method='umap', metric=metric, key_added='pmc_neighbors')
        
        # get pseudoMetaCell grouping
        pmc = _getPMC(adata_sub, weighted_dist_thres, k)
        
        keylist = []
        vallist = []
        for key,val in pmc.items():
            vallist=vallist+val
            keylist=keylist+[key]*len(val)
        print('n PMC barcodes:',len(pmc))
        print('average barcodes per PMC:',np.mean([len(value) for key,value in pmc.items()]))
        
        # aggregate
        groups = [x for x in [dict(zip(vallist,keylist))[x] for x in adata_sub.obs.index.tolist()]]
        pmc_ct = npg.aggregate(groups, np.array(adata_sub.raw.X.todense()), func='sum', axis=0, fill_value=0)
#         print(pmc_ct.dtype)
        
        # save new adata
        adata_new = ad.AnnData(pmc_ct, dtype=np.float32)
        adata_new.var = adata_sub.var[['featureid']]
        adata_new.obs['original_barcodekey'] = [','.join(pmc[int(x)]) for x in adata_new.obs.index.tolist()]
        adata_new.obs['barcodekey'] = [g+'_'+'{:05d}'.format(int(x)+1) for x in adata_new.obs.index.tolist()]
        adata_new.obs.index = adata_new.obs.barcodekey        
        del adata_new.obs['barcodekey']

        # save h5ad
        adata_new.raw = adata_new
        adata_new.write(g+'_'+postfix+'.h5ad')
        print('Saved',g+'_'+postfix+'.h5ad')

def _getPMC(adata, weighted_dist_thres, k):

    # donor graph
    donor_graph = np.array(adata.obsp['pmc_neighbors_connectivities'].todense())
    print('n cell barcodes:', donor_graph.shape[0])

    # fill 0 diagonally
    np.fill_diagonal(donor_graph, 0)

    # list of barcodes
    bc_list = adata.obs.index.to_list()

    # save barcodes pool
    bc_pool = bc_list.copy()

    # pseudo-meta-cell
    pmc = {}

    # reset index
    i = 0

    while len(bc_pool)>0:
        bc = bc_pool[0]
        bc_idx = bc_list.index(bc)

        # subgraph argsort
        sub_graph = donor_graph[bc_idx,:]
        sub_graph_idx = np.where(sub_graph>weighted_dist_thres)[0]
        sub_graph_idx_argsort = np.array(np.argsort(sub_graph[sub_graph_idx])).flatten()[::-1]
        sort_idx = sub_graph_idx[sub_graph_idx_argsort]

        # metacell
        metacell_final = [bc]
        bc_pool.remove(bc)
        for idx in sort_idx:
            b = bc_list[idx]
            b_idx = bc_list.index(b)
            s = donor_graph[bc_idx,b_idx]

            # if dist is greater than dist to other cells
#             if (b in bc_pool) & (s >= np.max(np.delete(donor_graph[:,b_idx],[bc_idx]))):
            if (b in bc_pool):
                metacell_final.append(b)
                bc_pool.remove(b)
            if len(metacell_final) >= k:
                break

        # make list
        pmc[i]=metacell_final
        i+=1
    #     print(metacell_final)
    
    return(pmc)

### clean unused categories
def clean_unused_categories(data):
    for obs_name in data.obs.columns:
        if data.obs[obs_name].dtype=='category':
            print('Removing unused categories from',obs_name)
            data.obs[obs_name] = data.obs[obs_name].cat.remove_unused_categories()
    for var_name in data.var.columns:
        if data.var[var_name].dtype=='category':
            print('Removing unused categories from',var_name)
            data.var[var_name] = data.var[var_name].cat.remove_unused_categories()
    return data

def read_everything_but_X(pth) -> ad.AnnData:
    # read all keys but X and raw
    with h5py.File(pth) as f:
        attrs = list(f.keys())
        attrs.remove('X')
        if 'raw' in attrs:
            attrs.remove('raw')
        adata = ad.AnnData(**{k: read_elem(f[k]) for k in attrs})
        print(adata.shape)
    return adata

def csc2csr_on_disk(input_pth, output_pth):
    """
    Params
    ------
    input_pth
        h5ad file in csc format
    output_pth
        h5ad file to write in csr format
    """
    annotations = read_everything_but_X(input_pth)
    annotations.write_h5ad(output_pth)
    n_variables = annotations.shape[1]
    del annotations

    with h5py.File(output_pth, 'w') as target:
        with h5py.File(input_pth, "r") as src:
            # Convert to csr format
            csc_mat = sparse.csc_matrix((src['X']['data'], src['X']['indices'], src['X']['indptr']))
            csr_mat = csc_mat.tocsr()
            write_elem(target, 'X', csr_mat)

def concat_on_disk(input_pths, output_pth, temp_pth='temp.h5ad'):
    """
    Params
    ------
    input_pths
        Paths to h5ad files which will be concatenated
    output_pth
        File to write as a result
    """
    annotations = ad.concat([read_everything_but_X(pth) for pth in input_pths])
    annotations.write_h5ad(output_pth)
    n_variables = annotations.shape[1]
    
    del annotations

    with h5py.File(output_pth, 'a') as target:
        
        # initiate empty X
        dummy_X = sparse.csr_matrix((0, n_variables), dtype=np.float32)
        dummy_X.indptr = dummy_X.indptr.astype(np.int64) # Guarding against overflow for very large datasets
        dummy_X.indices = dummy_X.indices.astype(np.int64) # Guarding against overflow for very large datasets
        write_elem(target, 'X', dummy_X)
        
        # append
        mtx = SparseDataset(target['X'])
        for p in input_pths:
            with h5py.File(p, 'r') as src:
                
                # IF: src is in csc format, convert to csr and save to temp_pth
                if src['X'].attrs['encoding-type']=='csc_matrix':

                    # Convert to csr format
                    csc_mat = sparse.csc_matrix((src['X']['data'], src['X']['indices'], src['X']['indptr']))
                    csr_mat = csc_mat.tocsr()         
                    
                    # save to temp_pth
                    with h5py.File(temp_pth, 'w') as tmp:
                        write_elem(tmp, 'X', csr_mat)
                    
                    # read from temp_pth
                    with h5py.File(temp_pth, 'r') as tmp:
                        mtx.append(SparseDataset(tmp['X']))
                        
                # ELSE: src is in csr format
                else:
                    mtx.append(SparseDataset(src['X']))
                
def write_h5ad_with_new_annotation(original_h5ad, adata, new_h5ad, raw = False):
    # new annotation
    new_uns=None
    if adata.uns:
        new_uns = adata.uns
    new_obsm=None
    if adata.obsm:
        new_obsm = adata.obsm
    new_varm=None
    if adata.varm:
        new_varm = adata.varm
    new_obsp=None
    if adata.obsp:
        new_obsp = adata.obsp
    new_varp=None
    if adata.varp:
        new_varp = adata.varp

    # save obs and var first
    ad.AnnData(None, obs=adata.obs, var=adata.var, uns=new_uns, obsm=new_obsm, varm=new_varm, obsp=new_obsp, varp=new_varp).write(new_h5ad)

    # append X
    with h5py.File(new_h5ad, "a") as target:
        # make dummy
        dummy_X = sparse.csr_matrix((0, adata.var.shape[0]), dtype=np.float32)
        dummy_X.indptr = dummy_X.indptr.astype(np.int64) # Guarding against overflow for very large datasets
        dummy_X.indices = dummy_X.indices.astype(np.int64) # Guarding against overflow for very large datasets
        
        with h5py.File(original_h5ad, "r") as src:
            write_elem(target, "X", dummy_X)
            SparseDataset(target["X"]).append(SparseDataset(src["X"]))
            # append raw/X if needed
            if raw:
                write_elem(target, "raw/X", dummy_X)
                SparseDataset(target["raw/X"]).append(SparseDataset(src["raw/X"]))

def ondisk_subset(orig_h5ad, new_h5ad, subset_obs, subset_var = None, chunk_size = 500000, raw = False, adata = None):

    if adata is None:
        
        # read annotations only
        adata = read_everything_but_X(orig_h5ad)

        # subset obs
        if subset_obs is not None:
            adata._inplace_subset_obs(subset_obs)

        # subset var
        if subset_var is not None:
            adata._inplace_subset_var(subset_var)

        # clean unused cat
        adata = clean_unused_categories(adata)
        
    # new annotation
    new_uns=None
    if adata.uns:
        new_uns = adata.uns

    new_obsm=None
    if adata.obsm:
        new_obsm = adata.obsm

    new_varm=None
    if adata.varm:
        new_varm = adata.varm

    new_obsp=None
    if adata.obsp:
        new_obsp = adata.obsp

    new_varp=None
    if adata.varp:
        new_varp = adata.varp

    new_layers=None
    if adata.layers:
        new_layers = adata.layers
    
    # save obs and var first
    ad.AnnData(None, obs=adata.obs, var=adata.var, uns=new_uns, obsm=new_obsm, varm=new_varm, obsp=new_obsp, varp=new_varp, layers=new_layers).write(new_h5ad)
    
    # initialize new_h5ad
    with h5py.File(new_h5ad, "a") as target:
        dummy_X = sparse.csr_matrix((0, adata.var.shape[0]), dtype=np.float32)
        dummy_X.indptr = dummy_X.indptr.astype(np.int64) # Guarding against overflow for very large datasets
        dummy_X.indices = dummy_X.indices.astype(np.int64) # Guarding against overflow for very large datasets
        write_elem(target, "X", dummy_X)
        if raw:
            write_elem(target, "raw/X", dummy_X)
        
    # get indptr first
    with h5py.File(orig_h5ad, 'r') as f:
        csr_indptr = f['X/indptr'][:]

    # append subset of X
    for idx in [i for i in range(0, csr_indptr.shape[0]-1, chunk_size)]:
        print('Processing', idx, 'to', idx+chunk_size)
        row_start, row_end = idx, idx+chunk_size

        if sum(subset_obs[row_start:row_end])>0:
            # X
            with h5py.File(orig_h5ad, 'r') as f:
                tmp_indptr = csr_indptr[row_start:row_end+1]
                
                new_data = f['X/data'][tmp_indptr[0]:tmp_indptr[-1]]
                new_indices = f['X/indices'][tmp_indptr[0]:tmp_indptr[-1]]
                new_indptr = tmp_indptr - csr_indptr[row_start]
                
                if subset_var is not None:
                    new_shape = [tmp_indptr.shape[0]-1, len(subset_var)]
                    tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                    tmp_csr = tmp_csr[subset_obs[row_start:row_end]][:,subset_var]
                else:
                    new_shape = [tmp_indptr.shape[0]-1, adata.shape[1]]
                    tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                    tmp_csr = tmp_csr[subset_obs[row_start:row_end]]
                    
                tmp_csr.sort_indices()

            # append X
            with h5py.File(new_h5ad, "a") as target:
                mtx = SparseDataset(target["X"])
                mtx.append(tmp_csr)

            # raw/X
            if raw:
                with h5py.File(orig_h5ad, 'r') as f:
                    tmp_indptr = csr_indptr[row_start:row_end+1]
                    
                    new_data = f['raw/X/data'][tmp_indptr[0]:tmp_indptr[-1]]
                    new_indices = f['raw/X/indices'][tmp_indptr[0]:tmp_indptr[-1]]
                    new_indptr = tmp_indptr - csr_indptr[row_start]
                    
                    if subset_var is not None:
                        new_shape = [tmp_indptr.shape[0]-1, len(subset_var)]
                        tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                        tmp_csr = tmp_csr[subset_obs[row_start:row_end]][:,subset_var]
                    else:
                        new_shape = [tmp_indptr.shape[0]-1, adata.shape[1]]
                        tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                        tmp_csr = tmp_csr[subset_obs[row_start:row_end]]

                    tmp_csr.sort_indices()

                # append raw/X
                with h5py.File(new_h5ad, "a") as target:
                    mtx = SparseDataset(target["raw/X"])
                    mtx.append(tmp_csr)

def pal_max():
    max_268 = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059","#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87","#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80","#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100","#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F","#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09","#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66","#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C","#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81","#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00","#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700","#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329","#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C","#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800","#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51","#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58","#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D","#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176","#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5","#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4","#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01","#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966","#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0","#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C","#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868","#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183","#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433","#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F","#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E","#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F","#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00","#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66","#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25","#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]
    return ','.join(max_268)

class PySankeyException(Exception):
    pass


class NullsInFrame(PySankeyException):
    pass


class LabelMismatch(PySankeyException):
    pass

def check_data_matches_labels(labels, data, side):
    if len(labels) > 0:
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch('{0} labels and data do not match.{1}'.format(side, msg))

def sankey(left, right, leftWeight=None, rightWeight=None, colorDict=None,
           leftLabels=None, rightLabels=None, aspect=4, rightColor=False,
           fontsize=10, figureName=None, closePlot=False, size_x=6, size_y=12):
    '''
    Make Sankey Diagram showing flow from left-->right
    Inputs:
        left = NumPy array of object labels on the left of the diagram
        right = NumPy array of corresponding labels on the right of the diagram
            len(right) == len(left)
        leftWeight = NumPy array of weights for each strip starting from the
            left of the diagram, if not specified 1 is assigned
        rightWeight = NumPy array of weights for each strip starting from the
            right of the diagram, if not specified the corresponding leftWeight
            is assigned
        colorDict = Dictionary of colors to use for each label
            {'label':'color'}
        leftLabels = order of the left labels in the diagram
        rightLabels = order of the right labels in the diagram
        aspect = vertical extent of the diagram in units of horizontal extent
        rightColor = If true, each strip in the diagram will be be colored
                    according to its left label
    Ouput:
        None
    '''
    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))

    if len(rightWeight) == 0:
        rightWeight = leftWeight

    plt.figure()
    plt.rc('text', usetex=False)
    plt.rc('font', family='sans-serif')

    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                              'rightWeight': rightWeight}, index=range(len(left)))

    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise NullsInFrame('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.r_[dataFrame.left.unique(), dataFrame.right.unique()]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(dataFrame.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(dataFrame.right.unique()).unique()
    else:
        check_data_matches_labels(rightLabels, dataFrame['right'], 'right')
    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The colorDict parameter is missing values for the following labels : "
            msg += '{}'.format(', '.join(missing))
            raise ValueError(msg)

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].leftWeight.sum()
            rightDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].rightWeight.sum()
        ns_l[leftLabel] = leftDict
        ns_r[leftLabel] = rightDict

    # Determine positions of left label patches and total widths
    leftWidths = defaultdict()
    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD['left'] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
            myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + 0.02 * dataFrame.leftWeight.sum()
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        leftWidths[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths = defaultdict()
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD['right'] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
            myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + 0.02 * dataFrame.rightWeight.sum()
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        rightWidths[rightLabel] = myD

    # Total vertical extent of diagram
    xMax = topEdge / aspect

    # Draw vertical bars on left and right of each  label's section & print label
    for leftLabel in leftLabels:
        plt.fill_between(
            [-0.02 * xMax, 0],
            2 * [leftWidths[leftLabel]['bottom']],
            2 * [leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']],
            color=colorDict[leftLabel],
            alpha=0.99
        )
        plt.text(
            -0.05 * xMax,
            leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
            leftLabel,
            {'ha': 'right', 'va': 'center'},
            fontsize=fontsize
        )
    for rightLabel in rightLabels:
        plt.fill_between(
            [xMax, 1.02 * xMax], 2 * [rightWidths[rightLabel]['bottom']],
            2 * [rightWidths[rightLabel]['bottom'] + rightWidths[rightLabel]['right']],
            color=colorDict[rightLabel],
            alpha=0.99
        )
        plt.text(
            1.05 * xMax,
            rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
            rightLabel,
            {'ha': 'left', 'va': 'center'},
            fontsize=fontsize
        )

    # Plot strips
    for leftLabel in leftLabels:
        for rightLabel in rightLabels:
            labelColor = leftLabel
            if rightColor:
                labelColor = rightLabel
            if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
                # Create array of y values for each strip, half at left value,
                # half at right, convolve
                ys_d = np.array(50 * [leftWidths[leftLabel]['bottom']] + 50 * [rightWidths[rightLabel]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [leftWidths[leftLabel]['bottom'] + ns_l[leftLabel][rightLabel]] + 50 * [rightWidths[rightLabel]['bottom'] + ns_r[leftLabel][rightLabel]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                # Update bottom edges at each label so next strip starts at the right place
                leftWidths[leftLabel]['bottom'] += ns_l[leftLabel][rightLabel]
                rightWidths[rightLabel]['bottom'] += ns_r[leftLabel][rightLabel]
                plt.fill_between(
                    np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65,
                    color=colorDict[labelColor]
                )
    plt.gca().axis('off')
    plt.gcf().set_size_inches(size_x, size_y)
    if figureName != None:
        plt.savefig("{}.png".format(figureName), bbox_inches='tight', dpi=150)
    if closePlot:
        plt.close()

def diff_markers(marker_dict_res, test, ref):
    test_set = marker_dict_res[test]['up']
    test_set = set(test_set[test_set.log2Mean_other<1][:50].index)
    ref_set = marker_dict_res[ref]['up']
    ref_set = set(ref_set[ref_set.log2Mean_other<1][:100].index)
    return test_set - ref_set

### Partial Least Squares Regression
def pls(
    data: ad.AnnData,
    y: str,
    n_components: int = 50,
    features: str = "highly_variable_features",
    standardize: bool = True,
    max_value: float = 10,
) -> None:
    """Perform PLS regression to the data.

    The calculation uses *scikit-learn* implementation of PLSRegression.

    Parameters
    ----------
    data: ``anndata.AnnData``
        Annotated data matrix with rows for cells and columns for genes.
    
    y: ``str``
        Keyword in ``data.obs`` to specify response variable.
        
    n_components: ``int``, optional, default: ``50``.
        Number of Principal Components to get.

    features: ``str``, optional, default: ``"highly_variable_features"``.
        Keyword in ``data.var`` to specify features used for PLS.

    standardize: ``bool``, optional, default: ``True``.
        Whether to scale the data to unit variance and zero mean or not.

    max_value: ``float``, optional, default: ``10``.
        The threshold to truncate data after scaling. If ``None``, do not truncate.

    random_state: ``int``, optional, default: ``0``.
        Random seed to be set for reproducing result.


    Returns
    -------
    ``None``.

    Update ``data.obsm``:

        * ``data.obsm["X_pls"]``: PLS matrix of the data.

    Update ``data.uns``:

        * ``data.uns["PLS_x_loadings"]``: The loadings of PLS transformation.

    Examples
    --------
    >>> pg.pls(adata)
    """

    keyword = pg.select_features(data, features)
    start = time.perf_counter()
    X = data.uns[keyword]
    assert y in data.obs
    
    if standardize:
        # scaler = StandardScaler(copy=False)
        # scaler.fit_transform(X)
        m1 = X.mean(axis=0)
        psum = np.multiply(X, X).sum(axis=0)
        std = ((psum - X.shape[0] * (m1 ** 2)) / (X.shape[0] - 1.0)) ** 0.5
        std[std == 0] = 1
        X -= m1
        X /= std

    if max_value is not None:
        X[X > max_value] = max_value
        X[X < -max_value] = -max_value
        
    # Perform PLS
    pls = PLSRegression(n_components=n_components)
    X_pls = pls.fit_transform(X, data.obs[y].values)[0]

    data.obsm["X_pls"] = X_pls
    data.uns["PLS_x_loadings"] = pls.x_loadings_ # cannot be varm because numbers of features are not the same

    end = time.perf_counter()
    logger.info("PLS is done. Time spent = {:.2f}s.".format(end - start))

# Plot a correlation circle plot for the first two PCs
def plot_correlation_circle(data, rep, features='highly_variable_features'):

#     tmp_X = np.array(data.X[:,data.var.highly_variable_features].todense())
    keyword = pg.select_features(data, features)
    tmp_X = data.uns[keyword]
    if(features):
        feature_names = data.var[data.var[features]].index.tolist()
    else:
        feature_names = data.var.index.tolist()

    (fig, ax) = plt.subplots(figsize=(8, 8), dpi=150)
    texts=[]
    for i in range(0, len(feature_names)):
        corr1, _ = stats.pearsonr(tmp_X[:, i], data.obsm['X_'+rep][:,0])
        corr2, _ = stats.pearsonr(tmp_X[:, i], data.obsm['X_'+rep][:,1])

        ax.arrow(0,
                 0,  # Start the arrow at the origin
                 corr1,  # for PC1
                 corr2,  #1 for PC2
                 head_width=0.01,
                 head_length=0.01,
                 lw=0.1)

        if (np.sqrt(corr1**2+corr2**2)>=0.25):    
            texts.append(plt.text(corr1, corr2, feature_names[i], size=4))

    # similar to ggrepel
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    ax.set_title('Correlation circle plot of '+rep)
    plt.show()

# save to anndata
def save(data, filename):
    if '_tmp_fmat_highly_variable_features' in data.uns:
        del data.uns['_tmp_fmat_highly_variable_features']
    data.to_anndata().write(filename)
    print('Saved',filename)

# calculate zscore per cluster
def calc_zscore_per_cluster(adata, cluster_labels='subclass', variable='n_genes', zscore_name='zscore_subclass_ngenes'):
    adata.obs[zscore_name] = 0
    for s in sorted(adata.obs[cluster_labels].cat.categories):
        print(s)
        adata.obs.loc[adata.obs[cluster_labels]==s, zscore_name] = stats.zscore(np.array(adata[(adata.obs[cluster_labels]==s)].obs[variable]))