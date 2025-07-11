from pathlib import Path
import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
sys.path.append('./')
from run_scanvi import run_scanvi

def parse_dirnames(xenium_dirs:list[Path|str]):
    """
    Parses the directory names of Xenium data to extract metadata.
    Returns a DataFrame with columns: 'date_processed', 'slide_id',
    'region_number', 'patient', 'sample_id'.
    """
    metadata = pd.DataFrame({})
    for idir, dir in enumerate(xenium_dirs):
        # based on directories from extracted zenodo files. See:
        # https://zenodo.org/records/14606776
        name_parts = os.path.basename(dir).split('_')
        slide_id, reg_num, donor = name_parts
        meta = pd.Series({
            'slide_id': slide_id,
            'region_number': int(reg_num),
            'donor': donor,
            'sample_id': f"{slide_id}_R{reg_num}_{donor}"
        })
        metadata = pd.concat([metadata, meta], axis=1)
    metadata = metadata.T.reset_index(drop=True)
    return metadata


def load_xenium(xenium_dirs:list[Path|str],
                sample_metadata:pd.DataFrame|None=None,
                save_dir:Path|str|None=None) -> dict[str, sc.AnnData]|None:
    """
    Load Xenium data and save to disk or return as dictionary of AnnData objects.
    inputs:
    - sample_metadata: is an optional DataFrame with sample-level metadata to
      be added to the AnnData objects. Must match order of xenium_dirs.
    - save_dir: if provided, the function will save the AnnData objects to
      this directory. If None, the function will return a collection of
      AnnData objects.
    returns:
    - adata_dict: if save_dir is `None`, a dictionary where keys are sample
      identifiers and values are the corresponding AnnData objects. The keys
      are derived from the sample_metadata['sample_id'] column if it is
      available. Otherwise, they are derived from the directory names.
    """
    adata_dict = {}
    for ifl, d in enumerate(xenium_dirs):
        # adding 'outs' is consistent with zenodo structure, see:
        # https://zenodo.org/records/14606776
        h5_file = d.joinpath("outs/cell_feature_matrix.h5")
        cell_metadata_file = d.joinpath("outs/cells.csv.gz")
        # load raw data
        adata = sc.read_10x_h5(h5_file)
        adata.layers['counts'] = adata.X.copy()
        # add metadata
        cell_metadata = pd.read_csv(cell_metadata_file).set_index('cell_id')
        adata.obs = cell_metadata
        # total feature counts
        ilocs, fcounts = np.unique(adata.X.nonzero()[0], return_counts=True)
        adata.obs['feature_counts'] = 0
        adata.obs.loc[adata.obs.index[ilocs], 'feature_counts'] = fcounts
        if sample_metadata is not None:
            for col in sample_metadata.columns:
                adata.obs[col] = sample_metadata.iloc[ifl, :][col]
        # move spatial coordinates to obsm
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]]\
            .copy().to_numpy()
        # clean up obs
        adata.obs.drop(['x_centroid', 'y_centroid', 'total_counts'],
                       axis=1, inplace=True)
        # add to dict
        key = sample_metadata.iloc[ifl, :]['sample_id'] \
            if sample_metadata is not None \
            and 'sample_id' in sample_metadata.columns \
            else os.path.basename(xenium_dirs[ifl])
        adata_dict[key] = adata.copy()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for key, adata in adata_dict.items():
            save_path = save_dir.joinpath(f"{key}.h5ad")
            adata.write_h5ad(save_path)
    else:
        return adata_dict
    

def concat_xenium(adatas:list[sc.AnnData], sample_names:list[str],
                  shift_x:str|None=None, shift_y:str|None=None,
                  nest_xy:bool=False, save_dir:Path|str|None=None
                  )-> sc.AnnData|None:
    """
    Merge individual Xenium AnnData objects into a single AnnData object.
    shift_x and shift_y are used to adjust the spatial coordinates of the
    cells in each sample to avoid overlap. Both are expected to be names of
    categorical (or effectively categorical) columns in the `obs` attribute of
    the AnnData objects.
    """
    concat_adata = sc.concat(
        adatas,
        keys = sample_names,
        index_unique='_'
    )
    if shift_x is None and shift_y is None:
        return concat_adata
    # shift positions
    if shift_x is not None:
        xshift_cats = concat_adata.obs[shift_x].unique()
        max_x = concat_adata.obsm['spatial'][:, 0].max()
        xdif = 1.5*max_x
    else:
        xshift_cats = []
    if shift_y is not None:
        yshift_cats = concat_adata.obs[shift_y].unique()
        max_y = concat_adata.obsm['spatial'][:, 1].max()
        ydif = 1.5*max_y
    else:
        yshift_cats = []
    for iy, batch_y in enumerate(yshift_cats):
        batch_idxs = concat_adata.obs[shift_y] == batch_y
        concat_adata.obsm['spatial'][batch_idxs, 1] += iy*ydif
        if nest_xy:
            for ix, batch_x in enumerate(xshift_cats):
                cell_ids = batch_idxs & (concat_adata.obs[shift_x] == batch_x)
                concat_adata.obsm['spatial'][cell_ids, 0] += ix*xdif
    if not nest_xy:
        for ix, batch_x in enumerate(xshift_cats):
            cell_ids = concat_adata.obs[shift_x] == batch_x
            concat_adata.obsm['spatial'][cell_ids, 0] += ix*xdif
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir.joinpath("concatenated.h5ad")
        concat_adata.write_h5ad(save_path)
    else:
        return concat_adata
    

def process_xenium(adata:sc.AnnData, min_counts:int=0, total_counts:int=1e4,
                   do_cluster:bool=True, n_pcs:int=50, leiden_res:float=1.0,
                  save_dir:Path|str|None=None) -> sc.AnnData|None:
    """
    Process a single Xenium AnnData object by filtering cells and clustering.
    min_counts: minimum number of features expressed for a cell to be retained.
    do_cluster: if True, perform clustering using the Leiden algorithm.
    leiden_res: resolution parameter for the Leiden clustering algorithm.
    """
    # filter cells based on transcript counts
    sc.pp.filter_cells(adata, min_counts=min_counts)
    # normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=total_counts)
    sc.pp.log1p(adata)
    # reduce dimensionality and cluster
    if do_cluster:
        sc.pp.pca(adata, n_pcs=n_pcs)
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=leiden_res)
    # save or return data
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir.joinpath("processed.h5ad")
        adata.write_h5ad(save_path)
    else:
        return adata


def filter_by_annot(adata:sc.AnnData, predict_col:str,
                    cluster_col:str='leiden', min_fraction:float=0.9,
                    do_filter:bool=True, save_dir:Path|str|None=None,
                    plot_dir:Path|str|None=None
                    ) -> sc.AnnData|None:
    """
    Filter cells based on annotation predictions.
    Keeps cells that are confidently predicted to a specific class.
    min_fraction: minimum fraction of cells in a cluster that must be
    confidently predicted to keep the cluster.
    If plot_dir is provided, generates a UMAP plot of the annotated data
    showing stably- and unstably-predicted cells, and saves it as a PDF.
    """
    intersection_df = pd.crosstab(
        adata.obs[predict_col], adata.obs[cluster_col]
        )
    clusters_labels = intersection_df.idxmax(axis=0)
    consistent_fractions = (intersection_df / intersection_df.sum(axis=0)) \
        .max(axis=0)
    annot_clust_col = f"{cluster_col}_annotated"
    adata.obs[annot_clust_col] = adata.obs[cluster_col]. \
        astype(str).apply(lambda x: clusters_labels[x]
                          if consistent_fractions[x] >= min_fraction
                          else 'Ambiguous'
                          ).astype('category')
    # reject "Ambiguous" clusters, as well as cells whose annotation is not
    # consistent with the cluster they belong to.
    consistent_nuclei = (
        adata.obs[annot_clust_col] ==
        adata.obs[predict_col].cat.add_categories('Ambiguous')
    )
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_s4c(adata, annot_col=annot_clust_col, ambig_val='Ambiguous',
                save_dir=plot_dir)
    if do_filter:
        filtered_adata = adata[consistent_nuclei, :].copy()
    else:
        filtered_adata = adata.copy()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir.joinpath("filtered.h5ad")
        filtered_adata.write_h5ad(save_path)
    else:
        return filtered_adata


def fig_s4b(adata:sc.AnnData, min_counts:int=30, save_dir:Path|str|None=None
             ) -> plt.Axes|None:
    """
    Histogram of counts per nucleus for unfiltered merged data
    """
    fig, ax = plt.subplots(1, 1, figsize=(2,2))
    sns.histplot(adata.obs, x='transcript_counts',
                 bins=50, stat='proportion', label='data', ax=ax)
    ax.vlines(x=min_counts, ymin=0, ymax=0.25, colors='red',
              linestyles='dashed', label='threshold')
    ax.set_xlabel('Number of detected transcripts (nucleus)')
    ax.legend()
    ax.set_title('Distribution of detected transcripts per nucleus '
                 'across all (11) tissue sections')
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir.joinpath('Fig_S4_B.pdf'),
                    bbox_inches='tight')
    else:
        return ax
    

def fig_s4c(adata:sc.AnnData, annot_col:str, ambig_val:str='Ambiguous',
             pal:dict[str:str]={'Stable': 'forestgreen', 'Ambiguous': 'grey'},
             save_dir:Path|str|None=None
             ) -> plt.Axes|None:
    """
    UMAP of the annotated data, showing stably- and unstably-predicted cells
    """
    adata_temp = adata.copy()
    adata_temp.obs['filter'] = (adata_temp.obs[annot_col] == ambig_val) \
        .astype('category').cat. \
        rename_categories({True: ambig_val, False: 'Stable'})
    ax = sc.pl.umap(adata, color=['filter'], show=False, palette=pal)
    ax.figure.set_size_inches(2,2)
    ax.set_title('Nuclei filtering by major cell type prediction')
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_dir.joinpath('Fig_S4_C.pdf'),
                          bbox_inches='tight')
    else:
        return ax


def fig_s4d(adata:sc.AnnData, annot_col:str,
             palette_dict:dict[str, str]|None=None,
             save_dir:Path|str|None=None) -> plt.Axes|None:
    """
    UMAP of the final annotated data, showing the predicted classes.
    """
    ax = sc.pl.umap(adata, color=[annot_col], show=False, palette=palette_dict)
    ax.figure.set_size_inches(2,2)
    ax.set_title('PsychAD class in Xenium data')
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_dir.joinpath('Fig_S4_D.pdf'),
                          bbox_inches='tight')
    else:
        return ax


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Process Xenium data and annotate class using scANVI. "
                    "Generate supplementary figures 4b-d.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Directory containing Xenium data folders. "
                             "Each subdirectory is assumed to look like an "
                             "unpacked Xeniu experiment from the associated "
                             "zenodo record.")
    parser.add_argument('-o', '--save_dir', type=str, default=None,
                        help="Directory to save processed data and plots. "
                        "Defaults to input directory if not provided.")
    parser.add_argument('-m', '--metadata', type=str, default=None,
                        help="Path to a CSV file containing sample-level "
                             "metadata. Should have a 'sample_id' column "
                             "matching the Xenium directories.")
    parser.add_argument('-r', '--scanvi_ref', type=str, required=True,
                        help="Path to the h5ad file used as a reference for "
                             "scANVI. This argument is *required*")
    parser.add_argument('-l', '--scanvi_label', type=str, default='class',
                        help="Name of the column in the reference AnnData "
                             "that contains the labels for scANVI training.")
    parser.add_argument('-b', '--scanvi_batch', type=str, default='dataset',
                        help="Name of the column in the reference AnnData "
                             "that contains the batch information.")
    parser.add_argument('--scvi_n_latent', type=int, default=30,
                        help="Number of latent dimensions for scVI.")
    parser.add_argument('--scvi_max_epochs', type=int, default=50,
                        help="Maximum number of epochs for scVI training.")
    parser.add_argument('--scanvi_max_epochs', type=int, default=50,
                        help="Maximum number of epochs for scANVI training.")
    parser.add_argument('--scvi_seed', type=int, default=0,
                        help="Random seed for scVI training. "
                             "Set to ensure reproducibility.")
    parser.add_argument('--min_transcripts', type=int, default=30,
                        help="Minimum number of transcripts for a cell to "
                             "be retained in the analysis.")
    parser.add_argument('--total_counts_norm', type=int, default=1e4,
                        help="Target total counts for normalization. "
                             "Cell transcript counts will be normalized "
                             "to this value before further processing.")
    parser.add_argument('--leiden_res', type=float, default=0.7,
                        help="Resolution parameter for the Leiden clustering "
                             "algorithm. Higher values lead to more clusters."
                             )
    parser.add_argument('-f', '--min_fraction', type=float, default=0.9,
                        help="Minimum fraction of cells in a cluster that "
                             "must be consistently predicted to keep the "
                             "cluster.")
    parser.add_argument('-x', '--shift_x_col', type=str, default='region_number',
                        help="Name of the column in the obs that contains "
                             "the categorical values to shift the x-coordinates.")
    parser.add_argument('-y', '--shift_y_col', type=str, default='slide_id',
                        help="Name of the column in the obs that contains "
                             "the categorical values to shift the y-coordinates.")
    parser.add_argument('--nest_xy', action='store_true',
                        help="If set, will nest the x and y shifts to avoid "
                             "overlap of samples in the UMAP plot.")
    parser.add_argument('-p', '--palette', type=str, default=None,
                        help="Path to a CSV file containing a color palette "
                             "for the cell types. Should have at least two "
                             "columns: 'name' and 'color_hex'. If not "
                             "provided, default colors will be used.")
    return parser


def main() -> int:
    parser = arg_parser()
    args = parser.parse_args()
    # save directories
    if args.save_dir is None:
        args.save_dir = args.input.joinpath("analysis")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = save_dir.joinpath("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    # input data
    xenium_dirs = [d for d in Path(args.input).glob("*")
                   if d.is_dir() and os.path.basename(d) != 'analysis']
    if len(xenium_dirs) == 0:
        print(f"No Xenium directories found in {args.input}.")
        return 1
    # color palette
    pal_dict = pd.read_csv(args.palette).set_index('name'). \
        color_hex.to_dict() \
        if args.palette else None
    # load metadata
    dir_metadata = parse_dirnames(xenium_dirs).set_index('sample_id')
    if args.metadata is not None:
        sample_metadata = pd.read_csv(args.metadata).set_index('sample_id')
        if not sample_metadata.index.equals(dir_metadata.index):
            print("Sample metadata does not match directory metada. "
                  "Please ensure they are consistent.")
            return 1
        sample_metadata = sample_metadata.join(dir_metadata).reset_index()
    else:
        sample_metadata = dir_metadata.reset_index()
    # load Xenium data
    _ = load_xenium(xenium_dirs=xenium_dirs,
                    sample_metadata=sample_metadata,
                    save_dir=save_dir
                    )
    sample_names = sample_metadata['sample_id'].tolist()
    h5ad_fls = [save_dir.joinpath(f"{nm}.h5ad") for nm in sample_names]
    _ = concat_xenium(
        adatas=[sc.read_h5ad(fl) for fl in h5ad_fls],
        sample_names=sample_names,
        shift_x=args.shift_x_col,
        shift_y=args.shift_y_col,
        nest_xy=args.nest_xy,
        save_dir=save_dir
    )
    # transfer labels
    xenium_adata = run_scanvi(
        ref_fl=args.scanvi_ref,
        query_fl=save_dir.joinpath("concatenated.h5ad"),
        label_col=args.scanvi_label,
        out_dir=save_dir.joinpath('scANVI'),
        batch_key=args.scanvi_batch,
        scvi_n_latent=args.scvi_n_latent,
        scvi_max_epochs=args.scvi_max_epochs,
        scanvi_max_epochs=args.scanvi_max_epochs,
        scvi_seed=args.scvi_seed,
        overwrite=True,
        ret_style='query'
    )
    xenium_adata.write_h5ad(save_dir.joinpath("annotated.h5ad"))
    # generate supplementary figure 4 panel B
    fig_s4b(
        adata=xenium_adata,
        min_counts=args.min_transcripts,
        save_dir=plot_dir
    )
    # process and QC the annotated data
    xenium_adata = process_xenium(
        adata=xenium_adata,
        min_counts=args.min_transcripts,
        total_counts=args.total_counts_norm,
        do_cluster=True,
        n_pcs=30,
        leiden_res=args.leiden_res,
        save_dir=None
    )
    # this function also generates supplementary figure 4 panel C
    xenium_adata = filter_by_annot(
        adata=xenium_adata,
        predict_col=f"{args.scanvi_label}_scanvi",  # the label column after transfer
        cluster_col='leiden',
        min_fraction=args.min_fraction,
        save_dir=None,
        plot_dir=plot_dir
    )
    fig_s4d(
        adata=xenium_adata,
        annot_col=f"{args.scanvi_label}_scanvi",
        palette_dict=pal_dict,
        save_dir=plot_dir
    )
    xenium_adata.write_h5ad(save_dir.joinpath("final.h5ad"))
    # clean up intermediate data
    for fl in [*h5ad_fls, save_dir.joinpath("concatenated.h5ad"),
               save_dir.joinpath("annotated.h5ad")]:
        os.remove(fl)
    return 0
    

if __name__ == "__main__":
    sys.exit(main())