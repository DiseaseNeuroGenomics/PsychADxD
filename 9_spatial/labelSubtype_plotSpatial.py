
from pathlib import Path
import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from typing import Callable, Union, List
sys.path.append('./')
from run_scanvi import run_scanvi
from labelSubclass_plotSpatial import subset_by_label, import_metadata, spatial_plot


def aggregate_obs(
    adata: sc.AnnData,
    group_by: str,
    var: Union[str, List[str], Callable[[sc.AnnData], np.ndarray], None] = None,             
    summary: Callable[[np.ndarray], np.ndarray] = None,
    variable_cols: bool = False
    ) -> pd.DataFrame:     
    """     
    Generic function to aggregate data or observables from AnnData in a programmable way.
    Defaults to pseudo-bulk sum of counts when var and summary are None.
    """
    if group_by not in adata.obs.columns:
        raise ValueError(f"'{group_by}' is not a valid column in adata.obs")
    # Extract Data and Column Names
    if var is None:
        raw_data = adata.layers.get('counts', adata.X)
        colnames = adata.var_names
    elif isinstance(var, str):
        raw_data = adata.obs[[var]].to_numpy()
        colnames = [var]
    elif isinstance(var, list):
        raw_data = adata.obs[var].to_numpy()
        colnames = var
    elif callable(var):
        raw_data = var(adata)
        colnames = [f"col_{i}" for i in range(raw_data.shape[1])] \
            if len(raw_data.shape) > 1 else ["var_col"]
    else:
        raise TypeError("Unable to interpret requested variable")

    # Harmonize Dimensions
    if len(raw_data.shape) == 1:
        raw_data = np.expand_dims(raw_data, -1)
    if raw_data.shape[0] != adata.shape[0]:
        raise ValueError(
            f"Incompatible row dimension ({raw_data.shape[0]}) "
            "for the requested observable."
            )

    # Handle Vectorized Standard Sum
    groups = adata.obs[group_by]
    if summary is None:
        # Specialized fast-path for standard pseudo-bulking (summation)
        import scipy.sparse as sp
        if sp.issparse(raw_data):
            # Pivot via matrix multiplication or split-apply for sparse safety
            df = pd.DataFrame.sparse.from_spmatrix(
                raw_data, index=groups, columns=colnames)
        else:
            df = pd.DataFrame(raw_data, index=groups, columns=colnames)
        return df.groupby(level=0, observed=False).sum()

    # Handle Custom Callables (Split-Apply-Combine)
    if sp.issparse(raw_data):
        raw_data = raw_data.toarray()
    df = pd.DataFrame(raw_data, index=groups)
    
    if variable_cols:
        result = df.groupby(level=0, observed=False).apply(
            lambda x: summary(x.to_numpy()))
        # Clean multi-index if summary output generated extra index overhead
        if isinstance(result, pd.DataFrame) and \
                isinstance(result.index, pd.MultiIndex):
            result = result.droplevel(1)
        return result
    # Standard structural mapping (aggregation resulting in a single row per group)
    else:
        return df.groupby(level=0, observed=False).\
            agg(summary).set_axis(colnames, axis=1)


def pseudobulk_concordance(
    adata_ref: sc.AnnData, 
    label_ref: str,
    adata_query: sc.AnnData, 
    label_query: str,
    similarity_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> pd.DataFrame:
    """
    Aggregate query and ref to pseudobulk counts based on provided labels.
    Log-transform, z-score, and calculate similarities using a specified function.
    """
    def pb_zscore(adata, obs_col):
        # Aggregate to clusters x genes DataFrame
        pb_df = aggregate_obs(adata, obs_col)
        # CPM normalization (Depth normalisation per cluster profile)
        cluster_sums = pb_df.sum(axis=1).to_numpy()[:, np.newaxis]
        # Avoid division by zero if a cluster is completely empty
        cluster_sums = np.where(cluster_sums == 0, 1, cluster_sums)
        cpm = (pb_df.to_numpy() / cluster_sums) * 1e6
        log_cpm = np.log2(cpm + 1)
        # Z-scoring across genes (axis=1)
        means = log_cpm.mean(axis=1, keepdims=True)
        stds = log_cpm.std(axis=1, ddof=1, keepdims=True)
        # Avoid division by zero for invariant genes
        stds = np.where(stds == 0, 1, stds)
        
        z_scores = (log_cpm - means) / stds
        return pd.DataFrame(z_scores,
                            index=pb_df.index, columns=pb_df.columns
                            )

    # Compute Z-scores
    pb_ref_z = pb_zscore(adata_ref, label_ref)
    pb_query_z = pb_zscore(adata_query, label_query)
    # Ensure similarity matrix matches row indices cleanly
    similarity_matrix = similarity_func(
        pb_ref_z.to_numpy(), pb_query_z.to_numpy()
        )
    result_df = pd.DataFrame(
        similarity_matrix,
        index=pb_ref_z.index,
        columns=pb_query_z.index
    )
    return match_axes_df(result_df)


def match_axes_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to match the values and order of index and columns
    """
    combined_axis = df.index.union(df.columns)
    return df.reindex(index=combined_axis, columns=combined_axis)


def cross_pearson(x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Pearson cross-correlation between rows of x and y
        """
        lx = x.shape[0]
        return np.corrcoef(x, y)[:lx, lx:]


def expression_heatmap(adata_xenium:sc.AnnData, ref_file:str|Path,
                       obs_col:str, figsize_in:tuple[float, float]=(10,6),
                       title:str|None=None, ref_name:str='reference',
                       save_path:str|Path|None=None
                      ) -> plt.Figure|None:
    """
    Plot heatmap and box-plot for pseudo-bulk expression Pearson correlation
    between Xenium and reference data.
    Note that Xenium and reference should both already be subset to the
    relevant parent group (class, in Figures S4G-H)
    """
    fig, axs = plt.subplots(ncols=2, nrows=1, width_ratios=[2,1],
                            figsize=figsize_in)
    if title:
        fig.suptitle(title)
    
    adata_ref = sc.read_h5ad(ref_file)
    common_vars = adata_ref.var_names.intersection(adata_xenium.var_names)
    concordance_df = pseudobulk_concordance(
        adata_ref[:, common_vars], obs_col,
        adata_xenium[:, common_vars], f'{obs_col}_scanvi',
        similarity_func=cross_pearson
    ).T
    sns.heatmap(concordance_df, vmin=-1, vmax=1, cmap='vlag',
                ax=axs[0], cbar_kws={'shrink': 0.82})
    axs[0].set_aspect('equal')
    axs[0].set_xticks([])
    axs[0].set_xlabel(ref_name)
    # refactor data for boxplot
    flat_series = concordance_df.stack(dropna=False).rename('R')
    concordance_series = flat_series.reset_index() 
    concordance_series['diag'] = np.where(
        concordance_series.iloc[:, 0] == concordance_series.iloc[:, 1], 
        'diagonal', 
        'off-diagonal'
    )
    sns.boxplot(concordance_series, x='diag', y='R', hue='diag',
                legend='full', ax=axs[1])
    axs[1].legend(loc='lower left')
    axs[1].set_xticks([])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer labels to Xenium data using scANVI, then plot "
                    "spatially. Deisgned to run on subtype labels for "
                    "multiple classes. 'class' needs to already be "
                    "transferred, but subclass is not strictly required. "
                    "Also generates supplementary figures 4g, h & j "
                    "from the paper."
        )
    parser.add_argument(
        '-r', '--scanvi_ref', type=str, required=True,
        help='Path to the scANVI reference AnnData object.'
        )
    parser.add_argument(
        '-i', '--xenium_fl', type=str, required=True,
        help='Path to the Xenium AnnData object.'
        )
    parser.add_argument(
        '-o', '--save_dir', type=str, default=None,
        help='Path to save the output files.'
        )
    parser.add_argument(
        '-l', '--scanvi_label', type=str, default='subtype',
        help='Label to transfer using scANVI.'
        )
    parser.add_argument(
        '-b', '--scanvi_batch', type=str, default='dataset',
        help='Batch key for scANVI in reference AnnData.'
        )
    parser.add_argument(
        '--scvi_seed', type=int, default=0,
        help='Random seed for scANVI.'
        )
    parser.add_argument(
        '--scvi_n_latent', type=int, default=30,
        help='Number of latent dimensions for scVI.'
        )
    parser.add_argument(
        '--scvi_max_epochs', type=int, default=50,
        help='Maximum number of epochs for scVI.'
        )
    parser.add_argument(
        '--scanvi_max_epochs', type=int, default=50,
        help='Maximum number of epochs for scANVI.'
        )
    parser.add_argument('--scanvi_overwrite', action='store_true',
                        help="Whether to use saved scANVI model files"
                             "(False; default) if they are found, or "
                             "rerun scANVI and overwrite them")
    parser.add_argument(
        '-s', '--sample_col', type=str, default='sample_id',
        help='Column name for subsetting to a sample t in Xenium AnnData.'
        )
    parser.add_argument(
        '--plot_sample', type=str, default='6799_R3_1495',
        help='Xenium sample to plot.'
        )
    parser.add_argument(
        '-c', '--subset_col', type=str, default='class',
        help='Column name for column by which to subset to generate '
             'references in the scANVI reference AnnData. In the Xenium '
             'data, this is assumed to be the same, with "_scanvi" appended. '
             'As a concrete example - if predicting subclass labels, this '
             'should be "class".'
        )
    parser.add_argument(
        '--subset_vals', type=str, nargs='+', default=['IN', 'EN'],
        help='Values of "subset_col" to subset by, then predict '
             '"scanvi_label".'
        )
    parser.add_argument(
        '--palette', type=str, default=None,
        help='Path to a CSV file with a "name" and "color_hex" column. '
             'Used to set the color palette for the spatial plot.'
        )
    parser.add_argument(
        '--dot_size', type=int, default=10,
        help='Dot size for spatial plot.'
        )
    parser.add_argument(
        '--line_width', type=int, default=0,
        help='Line width for spatial plot.'
        )
    return parser


def main() -> int:
    parser = arg_parser()
    args = parser.parse_args()
    # save directories
    if args.save_dir is None:
        args.save_dir = Path(os.path.dirname(args.xenium_fl))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    scanvi_dir = save_dir.joinpath("scANVI")
    scanvi_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = save_dir.joinpath("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    # transfer labels per subset
    for val in args.subset_vals:
        # create ref and query for subset
        ref_fl = scanvi_dir.joinpath(f"ref_{val}.h5ad")
        subset_by_label(sc.read_h5ad(args.scanvi_ref), args.subset_col, val,
                        save_path=ref_fl
                        )
        query_fl = scanvi_dir.joinpath(f"query_{val}.h5ad")
        subset_by_label(sc.read_h5ad(args.xenium_fl),
                        f"{args.subset_col}_scanvi", val, save_path=query_fl
                        )
        # get labels for query
        adata_query = run_scanvi(
            ref_fl=ref_fl,
            query_fl=query_fl,
            label_col=args.scanvi_label,
            out_dir=scanvi_dir.joinpath(f"{args.subset_col}_{val}__{args.scanvi_label}"),
            batch_key=args.scanvi_batch,
            scvi_n_latent=args.scvi_n_latent,
            scvi_max_epochs=args.scvi_max_epochs,
            scanvi_max_epochs=args.scanvi_max_epochs,
            scvi_seed=args.scvi_seed,
            overwrite=args.scanvi_overwrite,
            ret_style='query'
            )
        # save new data
        adata_query.write_h5ad(
            scanvi_dir.joinpath(f"query_{val}_annotated.h5ad"))

    # import all labels to xenium adata
    adata_xenium = sc.read_h5ad(args.xenium_fl)
    for val in args.subset_vals:
        adata_query = sc.read_h5ad(
            scanvi_dir.joinpath(f"query_{val}_annotated.h5ad"))
        import_metadata(adata_query, f"{args.scanvi_label}_scanvi",
                        adata_xenium, cast_as='category')
    # save to disk
    adata_xenium.write_h5ad(
        save_dir.joinpath(f"xenium_annotated_{args.scanvi_label}.h5ad"))

    # generate plots
    for val in args.subset_vals:
        # expression correlation
        if args.subset_col == 'class' and val == 'EN' and args.scanvi_label=='subtype':
            figname = "Figure_S4G"
        elif args.subset_col == 'class' and val == 'IN' and args.scanvi_label=='subtype':
            figname = "Figure_S4H"
        else:
            figname = f"{args.subset_col}_{val}__{args.scanvi_label}__pseudobulkPearson"
        _ = expression_heatmap(
                sc.read_h5ad(scanvi_dir.joinpath(f"query_{val}_annotated.h5ad")),
                ref_file = scanvi_dir.joinpath(f"ref_{val}.h5ad"),
                obs_col = args.scanvi_label,
                ref_name = 'RADC',  # this is what was used in the paper, but no check for validity is in the code
                save_path = plot_dir.joinpath(f"{figname}.pdf")
            )
        
        # spatial plots for single sample
        adata_sample = adata_xenium[
            adata_xenium.obs[args.sample_col]==args.plot_sample, :
            ].copy()
        sns_kwargs = dict(s=args.dot_size, linewidth=args.line_width)
        # plot subtype (or other transferred label) spatially for selected sample
        plot_col = f"{args.scanvi_label}_scanvi"
        adata_plot = adata_sample.copy()
        adata_plot.obs[plot_col] = \
            adata_plot.obs[plot_col].cat.remove_unused_categories()
        class_mask = adata_plot.obs[f"{args.subset_col}_scanvi"] == val
        allowed_subtypes = adata_plot.obs.loc[class_mask, plot_col].unique()
        cats = sorted([str(c) for c in allowed_subtypes if pd.notna(c)])
        figname = f"{args.scanvi_label}_{val}_{args.plot_sample}_spatial"
        plot_series = adata_plot.obs[plot_col].astype(str)
        if val == 'EN' and args.scanvi_label == 'subtype' \
                and args.subset_col=='class':
            oligo_mask = adata_plot.obs["class_scanvi"] == 'Oligo'
            plot_series[oligo_mask] = 'Oligo'
            cats = cats + ['Oligo']
            figname = "Figure_S4J"
        plot_series[~plot_series.isin(cats)] = 'Other'
        cats = cats + ['Other']
        adata_plot.obs[plot_col] = pd.Categorical(
            plot_series, categories=cats, ordered=False)
        cmap = sns.color_palette("husl", len(cats))
        cdict = {cat: cmap[i] for i, cat in enumerate(cats)}
        cdict['Other'] = "none"
        if 'Oligo' in cdict:
            cdict['Oligo'] = 'lightgrey'
        spatial_plot(adata_plot, plot_obs=plot_col,
                     pal=cdict, title=f"{args.scanvi_label}\n{val}",
                     legend_params=dict(loc="center right",
                                        bbox_to_anchor=(-0.05, 0.5)),
                     aspect='equal', xticks=[], yticks=[],
                     save_path=plot_dir.joinpath(f"{figname}.pdf"),
                     **sns_kwargs)
        
    # remove temp files
    for val in args.subset_vals:
        os.remove(scanvi_dir.joinpath(f"ref_{val}.h5ad"))
        os.remove(scanvi_dir.joinpath(f"query_{val}.h5ad"))
        os.remove(scanvi_dir.joinpath(f"query_{val}_annotated.h5ad"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
