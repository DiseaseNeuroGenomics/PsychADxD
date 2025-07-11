
from pathlib import Path
import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from typing import Callable
sys.path.append('./')
from run_scanvi import run_scanvi
from labelSubclass_plotSpatial import subset_by_label, import_metadata, spatial_plot


def aggregate_fixed_cols(raw_data:np.ndarray,
                         groups:pd.Series | pd.DataFrame | np.ndarray | list,
                         colnames:list | None,
                         summary:Callable[[np.ndarray], np.ndarray]
                        ) -> pd.DataFrame:
    """
    Helper for aggregate_obs, aggregating data when number of columns is fixed
    """
    redo_cols = colnames is None
    if colnames is None:
        colnames = range(raw_data.shape[1])
    agg = pd.DataFrame(np.nan, index=np.unique(groups), columns=colnames)
    for group in np.unique(groups):
        idxs = groups == group
        new_vals = summary(raw_data[idxs, :])
        agg.loc[group, :] = new_vals
    if redo_cols and isinstance(new_vals, pd.DataFrame):
        agg.columns = new_vals.columns
    return agg


def aggregate_var_cols(raw_data:np.ndarray,
                       groups:pd.Series | pd.DataFrame | np.ndarray | list,
                       colnames:list | None,
                       summary:Callable[[np.ndarray], np.ndarray]
                      ) -> pd.DataFrame:
    """
    Helper for aggregate_obs, aggregating data when number of columns varies
    """
    agg = pd.DataFrame()
    for group in np.unique(groups):
        idxs = groups == group
        agg = pd.concat([agg, summary(raw_data[idxs, :])])
    agg.index = np.unique(groups)
    if colnames is not None:
        agg.columns = colnames
    return agg


def aggregate_obs(adata:sc.AnnData, group_by:str,
                  var:str|list[str]|Callable[[AnnData], np.ndarray]|None=None,
                  summary:Callable[[np.ndarray], np.ndarray]|None=None,
                  variable_cols:bool=False
                 ) -> pd.DataFrame:
    """
    Generic function to aggregate data or observables from AnnData, in a
    programmable way (customizable underlying data, grouping parameter and
    summary method).
    Default to pseudo-bulk of counts with var and summary as None.
    -------------
    Inputs:
    adata - an AnnData object to summarize information from
    group_by - a column in obs to group by, then summarize observation for
        each group separately
    var - define which observable to aggregate. If None, defaults to raw
        counts. Can also be defined as a string or list of strings
        (interpreted as a column in obs), or a callable taking a single
        AnnData object and returning the observable to be aggregated
        (should have the same number of rows as the AnnData object).
        Note that, in the case var is a callable, it must satisfy the
        following conditions:
        * it should return an array-like object with the same number of rows
          as adata
        * the returned object should be capable of direct slicing using []
          notation
        * column names will default to a numerical range
    summary - define how to collapse the observable from var row-wise. If left
        as None, defaults to a simple sum. Can also be a custom callable
        function that takes a numpy array or DataFrame and returns a
        vector-like object
    variable_cols - whether the number of columns is fixed and the same as the
        number of columns in the observed data. Possible depending on summary
    
    Returns:
    a pd.DataFrame object with aggregated and summarized data
    -------------
    Example use 1:
    Get a Dataframe of size n_clusters x n_genes, with pseudo-bulk counts
        `pb = aggregate_obs(adata, group_by='clusters')`

    Example use 2:
    An example to demonstarte when variable_cols is useful.
    Get a DataFrame of size n_clusters x n_samples, with the number of cells
    from each cluster and sample combination. Notice the variable used for
    aggregation is one-dimensional (sample_id), but the resultant table is 2D
        ```
        def count_samples(samples):
            # count unique appearance and make horizontal vector
            vals, cts = np.unique(samples, return_counts=True)
            return pd.Dataframe(cts, index=['counts'], columns=vals)
        
        cluster_sample_counts = aggregate_obs(
            adata, group_by='clusters', var='sample_id',
            summary=count_samples, variable_cols=True
        )
        ```
    """
    if var is None:
        # consider replacing with a generic function to get raw counts
        try:
            raw_data = adata.layers['counts'].copy()
        except KeyError:
            print('Warning: no counts layer; using X')
            raw_data = adata.X.copy()
        colnames = adata.var_names
    elif isinstance(var, str):
        raw_data = adata.obs[var].to_numpy()
        colnames = [var]
    elif isinstance(var, list):
        raw_data = adata.obs.loc[:, var].to_numpy()
        colnames = var
    elif callable(var):
        raw_data = var(adata)
        colnames = None
    else:
        raise TypeError("Unable to interpret requested variable")
    if len(raw_data.shape) == 1:
        raw_data = np.expand_dims(raw_data, -1)
    assert raw_data.shape[0] == adata.shape[0],\
        f"Incompatible dimensions for observable {var}"
    if summary is None:
        from functools import partial
        summary = partial(np.sum, axis=0)

    agg_func = aggregate_var_cols if variable_cols else aggregate_fixed_cols
    return agg_func(
        raw_data, adata.obs[group_by], colnames, summary
    )


def pseudobulk_concordance(adata_ref:sc.AnnData, label_ref:str,
                           adata_query:sc.AnnData, label_query:str,
                           similarity_func:Callable[
                               [np.ndarray, np.ndarray], np.ndarray
                            ]
                          ) -> pd.DataFrame:
    """
    Aggregate query and ref to pseudobulk counts, based on provided labels.
    Then log1p-transform, get z-scores, calculate similarities using specified
    function, and return as a cleaned up DataFrame
    """
    def pb_zscore(adata, obs_col):
        return aggregate_obs(adata, obs_col).\
            apply(lambda x: np.log2(1e6*x/x.sum() + 1), axis=0).\
            apply(lambda y: (y-y.mean())/y.std(), axis=1)
    pb_ref_z = pb_zscore(adata_ref, label_ref)
    pb_query_z = pb_zscore(adata_query, label_query)
    return match_axes_df(
        pd.DataFrame(
            similarity_func(pb_ref_z, pb_query_z),
            index=pb_ref_z.index,
            columns=pb_query_z.index
        )
    )

def match_axes_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to match the values and order of index and columns
    """
    values = set(list(df.index) + list(df.columns))
    for val in values:
        if val not in df.index:
            df.loc[val, :] = np.nan
        if val not in df.columns:
            df[val] = np.nan
    df = df[df.index]
    return df


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
    fig, axs = plt.subplots(ncols=2, nrows=1, width_ratios=[2,1])
    fig.set_size_inches(figsize_in)
    if title:
        fig.suptitle(title)
    
    adata_ref = sc.read_h5ad(ref_file)
    common_vars = adata_ref.var_names.intersection(adata_xenium.var_names)
    concordance_df = pseudobulk_concordance(
        adata_ref[:, common_vars].copy(), obs_col,
        adata_xenium[:, common_vars].copy(), f'{obs_col}_scanvi',
        similarity_func=cross_pearson
    )
    axs[0] = sns.heatmap(concordance_df, vmin=-1, vmax=1, cmap='vlag',
                         ax=axs[0], cbar_kws={'shrink': 0.82})
    axs[0].set_aspect('equal')
    axs[0].set_xticks([])
    axs[0].set_xlabel(ref_name)
    concordance_series = pd.Series({
        f'{row}__{col}':concordance_df.loc[row,col]
        for row in concordance_df.index for col in concordance_df.columns
    }, name='R')
    concordance_series = pd.concat(
        [pd.Series(
            {idx: 'diagonal' if idx.split('__')[0] == idx.split('__')[1]
                   else 'off-diagonal'
                   for idx in concordance_series.index
            },
            name='diag'),
         concordance_series
        ],
        axis=1
    )
    axs[1] = sns.boxplot(concordance_series, x='diag', y='R', hue='diag',
                         legend='full', ax=axs[1])
    axs[1].legend(loc='lower left')
    axs[1].set_xticks([])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
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
        '--scanvi_seed', type=int, default=0,
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
    parser.add_argument(
        '-s', '--sample_col', type=str, default='sample_id',
        help='Column name for subsetting to a sample t in Xenium AnnData.'
        )
    parser.add_argument(
        '--plot_sample', type=str, default='6799_Region3_1495',
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
            overwrite=True,
            ret_style='query'
            )
        # save new data and remove run-specific query and reference
        adata_query.write_h5ad(
            scanvi_dir.joinpath(f"query_{val}_annotated.h5ad"))
        os.remove(ref_fl)
        os.remove(query_fl)
    # import all labels to xenium adata
    adata_xenium = sc.read_h5ad(args.xenium_fl)
    for val in args.subset_vals:
        adata_query = sc.read_h5ad(
            scanvi_dir.joinpath(f"query_{val}_annotated.h5ad"))
        import_metadata(adata_query, f"{args.scanvi_label}_scanvi",
                        adata_xenium, cast_as='category')
    adata_xenium.write_h5ad(
        save_dir.joinpath(f"xenium_annotated_{args.scanvi_label}.h5ad"))
    # expression correlation plots
    for val in args.subset_vals:
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
        adata_xenium.obs[args.sample_col==args.plot_sample, :]
        ].copy()
    sns_kwargs = dict(s=args.dot_size, linewidth=args.line_width)
    # plot subtype (or other transferred label) spatially for selected sample
    for val in args.subset_vals:
        adata_subset = adata_sample[
            adata_sample.obs[f"{args.subset_col}_scanvi"]==val, :].copy()
        cats = sorted(
            adata_subset.obs[f"{args.scanvi_label}_scanvi"].cat.categories)
        del adata_subset
        adata_plot = adata_sample.copy()
        cmap = sns.color_palette("husl", len(cats))
        cdict = {cat: cmap[i] for i, cat in enumerate(cats)}
        figname = f"{args.scanvi_label}_{val}_{args.plot_sample}_spatial"
        if val == 'EN' and args.scanvi_label == 'subtype':
            # add Oligos to the plot, to fill in for WM
            new_obs = adata_plot.obs["subtype_scanvi"].copy()
            new_obs = new_obs.cat.add_categories(['Oligo'])
            new_obs.loc[adata_plot.obs["class_scanvi"] == 'Oligo'] = 'Oligo'
            adata_plot.obs["subtype_scanvi"] = new_obs
            adata_plot.obs["subtype_scanvi"] = \
                adata_plot.obs["subtype_scanvi"].cat.remove_categories(
                    [cat
                     for cat in adata_plot.obs["subtype_scanvi"].cat.categories
                     if cat not in cats+['Oligo']
                    ]
                )
            cdict['Oligo'] = 'lightgrey'
            figname = "Figure_S4J"
        spatial_plot(adata_plot, plot_obs=f"{args.scanvi_label}_scanvi",
                     pal=cdict, title=f"{args.scanvi_label}\n{val}",
                     legend_loc="bottom right", aspect='equal', xticks=[],
                     yticks=[], save_path=plot_dir.joinpath(f"{figname}.pdf"),
                     **sns_kwargs)
    # remove remaining temp files
    for val in args.subset_vals:
        os.remove(scanvi_dir.joinpath(f"query_{val}_annotated.h5ad"))
    return 0
