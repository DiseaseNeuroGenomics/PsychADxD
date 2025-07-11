
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


# tab10 colors reordered. See here:
# https://stackoverflow.com/questions/64369710/what-are-the-hex-codes-of-matplotlib-tab10-palette
SPECTRAL_TAB10 = [
    '#8c564b',  # brown
    '#d62728',  # red
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#bcbd22',  # olive
    '#1f77b4',  # blue
    '#17becf',  # cyan
    '#9467bd',  # purple
    '#e377c2',  # pink
    '#7f7f7f',  # grey
]


def subset_by_label(adata:sc.AnnData, label_col:str, label_vals:str|list[str],
                    save_path:Path|str|None=None) -> sc.AnnData|None:
    """
    Subset an AnnData object by label values.
    Either save the subsetted AnnData object to a file or return it.
    """
    if isinstance(label_vals, str):
        label_vals = [label_vals]
    adata_subset = adata[adata.obs[label_col].isin(label_vals), :].copy()
    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        adata_subset.write(save_path)
        return None
    else:
        return adata_subset
    

def import_metadata(source_adata:sc.AnnData, source_col:str,
                    target_adata:sc.AnnData, target_col:str|None=None,
                    index_conversion:Callable|None=None,
                    default_val=np.nan, cast_as=None
                   ) -> None:
    """
    Import a metadata column from source to target adata,
    with possible index conversion and type-casting
    """
    source_data = source_adata.obs[source_col].copy()
    if target_col is None:
        target_col = source_col
    if index_conversion is not None:
        source_data.index = [index_conversion(idx)
                             for idx in source_data.index]
    subset_idxs = source_data.index.intersection(target_adata.obs.index)
    source_data = source_data.loc[subset_idxs]
    new_col = pd.Series(default_val, index=target_adata.obs.index)
    new_col.loc[source_data.index] = source_data
    if cast_as is not None:
        new_col = new_col.astype(cast_as)
    target_adata.obs[target_col] = new_col
    

def spatial_plot(adata:sc.AnnData, plot_obs:str,
                 pal:dict|None=None, title:str|None=None,
                 legend_loc:str|None=None,
                 aspect:str|None=None,
                 xticks:list[float]|None=None,
                 xlabels:list[str]|None=None,
                 yticks:list[float]|None=None,
                 ylabels:list[str]|None=None,
                 save_path:Path|str|None=None,
                 **sns_kwargs) -> plt.Axes|None:
    """
    Plot spatial data with seaborn scatterplot
    """
    # get spatial data
    ys, xs = zip(*adata.obsm['spatial'])
    df = pd.DataFrame({
        plot_obs: adata.obs[plot_obs],
        'x': xs,
        'y': ys
    })
    # plot
    ax = sns.scatterplot(data=df, x='x', y='y', hue=plot_obs,
                         palette=pal, **sns_kwargs)
    # make modifications based on input
    if title is None:
        title = plot_obs
    ax.set_title(title)
    if legend_loc is not None:
        sns.move_legend(ax, legend_loc)
    if aspect is not None:
        ax.set_aspect(aspect)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlabels is not None:
        ax.set_xlabel(xlabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylabels is not None:
        ax.set_ylabel(ylabels)
    # save or return
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight')
    else:
        return ax


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer labels to Xenium data using scANVI, then plot "
                    "spatially. Deisgned to run on subclass labels for "
                    "multiple classes, after class is already transferred - "
                    "then generate figures 2b, 2c and supplementary figure "
                    "4i from the paper."
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
        '-l', '--scanvi_label', type=str, default='subclass',
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
    # spatial plots for single sample
    adata_sample = adata_xenium[
        adata_xenium.obs[args.sample_col==args.plot_sample, :]
        ].copy()
    sns_kwargs = dict(s=args.dot_size, linewidth=args.line_width)
    # plot Figure 2B (spatial distribution of class/ parent category)
    pal_dict = pd.read_csv(args.palette).set_index('name'). \
        color_hex.to_dict() \
        if args.palette else None
    figname = f"{args.subset_col}_{args.plot_sample}_spatial"
    if args.subset_col == 'class':
        figname = "Figure_2B"
    spatial_plot(adata_sample, plot_obs=f"{args.subset_col}_scanvi",
                 pal=pal_dict, title=args.subset_col,
                 legend_loc="bottom right", aspect='equal', xticks=[],
                 yticks=[], save_path=plot_dir.joinpath(f"{figname}.pdf"),
                 **sns_kwargs)
    # plot subclass (or other transferred label) spatially for selected sample
    for val in args.subset_vals:
        adata_subset = adata_sample[
            adata_sample.obs[f"{args.subset_col}_scanvi"]==val, :].copy()
        cats = sorted(
            adata_subset.obs[f"{args.scanvi_label}_scanvi"].cat.categories)
        del adata_subset
        adata_plot = adata_sample.copy()
        spectral_pal = {cat: col for cat, col in zip(cats, SPECTRAL_TAB10)}
        figname = f"{args.scanvi_label}_{val}_{args.plot_sample}_spatial"
        if val in ['EN', 'IN'] and args.scanvi_label == 'subclass':
            # add Oligos to the plot, to fill in for WM
            new_obs = adata_plot.obs["subclass_scanvi"].copy()
            new_obs = new_obs.cat.add_categories(['Oligo'])
            new_obs.loc[adata_plot.obs["class_scanvi"] == 'Oligo'] = 'Oligo'
            adata_plot.obs["subclass_scanvi"] = new_obs
            adata_plot.obs["subclass_scanvi"] = \
                adata_plot.obs["subclass_scanvi"].cat.remove_categories(
                    [cat
                     for cat in adata_plot.obs["subclass_scanvi"].cat.categories
                     if cat not in cats+['Oligo']
                    ]
                )
            spectral_pal['Oligo'] = 'lightgrey'
            if val == 'EN':
                figname = "Figure_2C"
            elif val == 'IN':
                figname = "Figure_S4I"
        spatial_plot(adata_plot, plot_obs=f"{args.scanvi_label}_scanvi",
                     pal=spectral_pal, title=f"{args.scanvi_label}\n{val}",
                     legend_loc="bottom right", aspect='equal', xticks=[],
                     yticks=[], save_path=plot_dir.joinpath(f"{figname}.pdf"),
                     **sns_kwargs)
    # remove remaining temp files
    for val in args.subset_vals:
        os.remove(scanvi_dir.joinpath(f"query_{val}_annotated.h5ad"))
    return 0
