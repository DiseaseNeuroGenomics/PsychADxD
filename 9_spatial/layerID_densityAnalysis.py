import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from skimage.filters import gaussian
import os
from typing import Literal, Callable, Any
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
sys.path.append('./')
from labelSubclass_plotSpatial import spatial_plot


def get_nn_ind_mat(adata:sc.AnnData, n_nbs:int=15) -> np.ndarray:
    """
    Wrapper to get the nearest neighbor index matrix in one line
    """
    n_cells = adata.shape[0]
    try:
        neighbor_indices = adata.obsp['spatial_connectivities'].nonzero()
        n_nbs = len(neighbor_indices[0])//n_cells
    except KeyError:
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=False,
                                n_neighs=n_nbs)
        neighbor_indices = adata.obsp['spatial_connectivities'].nonzero()
    mat_shape = (n_cells, n_nbs)
    # double-check the first index is always the same number n_nbs times - i.e. the row index in our matrix
    assert all(
        neighbor_indices[0].reshape(mat_shape).mean(axis=1) == \
            np.arange(n_cells)
    )
    assert all(
        neighbor_indices[0].reshape(mat_shape).std(axis=1) == 0
    )
    # get matrix where each row 'i' is the 15 NN indexes of the 'i'th cell
    nn_mat = neighbor_indices[1].reshape(mat_shape)
    return nn_mat


def get_ncv(adata:sc.AnnData, annot_obs:str, nn_ind_mat:np.ndarray|None=None,
            n_nbs:int=15) -> pd.DataFrame:
    """
    For each cell, count number of nearest neighbors in each of the categories
    of adata.obs[annot_obs], generated the "neighborhood composition vector"
    (NCV).
    """
    if nn_ind_mat is None:
        nn_ind_mat = get_nn_ind_mat(adata, n_nbs=n_nbs)
    nn_annot = pd.Series(
        [adata.obs[annot_obs].iloc[row] for row in nn_ind_mat],
        index=adata.obs.index
    )
    nn_annot_counts = nn_annot.apply(pd.Series.value_counts)
    return nn_annot_counts


def otsu_helper(series:pd.Series,
                form:Literal['threshold', 'greater', 'lesser']
                ) -> pd.Series|float:
    """
    Helper function to apply Otsu thresholding to a pandas Series.
    If form is 'threshold', the function returns the threshold value.
    If form is 'greater', the function returns a boolean Series where
    values greater than the threshold are True.
    If form is 'lesser', the function returns a boolean Series where
    values less than the threshold are True.
    """
    from skimage.filters import threshold_otsu
    otsu_t = threshold_otsu(series.dropna().to_numpy())
    if form=='threshold':
        return otsu_t
    if form=='greater':
        return (series > otsu_t)
    if form=='lesser':
        return (series < otsu_t)
    print("Unknown form. Returning threshold")
    return otsu_t

def otsu_col(df:pd.DataFrame, col:str, split_by:str|None=None,
             form:Literal['threshold', 'greater', 'lesser']='threshold',
             drop_split:bool=False) -> pd.DataFrame|pd.Series|float:
    """
    Apply Otsu thresholding to a column in a pandas DataFrame. If split_by is
    provided, the threshold is computed for each group in split_by. The result
    is a Series with the same index as df. If drop_split is True, the
    split_by column is dropped from the result.
    """
    if split_by is None:
        return otsu_helper(df[col], form)
    res = df[[col, split_by]].groupby(split_by).apply(otsu_helper, form=form)
    if drop_split:
        res = res.droplevel(split_by)
    return res

def otsu_adata(adata:sc.AnnData, obs_col:str, split_by:str|None=None,
               inplace=False) -> pd.Series|None:
    """
    Apply Otsu thresholding to a column in adata.obs. If split_by is provided,
    the threshold is computed for each group in split_by. The result is a
    boolean column in adata.obs with the same name as obs_col, but with
    "_gtOtsu" appended to it. If inplace is False, the new column is returned
    instead of being added to adata.obs
    """
    new_col = otsu_col(adata.obs, obs_col, split_by, 'greater',
                       drop_split=True)
    if inplace:
        adata.obs[f"{obs_col}_gtOtsu"] = new_col
        return None
    return new_col


# functions used to summarize the values in each pixel
# these are passed to get_pseudo_image as the summary_method argument
def summarize_mean(vals_pixs:list, cell_counts:np.ndarray) -> np.ndarray:
    """Summarize by the mean of each pixel"""
    vals_means = np.array(
        [[np.nanmean(vals_pix) for vals_pix in vals_row]
         for vals_row in vals_pixs
        ]
    )
    return vals_means


def binarize_image(vals_pixs:list, cell_counts:np.ndarray) -> np.ndarray:
    """Summarize by the existence of nonzero signal"""
    vals_sums = np.array(
        [[sum(vals_pix) for vals_pix in vals_row]
         for vals_row in vals_pixs
        ]
    )
    return vals_sums > 0


def binarize_and_fill_holes(vals_pixs:list, cell_counts:np.ndarray
                            ) -> np.ndarray:
    """Binarize the image and fill holes"""
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(binarize_image(vals_pixs, cell_counts))


def get_pseudo_image(adata:sc.AnnData, val_name:str,
                     obsm_key:str='spatial', pix_sz:float=25.,
                     convert_vals:dict|Callable|None=None,
                     summary_method:Callable[[list,np.ndarray], np.ndarray]=summarize_mean,
                     nan_value:Callable|float=0.,
                     return_cell_counts:bool=True
                    ) -> np.ndarray|tuple[np.ndarray, np.ndarray]:
    """
    Generate a pseudo image from the spatial coordinates and values of the
    cells. The image is generated by dividing the spatial coordinates into
    pixels of size pix_sz, then applying the summary_method to the values of
    the cells in each pixel. Pixels with no cells, or with NaN entries for the
    value, are replaced with nan_value. The resulting image is a 2D array with
    the same shape as the spatial coordinates, and the values are the
    summarized values for each pixel. If return_cell_counts is True, the
    function returns a tuple of (pseudo_image, cell_counts), where cell_counts
    is a 2D array with the same shape as the pseudo_image, and the values are
    the number of cells in each pixel.
    """
    # setup
    xs, ys = zip(*adata.obsm[obsm_key])
    xs, ys = xs-np.min(xs), ys-np.min(ys)
    xd, yd = np.ptp(xs), np.ptp(ys)
    nx, ny = int(np.ceil(xd/pix_sz)), int(np.ceil(yd/pix_sz))
    vals_pixs = [[[] for _ in range(ny)] for _ in range(nx)]
    cell_counts = np.zeros(shape=(nx, ny), dtype=int)
    # gather data in relevant pixels
    if val_name in adata.obs.columns:
        cells_vals = adata.obs[val_name]
    elif val_name in adata.var_names:
        cells_vals = adata.X[:, list(adata.var_names).index(val_name)].toarray()
    if isinstance(convert_vals, dict):
        cells_vals = [convert_vals.get(val, np.nan) for val in cells_vals]
    elif callable(convert_vals):
        cells_vals = [convert_vals(val) for val in cells_vals]
    elif convert_vals is not None:
        raise ValueError('Unable to interpret `convert_vals`')
    for icell, val in enumerate(cells_vals):
        x, y = xs[icell], ys[icell]
        xpix, ypix = int(x/pix_sz), int(y/pix_sz)
        cell_counts[xpix, ypix] += 1
        vals_pixs[xpix][ypix].append(val)
    # generate image
    pseudo_image = summary_method(vals_pixs, cell_counts)
    nans = np.isnan(pseudo_image) | (cell_counts == 0)
    if callable(nan_value):
        pseudo_image = nan_value(pseudo_image, nans)
    else:
        pseudo_image[nans] = nan_value
    if return_cell_counts:
        return pseudo_image, cell_counts
    return pseudo_image


def fill_by_nearest(img:np.ndarray, empty_pxs:np.ndarray|None=None
                    ) -> np.ndarray:
    """
    A method for filling missing pixels in an image by the nearest non-missing
    pixel. This is done by using the NearestNeighbors algorithm from sklearn.
    This function can be passed to the `nan_value` argument of
    `get_pseudo_image`.
    """
    from sklearn.neighbors import NearestNeighbors
    if empty_pxs is None:
        empty_pxs = np.isnan(img)
    empty_locs = list(zip(*np.where(empty_pxs)))
    full_locs = list(zip(*np.where(~empty_pxs)))
    nn_model = NearestNeighbors().fit(full_locs)
    nn_idxs = np.squeeze(nn_model.kneighbors(empty_locs, 1, return_distance=False))
    new_img = img.copy()
    for empty_loc, nn_idx in zip(empty_locs, nn_idxs):
        nn_loc = full_locs[nn_idx]
        new_img[*empty_loc] = img[*nn_loc]
    return new_img


def blur_reconstruct(adata:sc.AnnData, val_name:str, pix_sz:float|None=None,
                     image_kwargs:dict=dict(pix_sz=25., nan_value=fill_by_nearest),
                     blur_method:Callable=gaussian,
                     blur_kwargs:dict=dict(sigma=4., mode='nearest')
                    ) -> pd.Series:
    """
    Convenience function to generate a smoothed image of val_name, then
    reconstruct the values at the cell locations.
    image_kwargs are passed to get_pseudo_image, and blur_kwargs are passed
    to the blur_method.
    Note that, with default settings, pix_sz is in microns, while sigma is in
    pixels.
    """
    if pix_sz is None:
        pix_sz = image_kwargs['pix_sz']
    else:
        image_kwargs['pix_sz'] = pix_sz
    pseudo_image, _ = get_pseudo_image(adata, val_name, **image_kwargs)
    blurred_image = blur_method(pseudo_image, **blur_kwargs)
    new_col = pd.Series(np.nan, index=adata.obs.index)
    xs, ys = zip(*adata.obsm['spatial'])
    xs, ys = xs-np.min(xs), ys-np.min(ys)
    for icell in range(adata.shape[0]):
        x, y = xs[icell], ys[icell]
        new_col.iloc[icell] = blurred_image[int(x/pix_sz), int(y/pix_sz)]
    return new_col


def domain_call(adata:sc.AnnData, obs_col:str, pixel_size_init:float=25.,
                pixel_size_final:float|None=None, sigma_microns:float=100.,
                inplace:bool=False
                ) -> pd.Series|None:
    """
    One-step domain calling conveience function. This function generates a
    smoothed image of obs_col, then applies Otsu thresholding to the smoothed
    image to identify the domain. The domain is then further smoothed by re-
    pixelizing at a coarses resolution, then filling holes. The final mask is
    returned as a boolean Series, or added to adata.obs if inplace is True.
    """
    if inplace:
        adata_temp = adata
    else:
        adata_temp = adata.copy()
    # blur image and apply Otsu thresholding
    sigma_pix = sigma_microns/pixel_size_init
    blurred_col = blur_reconstruct(
        adata_temp,
        obs_col,
        pix_sz=pixel_size_init,
        blur_kwargs=dict(sigma=sigma_pix, mode='nearest')
    )
    init_mask = otsu_helper(blurred_col, 'greater')
    temp_mask_key = f'{obs_col}_mask_temp'
    adata_temp.obs[temp_mask_key] = init_mask
    # fill holes in the mask
    if pixel_size_final is None:
        # coarser averaging to identify holes better
        pixel_size_final = pixel_size_init*2
    blurred_filled_mask = blur_reconstruct(
        adata_temp,
        temp_mask_key,
        pix_sz=pixel_size_final,
        image_kwargs=dict(
            convert_vals={True: 1, False: 0},
            summary_method=binarize_and_fill_holes
        ),
        blur_kwargs=dict(sigma=0., mode='nearest')
    )
    final_mask = otsu_helper(blurred_filled_mask, 'greater')
    final_mask_key = f'{obs_col}_mask_final'
    adata_temp.obs[final_mask_key] = final_mask
    if inplace:
        adata_temp.obs.drop(temp_mask_key, axis=1, inplace=True)
        return None
    return final_mask


def assign_cell(obs_row:pd.Series[bool],
                conflict_resolution:Literal['unique', 'sum']='unique',
                unassigned_label:str='Unassigned',
                ) -> str:
    n_assignments = obs_row.sum()
    if n_assignments == 0:
        return unassigned_label
    def layer_name(nm):
        return nm.replace('NNabundance_', '').replace('_count_mask_final', '')
    if n_assignments == 1:
        return layer_name(obs_row.idxmax())
    if conflict_resolution == 'unique':
        return unassigned_label
    elif conflict_resolution == 'sum':
        return '+'.join(
            [layer_name(idx) for idx in obs_row.index if obs_row[idx]]
            )
    else:
        raise ValueError(
            f"Unknown conflict resolution method: {conflict_resolution}"
        )
    

def calculate_area(adata:sc.AnnData, obs_col:str, obs_val:Any, sample_obs:str,
                   pixel_size:float) -> pd.Series:
    """
    Calculate the area of a given domain (defined by the value of obs_col
    being equal to obs_val) in microns squared. The area is calculated
    separately for each sample defined by sample_obs.
    """
    adata_temp = adata.copy()
    adata_temp.obs[f'{obs_col}_is_{obs_val}'] = \
        (adata_temp.obs[obs_col] == obs_val).astype(int)
    samples_areas = {
        sample: np.sum(get_pseudo_image(
            adata=adata_temp[adata_temp.obs[sample_obs] == sample, :],
            val_name=f'{obs_col}_is_{obs_val}',
            pix_sz=pixel_size,
            summary_method=binarize_and_fill_holes,
            return_cell_counts=False
            )) * pixel_size**2
            for sample in adata_temp.obs[sample_obs].unique()
        }
    return pd.Series(samples_areas, name=obs_val).astype(float)


def total_area(adata:sc.AnnData, sample_obs:str, pixel_size:float,
               name:str='total_area') -> pd.Series:
    """
    Calculate the total area of each sample in microns squared, based on
    the number of cell-containing pxiels in the pseudi-image.
    """
    adata_temp = adata.copy()
    samples_areas = {}
    for sample in adata_temp.obs[sample_obs].unique():
        adata_sample = adata_temp[adata_temp.obs[sample_obs] == sample, :]
        _, cell_counts = get_pseudo_image(
            adata=adata_sample,
            val_name=adata_sample.obs.columns[0],
            pix_sz=pixel_size,
            summary_method=binarize_and_fill_holes,
            return_cell_counts=True
        )
        samples_areas[sample] = np.sum(cell_counts) * pixel_size**2
    return pd.Series(samples_areas, name=name).astype(float)


def calculate_densities(adata:sc.AnnData, annot_obs:str, sample_obs:str,
                        layer_obs:str, area_df:pd.DataFrame,
                        select_obs:str|None=None, select_val:str|None=None,
                        save_dir:Path|None=None) -> pd.DataFrame:
    """
    Calculate the density of each annotation in each layer for each sample.
    If save_dir is provided, the raw counts and areas of each region are saved
    to a CSV file. Regardless, the function returns the densitirs of each
    annotation as a DataFrame.
    Note that this assumes that the area_df is indexed by sample and has
    columns for each layer.
    """
    # get counts per sample and layer
    if select_obs is not None and select_val is not None:
        adata = adata[adata.obs[select_obs] == select_val, :].copy()
    df = adata.obs[[sample_obs, layer_obs, annot_obs]].copy()
    df['sample_layer'] = df.apply(
        lambda row: '_'.join([row[sample_obs], row[layer_obs]]), axis=1)
    sample_layer_annot_counts = pd.crosstab(df['sample_layer'], df[annot_obs])
    # reshape area_df to match sample_layer_annot_counts
    areas = pd.Series({
        f"{sample}_{layer}": area_df.loc[sample, layer]
        for sample in area_df.index.tolist()
        for layer in area_df.columns.tolist()
    }, name='area')
    # save raw counts and areas
    if save_dir is not None:
        pd.concat([sample_layer_annot_counts, areas], axis=1).to_csv(
            save_dir.joinpath(f"sample_layer_{annot_obs}_counts.csv"))
    # calculate densities
    sample_layer_annot_dens = sample_layer_annot_counts.div(areas, axis=0)
    return sample_layer_annot_dens


def filter_samples(area_df:pd.DataFrame, min_minfraction_layer:float=0.05,
                   total_col:str='total_area') -> list[str]:
    """
    Filter the samples in area_df based on the lowest fraction of the total
    area represented by a single layer. The function returns a list of samples
    that pass the filter.
    """
    min_fraction = area_df.div(area_df[total_col], axis=0).min(axis=1)
    filtered_samples = min_fraction[min_fraction >= min_minfraction_layer].\
        index.tolist()
    return filtered_samples


def plot_layer_densities(density_df:pd.DataFrame,
                         figsize:tuple[float,float]=(8,8),
                         rescale_density:float=1e6,
                         layer_legend:str='cortical layer',
                         xlabel:str='Density (nuclei/mm^2)',
                         max_x_hline:float=50,
                         text_box_space:float=1,
                         significant_pval:float=0.05,
                         line_height:float=0.5,
                         line_width:float=1.5,
                         star_fontsize:float=8,
                         save_path:Path|None=None
                         ) -> plt.Figure|None:
    """
    Helper function to plot densities across cortical layers and include
    statistical significance (as in Figures S4E-F)
    """
    from scipy.stats import wilcoxon
    import seaborn as sns
    # parse stacked column names
    sample_col, layer_col, annot_col, dens_col = density_df.columns
    # rename layer - this is how it will appear in the legend
    density_df[layer_legend] = density_df[layer_col]
    # original densities are nuclei/um^2; rescaling by 1e6 converts to 1/mm^2
    density_df['density_rescaled'] = density_df[dens_col]*rescale_density
    # make figure
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    ax = sns.boxplot(density_df, y=annot_col, x='density_rescaled',
                     hue=layer_legend, ax=ax, dodge=True)
    ax.set_xlabel(xlabel)
    layer_names = density_df[layer_col].unique()
    cats = density_df[annot_col].unique()
    for y in np.arange(len(cats)-1)+0.5:
        ax.hlines(y=y, xmin=0, xmax=max_x_hline, linestyles='dashed',
                  colors='k', alpha=0.25)
    for y in range(len(cats)):
        group = cats[y]
        for iy in range(len(layer_names)-1):
            # layers should be ordered such that consecutive names are spatially adjacent
            layer1 = layer_names[iy]
            layer2 = layer_names[iy+1]
            test_res = wilcoxon(
                x=density_df.loc[
                    (density_df[annot_col]==group) & (density_df[layer_col]==layer1),
                    layer_col
                    ].sort_index(),
                y=density_df.loc[
                    (density_df[annot_col]==group) & (density_df[layer_col]==layer2),
                    layer_col
                    ].sort_index(),
                alternative='two-sided',
                method='exact'
            )
            if test_res.pvalue < significant_pval:
                y1 = y + (iy-1)*1/3
                y2 = y1 + 1/3
                x = density_df.loc[density_df[annot_col]==group, layer_legend].max() + text_box_space
                w, col = line_height, 'k'
                ax.plot([x, x+w, x+w, x], [y1, y1, y2, y2], lw=line_width, c=col)
                ax.text(x+2*w, 0.5*(y1+y2), "*", ha='center', va='center', color=col, fontsize=star_fontsize)
    ax.set_ylabel("")
    if save_path is None:
        return fig
    fig.savefig(save_path, bbox_inches='tight')


def arg_parser() -> argparse.ArgumentParser:
    """
    Argument parser for the script.
    """
    parser = argparse.ArgumentParser(
        description='Calculate density of cells from each annotation label '
                    'in a given level within a defined set of spatial '
                    'layers/ ROIs. These layers are defined by automatic '
                    'thresholding of the nearest neighbor abundance of '
                    'relevant annotations. '
                    'Also generate Figure S4K.'
        )
    parser.add_argument('-i', '--xenium_fl', type=str,
                        help='Path to the Xenium h5ad file')
    parser.add_argument('-l', '--layer_file', type=str,
                        help='Path to the layer definitions file. This file '
                             'should be a csv file with the first column as '
                             'the layer name and the second column as the '
                             'annotations in that layer. Note that a single '
                             'layer can appear in multiple rows, but it '
                             'should not be the case that the same annotated '
                             'group is associated with multiple layers')
    parser.add_argument('--ignore_layers', type=str, nargs='+',
                        default=['WM'],
                        help='List of layers to ignore in the final analysis.'
                             'These layers are included in the layer file and '
                             'undergo all calculations, but are excluded from '
                             'plotting and statsitics.'
                        )
    parser.add_argument('--sample_col', type=str, default='sample_id',
                        help='Name of the sample column in adata.obs')
    parser.add_argument('--annot_col', type=str, default='subclass_scanvi',
                        help='Name of the column in adata.obs used for layer '
                             'annotation')
    parser.add_argument('--dens_cols', type=str, margs='+',
                        default=['subclass_scanvi', 'subtype_scanvi'],
                        help='Name of columns in adata.obs to calculate '
                             'density for')
    parser.add_argument('--layer_obs', type=str, default='cortical_pseudolayer',
                        help='Name of the layer assignment column in adata.obs')
    parser.add_argument('--select_obs', type=str, default='class_scanvi',
                        help='Name of the column in adata.obs to filter by '
                             'for density calculation (i.e. filter by '
                             '"select_obs", calculate for each of '
                             '"dens_cols"). Typically class, if the analysis '
                             'is at subclass/ subtype level)')
    parser.add_argument('--select_val', type=str, default='IN',
                        help='Value to select from select_obs column')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results. If not provided, '
                             'results are saved in the same directory as the '
                             'input file')
    parser.add_argument('--n_neighbors', type=int, default=30,
                        help='Number of neighbors for NN calculation '
                             '(required for domain identification)')
    parser.add_argument('--pixel_size', type=float, default=25.0,
                        help='Pixel size in microns for pseudo-image generation')
    parser.add_argument('--sigma_microns', type=float, default=100.0,
                        help='Sigma for Gaussian smoothing in microns')
    parser.add_argument('--conflict_resolution', type=str,
                        choices=['unique', 'sum'], default='unique',
                        help='Conflict resolution method for domain calling. '
                             '`unique` means that only one layer can be '
                             'assigned to a cell, while `sum` means that '
                             'multiple layers can be assigned to a cell.')
    parser.add_argument('--min_minfraction_layer', type=float, default=0.05,
                        help='Threshold for sample inclusion. For each '
                             'sample, all layers must occupy at least this '
                             'fraction of the total area occupied by cells.')
    parser.add_argument(
        '--plot_sample', type=str, default='6799_Region3_1495',
        help='Xenium sample to plot spatially.'
        )
    parser.add_argument(
        '--dot_size', type=int, default=10,
        help='Dot size for spatial plot.'
        )
    parser.add_argument(
        '--line_width', type=int, default=0,
        help='Line width for spatial plot.'
        )
    parser.add_argument(
        '--alpha', type=float, default=0.2,
        help='Alpha for spatial plot.'
        )
    return parser


def main() -> int:
    parser = arg_parser()
    args = parser.parse_args()
    # save directories
    if args.save_dir is None:
        args.save_dir = Path(
            os.path.dirname(args.xenium_fl)).joinpath("analysis")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = save_dir.joinpath("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    # get coarse layer definitions
    layer_df = pd.read_csv(args.layer_file, index_col=0)
    layers = set(layer_df.index.tolist())
    # load data
    adata_xenium = sc.read_h5ad(args.xenium_fl)
    samples = adata_xenium.obs[args.sample_col].unique().tolist()
    adata_samples = {
        sample: adata_xenium[
            adata_xenium.obs[args.sample_col] == sample, :].copy()
        for sample in samples
    }
    # identify domains
    for adata_sample in adata_samples.values():
        ncv_df = get_ncv(adata_sample, args.annot_col, n_nbs=args.n_neighbors)
        mask_cols = []
        for layer in layers:
            cols = layer_df.loc[layer].tolist()
            ncv_df[layer] = ncv_df.loc[:, cols].sum(axis=1)
            layer_col = f"{layer}_NN{args.n_neighbors}_count"
            mask_col = f"{layer_col}_mask_final"
            mask_cols.append(mask_col)
            adata_sample.obs[layer_col] = ncv_df[layer]
            domain_call(adata=adata_sample,
                        obs_col=layer_col,
                        pixel_size_init=args.pixel_size,
                        sigma_microns=args.sigma_microns,
                        inplace=True)
        # combine to single assignment
        adata_sample.obs[args.layer_obs] = adata_sample.obs[mask_cols]. \
            apply(assign_cell, axis=1,
                  conflict_resolution=args.conflict_resolution)
    adata_xenium.obs[args.layer_obs] = pd.concat([
        adata_sample.obs[args.layer_obs]
        for adata_sample in adata_samples.values()])
    # save adata
    adata_xenium.write_h5ad(save_dir.joinpath("xenium_annotated_layers.h5ad"))
    # calculate densities
    area_df = pd.concat([
        calculate_area(adata_xenium, args.layer_obs, layer, args.sample_col,
                       2*args.pixel_size)  # factor of 2 to match coarsened pixel size in final domain call (see domain_call)
        for layer in layers
        ], axis=1)
    area_df = pd.concat([
        total_area(adata_xenium, args.sample_col, 2*args.pixel_size),
        area_df
        ], axis=1)
    for dens_col in args.dens_cols:
        density_df = calculate_densities(
            adata_xenium, dens_col, args.sample_col, args.layer_obs,
            area_df.drop('total_area', axis='columns', inplace=False),
            select_obs=args.select_obs, select_val=args.select_val,
            save_dir=save_dir
        )
        # filter table
        filtered_samples = filter_samples(
            area_df, min_minfraction_layer=args.min_minfraction_layer,
            total_col='total_area'
        )
        density_df['sample'] = [idx[:idx.rfind('_')] for idx in density_df.index]
        density_df['layer'] = [idx[idx.rfind('_')+1:] for idx in density_df.index]
        filtered_df = density_df.loc[density_df['sample'].isin(filtered_samples), :].copy()
        filtered_df = filtered_df.loc[~filtered_df['layer'].isin(args.ignore_layers), :].copy()
        # calculate statistics and plot
        stacked_densities = filtered_df.set_index(['sample', 'layer']).stack()
        stacked_densities.name = 'density'
        stacked_densities = stacked_densities.to_frame().reset_index()
        if dens_col=='subclass' and args.select_obs=='class_scanvi' and args.select_val=='IN':
            figname = "Figure_S4E"
        elif dens_col=='subtype' and args.select_obs=='class_scanvi' and args.select_val=='IN':
            figname = "Figure_S4F"
        else:
            figname = f"{dens_col}Densities_select{args.select_obs}_{select_val}_by{args.layer_obs}"
        plot_layer_densities(
            stacked_densities, save_path=plot_dir.joinpath(f"{figname}.pdf")
        )
        # generate Figure S24K
        # note that this portion of the code assumes everything was run as in
        # the paper (i.e. with default settings), but does not check for this
        if dens_col == 'subtype' and args.select_obs=='class_scanvi' and args.select_val=='IN':
            adata_sample = adata_samples[args.plot_sample].copy()
            sns_kwargs = dict(s=args.dot_size, linewidth=args.line_width, alpha=args.alpha)
            # generate color palette
            greys = sns.color_palette('tab20c')[-4:]
            pal = {'WM': greys[0], 'L5-6': greys[1], 'L3-5': greys[2], 'L2-3': greys[3]}
            cmap = sns.color_palette('tab10')
            cats=[cat for cat in adata_sample.obs['scanvi_subtype'].cat.categories
                  if cat.startswith('IN_SST')]
            for i, cat in enumerate(cats):
                pal[cat] = cmap[i]
            # set up figure
            fig, ax = plt.subplots()
            fig.set_size_inches((6,6))
            # plot layers
            adata_sample.obs[layer_col] = adata_sample.obs[layer_col].cat.remove_categories(
                [cat for cat in adata_sample.obs[layer_col].cat.categories
                 if cat not in pal.keys()]
            )
            ax = spatial_plot(adata_sample, layer_col, aspect='equal',
                              yticks=[], xticks=[], title='', pal=pal,
                              ax=ax, **sns_kwargs)
            # plot subtypes
            adata_sample.obs[annot_col] = adata_sample.obs[annot_col].cat.remove_categories(
                [cat for cat in adata_sample.obs[annot_col].cat.categories
                 if cat not in pal.keys()]
            )
            ax = spatial_plot(adata_sample, annot_col, aspect='equal',
                            yticks=[], xticks=[], title=sample, pal=pal,
                            ax=ax, **sns_kwargs)
            # finish up
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            fig.savefig(plot_dir.joinpath("Figure_S4K.pdf"), bbox_inches='tight')

    return 0
        