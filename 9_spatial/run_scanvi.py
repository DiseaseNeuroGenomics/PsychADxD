"""
Command line tool to run label transfer with scANVI.
   
Author: Seon Kinrot <seon.kinrot@mssm.edu>
"""

# imports
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scvi.model import SCVI, SCANVI
# imports for typing
from typing import Callable, Literal
from anndata import AnnData
# params for scVI
ARCHES_PARAMS = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    encode_covariates=True,
    dropout_rate=0.2,
    n_layers=5,
)
# other globals
SCANVI_EMBED_NAME = "X_scanvi"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", 
                        help="file path or glob for reference data",
                        required=True)
    parser.add_argument("-q", "--query",
                        help="file path or glob for query data",
                        required=True)
    parser.add_argument("-l", "--label_col",
                        help="column in obs to transfer labels for",
                        required=True)
    parser.add_argument("-o", "--output",
                        help="output folder path; defaults to query folder")
    parser.add_argument("--predict_col",
                        help="column in obs to transfer predicted labels "
                             "into; default to 'label_col'_scanvi")
    parser.add_argument("--ref_batch_key",
                        help="column in obs to identify batches in "
                             "reference. If a column by the same name "
                             "appears in the query, it willa lso be treated "
                             "as a batch key",
                             default='dataset')
    parser.add_argument("--hvgs",
                        help="can be a file containing hvgs, number of "
                             "highly variable genes to select, or None. "
                             "If this number is smaller than the overlap "
                             "between the query and reference, run hvg "
                             "selection to this number",
                             default=None)
    parser.add_argument("--overwrite",
                        help="flag to overwrite existing models in output "
                             "directory",
                        action="store_true")
    parser.add_argument("--scvi_seed",
                        help="random seed for scvi",
                        type=int, default=0)
    parser.add_argument("--scvi_n_latent",
                        help="number of latent dimensions in scVI embedding",
                        type=int, default=10)
    parser.add_argument("--scvi_max_epochs",
                        help="max epochs to train for scVI model",
                        type=int, default=None)
    parser.add_argument("--scanvi_max_epochs",
                        help="max epochs to train for scANVI model",
                        type=int, default=20)
    parser.add_argument("--scanvi_samples_label",
                        help="number of samples per label in scANVI training",
                        type=int, default=100)
    parser.add_argument("--transfer_max_epochs",
                        help="max epochs to label transfer model",
                        type=int, default=100)
    return parser


def prep_adata(adata:AnnData) -> None:
    """
    Modify AnnData layers to conform to scVI standard
    """
    layer_keys = adata.layers.keys()
    has_counts =  'counts' in layer_keys
    has_raw = hasattr(adata, 'raw')
    if not has_counts and not has_raw:
        adata.layers['counts'] = adata.X.copy()
    if not has_counts and has_raw:
        adata.layers['counts'] = adata.raw.X.copy()
        adata.raw = None
    if has_counts and has_raw:
        adata.raw = None
    return None


def load_merge_adata(path:str, merge_col:str='dataset') -> AnnData:
    """
    Load all files in glob path, marge if multiple
    """
    is_glob = '*' in path
    if is_glob:
        files = glob.glob(path)
    else:
        files = [path]
    assert all(os.path.exists(fl) for fl in files), \
        "Some of the specified files do not exist"

    return sc.concat(
        [sc.read_h5ad(fl) for fl in files],
        label=merge_col
    )


def find_shared_pcg(adata_ref:AnnData, adata_query:AnnData,
                    rbp_col:str | None='robust_protein_coding'
                   ) -> None:
    """
    Find shared protein coding genes for two input AnnData objects,
    based on rbp_col in adata_ref.var. If this column does not exist,
    subset to all shared genes
    """
    all_pcg = adata_ref.var_names
    if rbp_col is not None:
        try:
            all_pcg = all_pcg[adata_ref.var[rbp_col]]
        except KeyError:
            pass
    common_pcg = all_pcg.intersection(adata_query.var_names)
    return common_pcg


def scvi_preprocess(adata_ref:AnnData, adata_query:AnnData,
                    label_col:str, batch_key:str='dataset',
                    rbp_col:str | None='robust_protein_coding',
                    hvgs:str|int|None=None
                    ) -> tuple[AnnData]:
    """
    Helper function to prepare both ref and query data for scVI process.
    Returns a processed copy of both inputs
    """
    # handle layers and raw counts
    prep_adata(adata_ref)
    prep_adata(adata_query)
    # add obs columns
    assert label_col in adata_ref.obs.columns,\
        f"Can't find column {label_col} in reference data"
    if batch_key not in adata_ref.obs.columns:
        adata_ref.obs[batch_key] = 'reference'
    if label_col in adata_query.obs.columns:
        print(f'Warning! Overwriting existing "{label_col}" column in query')
    adata_query.obs[label_col] = 'Unknown'
    if batch_key not in adata_query.obs.columns:
        adata_query.obs[batch_key] = 'query'
    # subset to shared genes
    common_pcg = find_shared_pcg(adata_ref, adata_query, rbp_col)
    if hvgs is not None:
        try:
            n_hvgs = int(hvgs)
            if n_hvgs < len(common_pcg):
                hvgs_df = sc.pp.highly_variable_genes(
                    adata_ref[:, common_pcg].copy(), flavor='cell_ranger',
                    n_top_genes=n_hvgs, batch_key=batch_key, inplace=False
                    )
                hvgs = hvgs_df[hvgs_df.highly_variable].index.to_list()
            else:
                hvgs = common_pcg
        except ValueError:
            if not os.path.exists(hvgs):
                raise ValueError(f"Unable to interpret hvgs - {hvgs}")
            full_hvgs = np.squeeze(pd.read_csv(hvgs).values)
            hvgs = np.intersect1d(full_hvgs, common_pcg)
            if hvgs.size == 0:
                raise ValueError(
                    f"No common genes found in {hvgs} and reference/query"
                )
    else:
        hvgs = common_pcg
    return adata_ref[:, hvgs].copy(), adata_query[:, hvgs].copy()


def load_and_preprocess(ref_path:str, query_path:str,
                        label_col:str, batch_key:str='dataset',
                        rbp_col:str | None='robust_protein_coding',
                        hvgs:str|int|None=None
                       ) -> tuple[AnnData]:
    """
    Convenience wrapper for loading and preprocessing data in one line
    """
    adata_ref = load_merge_adata(ref_path)
    adata_query = load_merge_adata(query_path)
    return scvi_preprocess(
        adata_ref,
        adata_query,
        label_col,
        batch_key,
        rbp_col,
        hvgs
    )


def train_scvi(adata_ref:AnnData, out_dir:str | Path, nametag:str,
               batch_key:str='dataset', overwrite:bool=False, n_latent:int=10,
               max_epochs:int | None=None,
               log_file:str | Path | None=None) -> SCVI:
    SCVI.setup_anndata(adata_ref, batch_key=batch_key, layer="counts")
    if overwrite or len(list(Path(out_dir).glob(f"{nametag}_*.pt"))) == 0:
        scvi_model = SCVI(adata_ref, n_latent=n_latent,
                                     **ARCHES_PARAMS
                                    )
        scvi_model.train(max_epochs=max_epochs)
        scvi_model.save(dir_path=out_dir, prefix=f"{nametag}_", overwrite=True)
        if log_file:
            log_text = (
                f'Trained reference SCVI model and saved with prefix '
                f'{nametag}\nParameters:\n'
                f"'batch_key': '{batch_key}', 'n_latent': {n_latent}, "
                f"'max_epochs': {max_epochs}, {str(ARCHES_PARAMS)[1:-1]}"
                )
            add_to_log(log_file, log_text)
    else:
        scvi_model = SCVI.load(
            out_dir, adata=adata_ref, prefix=f"{nametag}_"
        )
        if log_file:
            add_to_log(
                log_file,
                f'Loaded existing SCVI model with prefix {nametag}'
            )
    return scvi_model


def train_scanvi(scvi_model:SCVI, adata_ref:AnnData, label_col:str,
                 out_dir:str | Path, nametag:str,  overwrite:bool=False,
                 log_file:str | Path | None=None,
                 unlabeled_str:str='Unknown',
                 max_epochs:int=20, n_samples_per_label:int=100) -> SCANVI:
    if overwrite or len(list(Path(out_dir).glob(f"{nametag}_*.pt"))) == 0:
        scanvi_model = SCANVI.from_scvi_model(
            scvi_model, unlabeled_category=unlabeled_str, labels_key=label_col
        )
        scanvi_model.train(max_epochs=max_epochs,
                           n_samples_per_label=n_samples_per_label
                          )
        scanvi_model.save(
            dir_path=out_dir, prefix=f"{nametag}_", overwrite=True
        )
        if log_file:
            log_text = (
                f'Trained reference scANVI model and saved with prefix '
                f'{nametag}\nParameters:\n'
                f'max_epochs={max_epochs}; labels_key={label_col}; '
                f'n_samples_per_label={n_samples_per_label}'
                )
            add_to_log(log_file, log_text)
    else:
        scanvi_model = SCANVI.load(
            out_dir, adata=adata_ref, prefix=f"{nametag}_"
        )
        if log_file:
            add_to_log(
                log_file,
                f'Loaded existing SCANVI model with prefix {nametag}'
            )
    return scanvi_model


def train_label_transfer(
        scanvi_model:SCANVI, adata_query:AnnData, out_dir:str | Path,
        nametag:str,  overwrite:bool=False,
        log_file:str | Path | None=None,
        max_epochs:int=100, weight_decay:float=0.0, check_frequency:int=10
        ) -> SCANVI:
    if overwrite or len(list(Path(out_dir).glob(f"{nametag}_*.pt"))) == 0:
        SCANVI.prepare_query_anndata(adata_query, scanvi_model)
        scanvi_model_q = SCANVI.load_query_data(
            adata_query, scanvi_model
        )
        scanvi_model_q.train(
            max_epochs=max_epochs,
            plan_kwargs=dict(weight_decay=weight_decay),
            check_val_every_n_epoch=check_frequency
        )
        scanvi_model_q.save(
            dir_path=out_dir, prefix=f"{nametag}_", overwrite=True
        )
        if log_file:
            log_text = (
                f'Trained label transfer model and saved with prefix '
                f'{nametag}\nParameters:\n'
                f'max_epochs={max_epochs}; weight_decay={weight_decay}; '
                f'check_val_every_n_epoch={check_frequency}'
                )
            add_to_log(log_file, log_text)
    else:
        scanvi_model_q = SCANVI.load(
            out_dir, adata=adata_query, prefix=f"{nametag}_"
        )
        if log_file:
            add_to_log(
                log_file,
                f'Loaded existing SCANVI transfer model with prefix {nametag}'
            )
    return scanvi_model_q


def default_idx_converter(idx_full:str):
    """
    Helper that strips the batch ID from the index of the full adata,
    to allow adding metadata back to the query
    """
    return idx_full[:idx_full.rfind('-')]


def convert_obs_column(adata:AnnData, target_col:str,
                       batch_col:str='query_ref', batch_val:str | int='query',
                       idx_conversion:Callable[[str], str]=\
                        default_idx_converter,
                      ) -> pd.DataFrame | pd.Series:
    """
    Get target_col from adata, subsetting by batch_col to the value batch_val,
    convert indexes using idx_conversion and return
    """
    orig_idxs = adata.obs.index[adata.obs[batch_col] == batch_val]
    new_col = adata.obs.loc[orig_idxs, target_col].copy()
    new_col.index = [idx_conversion(idx) for idx in orig_idxs]
    return new_col


def get_embedding_df(adata:AnnData, embedding_name:str,
                     var_names:list | str | None=None
                    ) -> pd.DataFrame:
    """
    Extract embedding from obsm slot and convert to dataframe
    """
    embed_df = pd.DataFrame(adata.obsm[embedding_name], index=adata.obs.index)
    if var_names is None:  # default to calling based on embedding_name
        var_names = embedding_name
    if isinstance(var_names, list):  # explicit list of dimension labels
        assert len(var_names) == len(embed_df.columns),\
            "var_names length doesn't match number of dimensions!"
        embed_df.columns = var_names
    elif isinstance(var_names, str):  # treat as prefix
        embed_df.columns = [f'{var_names}_{i+1}' for i in embed_df.columns]
    else:
        raise TypeError(
            f"Invalid value type ({type(var_names)}) passed for var_names"
        )
    return embed_df


def add_to_log(log_path:str | Path | None, new_text:str,
               newline:bool=True, restart_log:bool=False
              ) -> None:
    """
    Helper function to write text to specified log file
    """
    if log_path is None:
        return None
    mode = 'w' if restart_log else 'a'
    with open(log_path, mode) as log_out:
        log_out.write(new_text)
        if newline:
            log_out.write('\n')


def run_scanvi(ref_fl:Path | str, query_fl:Path | str, label_col:str,
              out_dir:str | Path | None=None, predict_col:str=None,
              batch_key:str='dataset', hvgs:str | int | None=None,
              query_ref_key:str='query_ref',
              query_ref_vals:list=["query", "reference"],
              log_file:str | Path | None=None, scvi_seed:int|None=0,
              scvi_n_latent:int=30, scvi_max_epochs:int | None=50,
              scanvi_max_epochs:int | None=50,
              scanvi_samples_label:int | None=100,
              transfer_max_epochs:int | None=100,  overwrite:bool=False,
              ret_style:Literal['full', 'query', 'predictions', None]=None
             ) -> AnnData | pd.DataFrame | None:
    """
    Main function to run the scANVI pipeline for label transfer.
    As opposed to `main`, this function is designed to be importable and
    programmable within python.
    """
    if scvi_seed is None:
        scvi_seed = np.random.randint(0, 1000000)
    scvi.settings.seed = scvi_seed
    if predict_col is None:
        predict_col = f"{label_col}_scanvi"
    if out_dir is None:
        out_dir = Path(os.path.dirname(query_fl))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # load data
    adata_ref, adata_query = load_and_preprocess(
        ref_path=ref_fl, query_path=query_fl,
        label_col=label_col, batch_key=batch_key,
        hvgs=hvgs
    )
    adata_full = adata_query.concatenate(adata_ref, batch_key=query_ref_key)
    adata_full.obs[query_ref_key] = \
        adata_full.obs[query_ref_key].cat.rename_categories(query_ref_vals)
    add_to_log(
        log_file,
        f'Loaded data successfully! Batch key in obs is: {batch_key}'
    )
    # train scvi model
    scvi_model = train_scvi(
        adata_ref, out_dir=out_dir, nametag="scviRef",
        batch_key=batch_key, overwrite=overwrite,
        n_latent=scvi_n_latent, max_epochs=scvi_max_epochs,
        log_file=log_file
    )
    # train scANVI model
    scanvi_model = train_scanvi(
        scvi_model, adata_ref, label_col=label_col,
        out_dir=out_dir, nametag="scANVI", overwrite=overwrite,
        log_file=log_file,
        unlabeled_str='Unknown', max_epochs=scanvi_max_epochs,
        n_samples_per_label=scanvi_samples_label
    )
    # train label transfer model
    scanvi_model_q = train_label_transfer(
        scanvi_model, adata_query=adata_query,
        out_dir=out_dir, nametag="labelTransfer", overwrite=overwrite,
        log_file=log_file, max_epochs=transfer_max_epochs
    )
    # transfer the labels
    adata_full.obs[predict_col] = scanvi_model_q.predict(adata_full)
    adata_query.obs[predict_col] = convert_obs_column(
        adata=adata_full, target_col=predict_col,
        batch_col=query_ref_key, batch_val=query_ref_vals[0]
        )
    adata_full.obsm[SCANVI_EMBED_NAME] = \
        scanvi_model_q.get_latent_representation(adata_full)
    # save predictions and embeddings
    predictions_df = adata_full.obs[
        [query_ref_key, batch_key, label_col, predict_col]
    ]
    predictions_df = pd.concat(
        [predictions_df,
         get_embedding_df(adata_full, SCANVI_EMBED_NAME)
        ],
        axis=1
    )
    predictions_df.to_csv(
        f"{out_dir}/{label_col}_scanviPredictions_complete.csv"
    )
    add_to_log(log_file, "Successfully completed full run! Shutting down")
    # if required, return the full adata, query adata, or predictions df
    if ret_style is None:
        return None
    if ret_style=='full':
        return adata_full
    if ret_style == 'query':
        return adata_query
    if ret_style == 'predictions':
        return predictions_df


def main() -> int:
    # handle input args
    parser = arg_parser()
    args = parser.parse_args()
    today = datetime.today().strftime("%Y%m%d")
    label_col = args.label_col
    predict_col = f'{label_col}_scanvi' if args.predict_col is None \
        else args.predict_col
    out_dir = Path(os.path.dirname(args.query)) if args.output is None \
        else Path(args.output)
    assert os.path.isdir(out_dir.parent.absolute())
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    log_file = f"{out_dir}/{today}_log.txt"
    log_text = '\n'.join([
        today,
        f'scANVI prediction for reference files: {args.ref}',
        f'query files: {args.query}',
        f'label being predicted: {args.label_col}',
        f'Saving data to: {out_dir}'
    ])
    # initiate log
    add_to_log(log_file, log_text, restart_log=True)
    # run label transfer
    run_scanvi(
        ref_fl=args.ref, query_fl=args.query, label_col=label_col,
        predict_col=predict_col, out_dir=out_dir,
        batch_key=args.ref_batch_key, hvgs=args.hvgs,
        log_file=log_file, scvi_seed=args.scvi_seed,
        scvi_n_latent=args.scvi_n_latent, scvi_max_epochs=args.scvi_max_epochs,
        scanvi_max_epochs=args.scanvi_max_epochs,
        scanvi_samples_label=args.scanvi_samples_label,
        transfer_max_epochs=args.transfer_max_epochs,
        overwrite=args.overwrite
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())