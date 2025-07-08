import os, sys
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.optimize as optimize
from sklearn.cross_decomposition import PLSRegression
import gc


class PLS:
    """Class to perform PLS regression to predict Braak (and possibly dementia)
    from raw spike counts.
    Used for Supplementary Figure 15
    """
    def __init__(self, meta_fn, data_path, splits_fn, restrictions):

        self.remove_sex_chrom = True
        self.protein_coding_only = True
        self.include_analysis = True
        self.top_k_genes = 10_000
        self.splits = pickle.load(open(splits_fn, "rb"))

        self.data_path = data_path
        self.metadata = pickle.load(open(meta_fn, "rb"))
        self._restrict_samples(restrictions)
        self._get_gene_index()

    def _restrict_samples(self, restrictions):

        self.cell_idx = np.arange(len(self.metadata["obs"]["class"]))

        cond = np.zeros(len(self.metadata["obs"]["class"]), dtype=np.uint8)
        cond[self.cell_idx] = 1

        if self.include_analysis:
            if "include_analysis" in self.metadata["obs"].keys():
                cond *= (self.metadata["obs"]["include_analysis"] > 0)
                print("only analyzing include_analysis = True")

        idx = np.where(np.array(self.metadata["obs"]["SubID"]) == "PM-MS_55245")[0]
        cond[idx] = 0
        print(f"Removing SubID PM-MS_55245, number of samples removed {len(idx)}")

        if restrictions is not None:
            for k, v in restrictions.items():
                if isinstance(v, list):
                    cond *= np.sum(np.stack([np.array(self.metadata["obs"][k]) == v1 for v1 in v]), axis=0).astype(
                        np.uint8)
                else:
                    cond *= np.array(self.metadata["obs"][k]) == v

        self.cell_idx = np.where(cond)[0]
        self.n_samples = len(self.cell_idx)

        for k in self.metadata["obs"].keys():
            self.metadata["obs"][k] = np.array(self.metadata["obs"][k])[self.cell_idx]

    def _get_gene_index(self):

        self.n_genes_full = len(self.metadata["var"]["gene_name"])
        cond = self.metadata["var"]['percent_cells'] >= 0.0

        if self.remove_sex_chrom:
            cond *= self.metadata["var"]['gene_chrom'] != "X"
            cond *= self.metadata["var"]['gene_chrom'] != "Y"
            self.metadata["var"]['percent_cells'][~cond] = 0.0

        if self.protein_coding_only:
            cond *= self.metadata["var"]['protein_coding']
            self.metadata["var"]['percent_cells'][~cond] = 0.0

        if self.top_k_genes is not None:
            th = np.sort(self.metadata["var"]['percent_cells'])[-self.top_k_genes]
            cond *= self.metadata["var"]['percent_cells'] > th
            print(f"Top {self.top_k_genes} genes selected; threshold = {th:1.4f}")

        self.gene_idx = np.where(cond)[0]

        self.n_genes = len(self.gene_idx)
        self.gene_names = self.metadata["var"]["gene_name"][self.gene_idx]  # needed for pathway networks
        print(f"Sub-sampling genes. Number of genes is now {self.n_genes}")

    def _normalize_data(self, x):

        x = np.float32(x)
        x = 10_000 * x / np.sum(x)
        x = np.log1p(x)
        return x

    def load_data(self):

        self.data = np.zeros((len(self.cell_idx), len(self.gene_idx)), dtype=np.float32)

        for n, i in enumerate(self.cell_idx):
            data = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_full,), offset=i * self.n_genes_full,
            ).astype(np.float32)
            data = self._normalize_data(data)
            self.data[n, :] = data[self.gene_idx]

    def cross_val_pls(self, n_components=10):

        pls = PLSRegression(n_components=n_components)
        y = np.hstack((
            np.reshape(self.metadata["obs"]["BRAAK_AD"], (-1, 1)),
            # np.reshape(self.metadata["obs"]["Dementia"], (-1, 1)),
            # np.reshape(adata.obs["CERAD"].values, (-1, 1)),
        ))
        N = y.shape[0]
        y_valid = np.sum(y, axis=1) > -1

        self.y_hat = np.zeros(y.shape)
        self.results = {
            "cell_idx": self.cell_idx,
            "pred_BRAAK_AD": np.zeros(N),
            "pred_Dementia": np.zeros(N),
            "BRAAK_AD": y[:, 0],
            "Dementia": np.reshape(self.metadata["obs"]["Dementia"], (-1, 1)),
        }

        for split_num in range(0, 20):
            print(f"Split number {split_num}")
            train_idx_full = set(self.splits[split_num]["train_idx"])
            test_idx_full = set(self.splits[split_num]["test_idx"])

            train_idx = np.where(
                np.isin(self.cell_idx, list(train_idx_full)) * y_valid
            )[0]
            test_idx = np.where(np.isin(self.cell_idx, list(test_idx_full)))[0]

            pls.fit(self.data[train_idx, :], y[train_idx, :])

            y_hat_split = pls.predict(self.data[test_idx, :])
            self.results["pred_BRAAK_AD"][test_idx] = y_hat_split[:, 0]
            # self.results["pred_Dementia"][test_idx] = y_hat_split[:, 1]

    def cross_val_pls_pd(self, n_components=10):

        k = "path_braak_lb_condensed_v3"

        pls = PLSRegression(n_components=10)
        y = np.hstack((
            np.reshape(self.metadata["obs"][k], (-1, 1)),
        ))
        N = y.shape[0]
        y_valid = np.sum(y, axis=1) > -99

        self.y_hat = np.zeros(y.shape)
        self.results = {
            "cell_idx": self.cell_idx,
            f"pred_{k}": np.zeros(N),
            k: y[:, 0],
        }

        for split_num in range(0, 20):
            print(f"Split number {split_num}")
            train_idx_full = set(self.splits[split_num]["train_idx"])
            test_idx_full = set(self.splits[split_num]["test_idx"])

            train_idx = np.where(
                np.isin(self.cell_idx, list(train_idx_full)) * y_valid
            )[0]
            test_idx = np.where(np.isin(self.cell_idx, list(test_idx_full)))[0]

            pls.fit(self.data[train_idx, :], y[train_idx, :])

            y_hat_split = pls.predict(self.data[test_idx, :])
            self.results[f"pred_{k}"][test_idx] = y_hat_split[:, 0]


class ProcessData:

    def __init__(self, meta_fn, data_path):

        self.meta = pickle.load(open(meta_fn, "rb"))
        self.data_path = data_path
        self.n_genes = len(self.meta["var"]["gene_name"])
        self.gene_names = self.meta["var"]["gene_name"]

        self.obs_list = ["pred_BRAAK_AD", "pred_Dementia"]
        self.obs_from_metadata = [
            "BRAAK_AD", "CERAD", "Dementia", "class", "subclass", "subtype", "SubID",
            "include_analysis", "Age", "Sex", "Brain_bank", "barcode", "MCI", "Dementia_graded",
            "CDRScore",
        ]
        self.donor_stats = [
            "Sex", "Age", "Brain_bank", "Dementia", "BRAAK_AD", "pred_BRAAK_AD",
            "pred_Dementia", "CERAD", "CDRScore",
        ]


    def create_data(self, results_fn):

        z = pickle.load(open(results_fn, "rb"))
        adata = self._create_base_anndata(z)
        adata = self._add_unstructured_data(adata)
        adata = self._add_donor_stats(adata)
        return adata

    def _create_base_anndata(self, z):

        k = self.obs_list[0]
        n = z[k].shape[0]
        self.gene_idx = np.where(self.meta["var"]["protein_coding"])[0]
        latent = np.zeros((n, 1), dtype=np.float32)  # dummy variable to create AnnData

        a = ad.AnnData(latent)
        for m, k in enumerate(self.obs_list):
            a.obs[k] = z[k]

        a.obs["cell_idx"] = z["cell_idx"]
        a = self._add_obs_from_metadata(a)
        return a

    def _add_obs_from_metadata(self, adata):

        for k in self.obs_from_metadata:
            x = []
            for n in adata.obs["cell_idx"]:
                x.append(self.meta["obs"][k][n])
            adata.obs[k] = x

        return adata

    def _add_unstructured_data(self, adata):

        for k in self.obs_list:
            adata.uns[k] = []

        adata.uns["donors"] = []
        for subid in np.unique(adata.obs["SubID"]):
            adata.uns["donors"].append(subid)
            idx = np.where(np.array(adata.obs["SubID"].values) == subid)[0][0]
            for k in self.obs_list:
                adata.uns[k].append(adata.obs[k].values[idx])

        return adata

    def _normalize_data(self, x):
        x = np.float32(x)
        x = 10_000 * x / np.sum(x)
        x = np.log1p(x)
        return x

    def _add_donor_stats(self, adata):

        n_donors = len(adata.uns["donors"])

        adata.uns[f"donor_gene_means"] = np.zeros((n_donors, len(self.gene_idx)), dtype=np.float32)
        adata.uns[f"donor_cell_count"] = np.zeros((n_donors,), dtype=np.float32)

        for k in self.donor_stats:
            adata.uns[f"donor_{k}"] = np.zeros((n_donors,), dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            a = adata[adata.obs["SubID"] == subid]
            count = 1e-6  # to prevent 1 / 0 errors
            go_scores = {}

            for n, i in enumerate(a.obs["cell_idx"]):
                data = np.memmap(
                    self.data_path, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)
                data = self._normalize_data(data)
                adata.uns[f"donor_gene_means"][m, :] += data[self.gene_idx]
                count += 1

                for k in self.donor_stats:
                    if "pred_" in k:
                        if np.isnan(a.obs[f"{k}"][n]):
                            print(k, subid, a.obs[f"{k}"][n])
                            continue
                        else:
                            adata.uns[f"donor_{k}"][m] += a.obs[f"{k}"][n]
                    elif "Sex" in k:
                        adata.uns[f"donor_{k}"][m] = float(a.obs[f"{k}"][n] == "Male")
                    elif "Brain_bank" in k:
                        adata.uns[f"donor_{k}"][m] = float(a.obs[f"{k}"][n] == "MSSM")
                    else:
                        adata.uns[f"donor_{k}"][m] = a.obs[f"{k}"][n]

            adata.uns[f"donor_cell_count"][m] = count
            adata.uns[f"donor_gene_means"][m, :] /= count

            for k in self.donor_stats:
                if "pred_" in k:
                    adata.uns[f"donor_{k}"][m] /= count

        return adata