from typing import Optional, List
import pickle
import copy
import pandas as pd
import numpy as np
import anndata as ad
import scipy.stats as stats
from sklearn.decomposition import PCA



class ModelResults:

    def __init__(
        self,
        data_fn: str,
        meta_fn: str,
        obs_list: List = ["pred_BRAAK_AD", "pred_Dementia"],
        gene_pathway_fn: Optional[str] = None,  # dict of GO BP pathways
        gene_count_prior: Optional[float] = None,
        include_analysis_only: bool = True,
        load_gene_count: bool = False,
        normalize_gene_counts: bool = False,
        log_gene_counts: bool = False,
        add_gene_scores: bool = False,
    ):

        self.data_fn = data_fn
        self.meta = pickle.load(open(meta_fn, "rb"))
        self.gene_pathways = pickle.load(open(gene_pathway_fn, "rb")) if gene_pathway_fn is not None else None
        self.convert_apoe()  # currently not used in the analysis

        self.obs_list = obs_list
        self.n_genes = len(self.meta["var"]["gene_name"])
        self.gene_names = self.meta["var"]["gene_name"]

        self.gene_count_prior = gene_count_prior
        self.include_analysis_only = include_analysis_only
        self.load_gene_count = load_gene_count
        self.normalize_gene_counts = normalize_gene_counts
        self.log_gene_counts = log_gene_counts
        self.add_gene_scores = add_gene_scores

        self.extra_obs = []
        self.obs_from_metadata = [
            "BRAAK_AD", "Dementia", "class", "subclass", "subtype", "SubID",
            "include_analysis", "Age", "Sex", "Brain_bank", "barcode",
        ]

        self.donor_stats = [
            "Sex", "Age", "Brain_bank", "Dementia", "BRAAK_AD", "pred_BRAAK_AD", "pred_Dementia",
        ]

    def create_data(self, model_fns, model_average=False, cell_restrictions=None):

        if model_average:
            adata = self.create_base_anndata_repeats(model_fns, cell_restrictions=cell_restrictions)
        else:
            adata = self.create_base_anndata(model_fns, cell_restrictions=cell_restrictions)

        # save directories where data came from
        adata.uns["model_fns"] = model_fns

        # print out which cell subclasses are present to confirm we're processing right data
        subclasses = adata.obs.subclass.unique()
        print(f"Subclasses present: {subclasses}")
        adata = self.add_unstructured_data(adata)
        adata = self.add_donor_stats(adata, add_gene_scores=self.add_gene_scores)

        return adata

    def normalize_data(self, x):

        x = np.float32(x)
        if self.normalize_gene_counts:
            x = 10_000 * x / np.sum(x)
        if self.log_gene_counts:
            x = np.log1p(x)

        return x

    def add_obs_from_metadata(self, adata):

        for k in self.obs_from_metadata:
            x = []
            for n in adata.obs["cell_idx"]:
                x.append(self.meta["obs"][k][n])
            adata.obs[k] = x

        return adata

    def create_single_anndata(self, z):

        k = self.obs_list[0]
        n = z[k].shape[0]
        latent = np.zeros((n, 1), dtype=np.uint8)  # dummy variable to create AnnData
        a = ad.AnnData(latent)

        for m, k in enumerate(self.obs_list):
            if z[k].ndim == 2:
                a.obs[k] = self.weight_probs(z[k], k)
            else:
                a.obs[k] = z[k]

        a.obs["cell_idx"] = z["cell_idx"]
        for k in self.extra_obs:
            a.obs[k] = np.array(self.meta["obs"][k])[z["cell_idx"]]
            if not isinstance(a.obs[k][0], str):
                idx = np.where(a.obs[k] < -99)[0]
                a.obs[k][idx] = np.nan

        a = self.add_obs_from_metadata(a)

        if "donor_px_r" in z.keys():
            a.uns["donor_px_r"] = z["donor_px_r"]

        # Only include samples with include_analysis=True
        if self.include_analysis_only:
            a = a[a.obs["include_analysis"] > 0]

        return a

    def concat_arrays(self, fns):

        for n, fn in enumerate(fns):
            z = pickle.load(open(fn, "rb"))

            for k in self.obs_list:
                if isinstance(z[k], list):
                    z[k] = np.concatenate(z[k], axis=0)

            if n == 0:
                x = copy.deepcopy(z)
            else:
                for k in self.obs_list:
                    x[k] = np.concatenate((x[k], z[k]), axis=0)
                for k in self.extra_obs + ["cell_idx"]:
                    x[k] = np.concatenate((x[k], z[k]), axis=0)
                if "donor_px_r" in z.keys():
                    for k in z["donor_px_r"].keys():
                        x["donor_px_r"][k] = z["donor_px_r"][k]

        return x

    def create_base_anndata_repeats(self, model_fns, cell_restrictions=None):

        x = []
        n_models = len(model_fns)
        for fns in model_fns:
            x0 = self.concat_arrays(fns)
            idx = np.argsort(x0["cell_idx"])
            for k in self.obs_list:
                x0[k] = x0[k][idx]
            for k in self.extra_obs + ["cell_idx"]:
                x0[k] = x0[k][idx]

            x.append(x0)

        x_new = copy.deepcopy(x0)
        for k in self.obs_list:
            x_new[k] = 0
            for n in range(n_models):
                x_new[k] += x[n][k] / n_models

        if "donor_px_r" in x0.keys():
            for k in x0["donor_px_r"].keys():
                x_new["donor_px_r"][k] = 0
                for n in range(n_models):
                    x_new["donor_px_r"][k] += x[n]["donor_px_r"][k] / n_models

        adata = self.create_single_anndata(x_new)
        if cell_restrictions is not None:
            for k, v in cell_restrictions.items():
                adata = adata[adata.obs[k] == v]

        return adata

    def create_base_anndata(self, model_fns, cell_restrictions=None):

        for n, fn in enumerate(model_fns):
            z = pickle.load(open(fn, "rb"))
            a = self.create_single_anndata(z)
            a.obs["split_num"] = n
            if n == 0:
                adata = a.copy()
            else:
                uns = adata.uns
                adata = ad.concat((adata, a), axis=0)
                adata.uns = uns
                if "donor_px_r" in a.uns.keys():
                    for k in a.uns["donor_px_r"].keys():
                        adata.uns["donor_px_r"][k] = a.uns["donor_px_r"][k]

        if cell_restrictions is not None:
            for k, v in cell_restrictions.items():
                adata = adata[adata.obs[k] == v]

        return adata

    def get_cell_index(self, model_fns):

        cell_idx = []
        cell_class = []
        cell_subclass = []

        for n, fn in enumerate(model_fns):
            z = pickle.load(open(fn, "rb"))
            cell_idx += z["cell_idx"].tolist()
            cell_class += self.meta["obs"]["class"][z["cell_idx"]].tolist()
            cell_subclass += self.meta["obs"]["subclass"][z["cell_idx"]].tolist()

        # assuming cell class is the same
        index = {cell_class[0]: cell_idx}
        for sc in np.unique(cell_subclass):
            idx = np.where(np.array(cell_subclass) == sc)[0]
            index[sc] = np.array(cell_idx)[idx]

        return index, np.unique(cell_class), np.unique(cell_subclass)

    @staticmethod
    def weight_probs(prob, k):
        # used to convert a vector prediction into a scalar

        if k == "Dementia_graded":
            w = np.array([0, 0.5, 1.0])
        elif k == "CERAD":
            w = np.array([1, 2, 3, 4])
        elif k == "BRAAK_AD":
            w = np.array([0, 1, 2, 3, 4, 5, 6])
        else:
            w = np.array([0, 1])
        w = w[None, :]

        return np.sum(prob * w, axis=1)

    def convert_apoe(self):

        self.meta["obs"]["apoe"] = np.zeros_like(self.meta["obs"]["ApoE_gt"])
        idx = np.where(self.meta["obs"]["ApoE_gt"] == 44)[0]
        self.meta["obs"]["apoe"][idx] = 2
        idx = np.where((self.meta["obs"]["ApoE_gt"] == 24) + (self.meta["obs"]["ApoE_gt"] == 34))[0]
        self.meta["obs"]["apoe"][idx] = 1
        # idx = np.where(np.isnan(self.meta["obs"]["ApoE_gt"]))[0]
        idx = np.where(self.meta["obs"]["ApoE_gt"] == "nan")[0]
        self.meta["obs"]["apoe"][idx] = np.nan

    def add_unstructured_data(self, adata):

        for k in self.obs_list:
            adata.uns[k] = []

        if self.gene_pathways is not None:
            adata.uns["go_bp_pathways"] = []
            adata.uns["go_bp_ids"] = []
            for k, v in self.gene_pathways.items():
                adata.uns["go_bp_pathways"].append(v["pathway"])
                adata.uns["go_bp_ids"].append(k)

        adata.uns["donors"] = []
        for subid in np.unique(adata.obs["SubID"]):
            adata.uns["donors"].append(subid)
            idx = np.where(np.array(adata.obs["SubID"].values) == subid)[0][0]
            for k in self.obs_list:
                adata.uns[k].append(adata.obs[k].values[idx])

        return adata

    def add_donor_stats(self, adata, add_gene_scores=True):

        n_donors = len(adata.uns["donors"])
        if self.gene_pathways is not None:
            n_pathways = len(self.gene_pathways)
            adata.uns[f"donor_pathway_means"] = np.zeros((n_donors, n_pathways), dtype=np.float32)
        if add_gene_scores:
            adata.uns[f"donor_gene_means"] = np.zeros((n_donors, self.n_genes), dtype=np.float32)

        adata.uns[f"donor_cell_count"] = np.zeros((n_donors,), dtype=np.float32)

        for k in self.donor_stats:
            adata.uns[f"donor_{k}"] = np.zeros((n_donors,), dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            a = adata[adata.obs["SubID"] == subid]
            count = 1e-6  # to prevent 1 / 0 errors
            go_scores = {}

            for n, i in enumerate(a.obs["cell_idx"]):
                if add_gene_scores:
                    data = np.memmap(
                        self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                    ).astype(np.float32)
                    data = self.normalize_data(data)
                    adata.uns[f"donor_gene_means"][m, :] += data
                count += 1

                if self.gene_pathways is not None:
                    for go_id in self.gene_pathways.keys():
                        go_idx = self.gene_pathways[go_id]["gene_idx"]
                        gene_exp = np.sum(np.log1p(data[go_idx]))
                        if not go_id in go_scores.keys():
                            go_scores[go_id] = [gene_exp]
                        else:
                            go_scores[go_id].append(gene_exp)

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

            if self.gene_pathways is not None:
                for n, go_id in enumerate(self.gene_pathways.keys()):
                    adata.uns[f"donor_pathway_means"][m, n] = np.mean(go_scores[go_id])

            adata.uns[f"donor_cell_count"][m] = count
            if add_gene_scores:
                adata.uns[f"donor_gene_means"][m, :] /= count

            for k in self.donor_stats:
                if "pred_" in k:
                    adata.uns[f"donor_{k}"][m] /= count

        return adata

    def add_donor_gene_pca(self, adata):

        # currently unused

        n_components = 20
        pca = PCA(n_components=n_components)

        gene_idx = self.important_genes["gene_idx"]
        gene_names = self.important_genes["gene_names"]
        adata.uns["gene_corr_names"] = gene_names

        n_donors = len(adata.uns["donors"])
        n_genes = len(gene_idx)
        adata.uns[f"donor_pca_explained_var"] = np.zeros((n_donors, n_components), dtype=np.float32)
        adata.uns[f"donor_pca_explained_var_ratio"] = np.zeros((n_donors, n_components), dtype=np.float32)
        adata.uns[f"donor_pca_components"] = np.zeros((n_donors, n_components, n_genes), dtype=np.float32)
        adata.uns[f"donor_pca_var"] = np.zeros((n_donors, n_genes), dtype=np.float32)
        adata.uns[f"donor_pca_noise_var"] = np.zeros((n_donors), dtype=np.float32)
        adata.uns[f"donor_pca_counts"] = np.zeros(n_donors, dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            print(m, len(adata.uns["donors"]), subid)
            a = adata[adata.obs["SubID"] == subid]
            gene_counts = []

            for n, i in enumerate(a.obs["cell_idx"]):
                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)

                if np.sum(data) < 100:
                    # rough quality check
                    continue

                data = self.normalize_data(data)
                gene_counts.append(data[gene_idx])

            gene_counts = np.stack(gene_counts, axis=0)
            gene_counts -= np.mean(gene_counts, axis=0, keepdims=True)
            gene_counts /= (0.1 + np.std(gene_counts, axis=0, keepdims=True))
            adata.uns[f"donor_pca_counts"][m] = gene_counts.shape[0]

            if gene_counts.shape[0] < 20:
                continue

            pca.fit(gene_counts)
            adata.uns[f"donor_pca_explained_var"][m, :] = pca.explained_variance_
            adata.uns[f"donor_pca_explained_var_ratio"][m, :] = pca.explained_variance_ratio_
            adata.uns[f"donor_pca_components"][m, :, :] = pca.components_
            # adata.uns[f"donor_pca_cov"][m, :, :] = pca.get_covariance()
            adata.uns[f"donor_pca_var"][m, :] = np.var(gene_counts, axis=0)
            adata.uns[f"donor_pca_noise_var"][m] = pca.noise_variance_

        return adata

