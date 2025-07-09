import os, sys
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.optimize as optimize
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import gc


class PLS:
    """Class to perform PLS regression to predict Braak (and possibly dementia)
    from raw spike counts.
    Used for Supplementary Figure 15
    """
    def __init__(
        self,
        meta_fn,
        data_path,
        splits_fn,
        restrictions,
        top_k_genes = 10_000,
        predictions = ["BRAAK_AD"],
    ):

        self.remove_sex_chrom = True
        self.protein_coding_only = True
        self.include_analysis = True
        self.top_k_genes = top_k_genes
        self.predictions = predictions
        self.splits = pickle.load(open(splits_fn, "rb"))

        self.data_path = data_path
        self.metadata = pickle.load(open(meta_fn, "rb"))
        self._restrict_samples(restrictions)
        self._get_gene_index()
        gc.collect()

    def _restrict_samples(self, restrictions):

        n_cells = len(self.metadata["obs"]["class"])
        cond = np.ones(n_cells, dtype=np.uint8)

        if self.include_analysis:
            if "include_analysis" in self.metadata["obs"].keys():
                cond *= (self.metadata["obs"]["include_analysis"] > 0)
                print("only analyzing include_analysis = True")

        if restrictions is not None:
            for k, v in restrictions.items():
                if isinstance(v, list):
                    cond *= np.sum(np.stack([np.array(self.metadata["obs"][k]) == v1 for v1 in v]), axis=0).astype(
                        np.uint8)
                else:
                    cond *= np.array(self.metadata["obs"][k]) == v

        self.cell_idx = np.where(cond)[0]
        self.n_samples = len(self.cell_idx)
        print(f"Number of samples: {self.n_samples}")

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
            cond *= self.metadata["var"]['percent_cells'] >= th
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

        n_splits = len(self.splits.keys())

        pls = PLSRegression(n_components=n_components)

        y = []
        for p in self.predictions:
            y.append(np.reshape(self.metadata["obs"][p], (-1, )))
        y = np.stack(y, axis=1)
        print(f"Shape of y: {y.shape}")

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

        for split_num in range(0, n_splits):
            train_idx_full = set(self.splits[split_num]["train_idx"])
            test_idx_full = set(self.splits[split_num]["test_idx"])

            train_idx = np.where(
                np.isin(self.cell_idx, list(train_idx_full)) * y_valid
            )[0]
            test_idx = np.where(np.isin(self.cell_idx, list(test_idx_full)))[0]

            pls.fit(self.data[train_idx, :], y[train_idx, :])

            y_hat_split = pls.predict(self.data[test_idx, :])
            for n, p in enumerate(self.predictions):
                self.results[f"pred_{p}"][test_idx] = y_hat_split[:, n]


class ProcessData:

    def __init__(self, meta_fn, data_path):

        self.meta = pickle.load(open(meta_fn, "rb"))
        self.data_path = data_path
        self.n_genes = len(self.meta["var"]["gene_name"])
        self.gene_names = self.meta["var"]["gene_name"]

        self.obs_list = ["pred_BRAAK_AD", "pred_Dementia"]
        self.obs_from_metadata = [
            "BRAAK_AD", "CERAD", "Dementia", "class", "subclass",  "SubID", "include_analysis",
        ]
        self.donor_stats = [
            "Dementia", "BRAAK_AD", "pred_BRAAK_AD", "pred_Dementia", "CERAD",
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

class PLSAnalysis:
    """Code to plot analysis comparing neural network moodel output to that of PLS regression
    Used for Supplementary Figure 15"""

    def __init__(self, model_fns, min_cell_count = 5, model_names = ["Neural net", "PLS"]):

        # model fns is a list, 1st entry is filename of neural network model output
        # 2nd entry it the model output of the PLS regression
        self.model_fns = model_fns
        self.min_cell_count = min_cell_count
        self.model_names = model_names

        self._model_accuracy()

    def _model_accuracy(self, n_repeats = 500):

        n_models = len(self.model_fns)
        self.braak_scores = np.zeros((n_models, n_repeats))
        self.dementia_scores = np.zeros((n_models, n_repeats))
        self.pred_braak = []
        self.pred_dementia = []

        for j, fn in enumerate(self.model_fns):

            adata = sc.read_h5ad(fn, "r")
            idx_braak = np.where(
                (adata.uns["donor_cell_count"] >= self.min_cell_count) * (adata.uns["donor_BRAAK_AD"] > -1)
            )[0]
            idx_dementia = np.where(
                (adata.uns["donor_cell_count"] >= self.min_cell_count) * (adata.uns["donor_Dementia"] > -1)
            )[0]

            self.pred_braak.append(adata.uns["donor_pred_BRAAK_AD"])
            self.pred_dementia.append(adata.uns["donor_pred_Dementia"])

            for n in range(n_repeats):
                idx0 = np.random.choice(idx_braak, len(idx_braak), replace=True)
                idx1 = np.random.choice(idx_dementia, len(idx_dementia), replace=True)

                self.braak_scores[j, n], _ = stats.pearsonr(
                    adata.uns["donor_pred_BRAAK_AD"][idx0],
                    adata.uns["donor_BRAAK_AD"][idx0]
                )

                self.dementia_scores[j, n] = self._classification_score(
                    adata.uns["donor_pred_Dementia"][idx1],
                    adata.uns["donor_Dementia"][idx1]
                )

    @staticmethod
    def _classification_score(x_pred, x_real):
        s0 = np.sum((x_real == 0) * (x_pred < 0.5)) / np.sum(x_real == 0)
        s1 = np.sum((x_real == 1) * (x_pred >= 0.5)) / np.sum(x_real == 1)
        return (s0 + s1) / 2

    def plot_model_accuracy(self, save_fig_fn = None):

        fig, ax = plt.subplots(1, 3, figsize=(7, 2.5))
        n_models = 2

        color = 'g'
        ax1 = ax[0]
        ax1.set_ylabel('Dementia class. \n accurary', color=color)
        u = np.mean(self.dementia_scores, axis=1)
        sd = np.std(self.dementia_scores, axis=1)
        ax1.bar(np.arange(n_models) - 0.166, u, yerr=sd, color=color, width=0.333, edgecolor='k', linewidth=0.5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(np.arange(n_models), [])
        ax1.set_ylim([0.5, 0.72])
        ax1.set_yticks([0.5, 0.6, 0.7])

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'm'
        ax2.set_ylabel('Braak Pearsonr R', color=color)  # we already handled the x-label with ax1
        u = np.mean(self.braak_scores, axis=1)
        sd = np.std(self.braak_scores, axis=1)
        ax2.bar(np.arange(n_models) + 0.166, u, yerr=sd, color=color, width=0.333, edgecolor='k', linewidth=0.5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks([0, 0.2, 0.4, 0.6])
        ax2.set_ylim([0, 0.6])
        ax1.set_xticks(np.arange(n_models), self.model_names, rotation=-45, ha="left")

        for j in range(n_models):
            r, _ = stats.pearsonr(self.pred_braak[j], self.pred_dementia[j])
            ax[j + 1].plot(self.pred_braak[j], self.pred_dementia[j], 'k.', markersize=2, label=f"R={r:1.3f}")
            ax[j + 1].set_xlabel("Predicted Braak")
            ax[j + 1].set_ylabel("Predicted Dementia")
            ax[j + 1].set_xlim([0.5, 6.])
            ax[j + 1].set_ylim([0.2, 1.1])
            ax[j + 1].legend(loc="upper left", fontsize=8)
            ax[j + 1].set_title(self.model_names[j])

        # ax2.grid(axis='y')
        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()

    def load_zenith_data(self, braak_fn0, braak_fn1, resilience_fn0, resilience_fn1, pval_threshold = 0.05):

        # load and concatenate the Zenith output files from the neural network and PLS models
        dfs = []
        for fn in [braak_fn0, braak_fn1, resilience_fn0, resilience_fn1]:
            df = pd.read_csv(fn)
            dfs.append(
                pd.DataFrame({
                    "pathway": df["Unnamed: 0"].values,
                    "z-score": df.delta.values / df.se.values,
                    "FDR": df.FDR.values,
                })
            )


        df_neural_net = pd.merge(dfs[0], dfs[1], on="pathway", how="outer", suffixes=("_braak", "_resilience"))
        df_pls = pd.merge(dfs[2], dfs[3], on="pathway", how="outer", suffixes=("_pls_braak", "_pls_resilience"))
        self.df = df_neural_net.merge(df_pls, on="pathway", how="outer")

        # only include pathways in which at least one of braak/resilience, NN/PLS is significant
        idx_significant = (
                (self.df["FDR_braak"].values < pval_threshold) +
                (self.df["FDR_resilience"].values < pval_threshold) +
                (self.df["FDR_pls_braak"].values < pval_threshold) +
                (self.df["FDR_pls_resilience"].values < pval_threshold)
        )
        self.df  = self.df[idx_significant]

    def pathway_scatter_plot(self, save_fig_fn = None):

        idx = []
        idx.append((self.df["z-score_braak"].values > 0) * (self.df["z-score_resilience"].values > 0))
        idx.append((self.df["z-score_braak"].values < 0) * (self.df["z-score_resilience"].values < 0))
        idx.append((self.df["z-score_braak"].values > 0) * (self.df["z-score_resilience"].values < 0))
        idx.append((self.df["z-score_braak"].values < 0) * (self.df["z-score_resilience"].values > 0))

        diff = []
        mean_diff = []
        pvals = []

        for i in range(4):
            d = self.df["z-score_braak"].values[idx[i]] - self.df["z-score_pls_braak"].values[idx[i]]
            p = stats.wilcoxon(d)[1]
            diff.append(d)
            mean_diff.append(np.mean(d))
            pvals.append(p)

        f, ax = plt.subplots(1, 2, figsize=(5, 2.5), sharex=True, sharey=True)
        colors = ["r", "b"]
        for i in range(2):
            ax1 = ax[i]
            for j in range(2):
                k = idx[i * 2 + j]
                ax1.plot([-7, 7], [-7, 7], 'k--')
                ax1.plot(self.df["z-score_braak"].values[k], self.df["z-score_pls_braak"].values[k], '.', color=colors[j])
                ax1.set_xlabel("z-score - Network model")
                ax1.set_ylabel("z-score - PLS")
                if j == 0:
                    if pvals[i * 2 + j] < 0.001:
                        ax1.text(-2, 5, f"Diff = {mean_diff[i * 2 + j]:1.2f} \n P = {pvals[i * 2 + j]:1.1e}",
                                 color=colors[j])
                    else:
                        ax1.text(-2, 5, f"Diff = {mean_diff[i * 2 + j]:1.2f} \n P = {pvals[i * 2 + j]:1.3f}",
                                 color=colors[j])
                else:
                    if pvals[i * 2 + j] < 0.001:
                        ax1.text(-1, -5, f"Diff = {mean_diff[i * 2 + j]:1.2f} \n P = {pvals[i * 2 + j]:1.1e}",
                                 color=colors[j])
                    else:
                        ax1.text(-1, -5, f"Diff = {mean_diff[i * 2 + j]:1.2f} \n P = {pvals[i * 2 + j]:1.3f}",
                                 color=colors[j])

        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()

    def pathway_comparison_figure(self, save_fig_fn = None):

        diff = []
        z0 = []
        z1 = []
        p0 = []
        p1 = []
        pathways = []
        for i in range(len(self.df)):
            diff.append(self.df["z-score_braak"].values[i] - self.df["z-score_pls_braak"].values[i])
            z0.append(self.df["z-score_braak"].values[i])
            z1.append(self.df["z-score_pls_braak"].values[i])
            p0.append(self.df["FDR_braak"].values[i])
            p1.append(self.df["FDR_pls_braak"].values[i])
            pathways.append(self.df["pathway"].values[i].split(":")[1])

        diff = np.stack(diff)
        z0 = np.stack(z0)
        z1 = np.stack(z1)
        p0 = np.stack(p0)
        p1 = np.stack(p1)
        pathways = np.stack(pathways)

        idx_diff = np.argsort(diff)[::-1][:10][::-1]

        y_positions = np.arange(len(pathways[idx_diff]))  # Base y positions
        bar_height = 0.4

        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        ax.barh(y_positions - bar_height / 2, z0[idx_diff], bar_height, label="Network model")
        ax.barh(y_positions + bar_height / 2, z1[idx_diff], bar_height, label="PLS")
        ax.legend(fontsize=7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(pathways[idx_diff], fontsize=8)
        ax.set_xlabel("z-score", fontsize=8)
        for i in range(10):
            if p0[idx_diff[i]] < 0.05:
                ax.text(0.5, i - 0.5, "*", color="white", fontsize=12)
            if p1[idx_diff[i]] < 0.05:
                ax.text(0.5, i - 0.1, "*", color="white", fontsize=12)

        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()

