import os
import copy
import pandas as pd
import numpy as np
import scanpy as sc

import scipy.stats as stats
import scipy.signal as signal
import scipy.optimize as optimize
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.patches as patches
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

import seaborn as sns

import palantir
from py_monocle import (
    learn_graph,
    order_cells,
    compute_cell_states,
    regression_analysis,
    differential_expression_genes,
)

import figure_utils

def classification_score(x_pred, x_real):
    s0 = np.sum((x_real == 0) * (x_pred < 0.5)) / np.sum(x_real == 0)
    s1 = np.sum((x_real == 1) * (x_pred >= 0.5)) / np.sum(x_real == 1)
    return (s0 + s1) / 2

def bootstrap_accuracy(data_fns, n_repeats, min_cell_count):

    braak_scores = np.zeros((len(data_fns), n_repeats))
    dementia_scores = np.zeros((len(data_fns), n_repeats))

    for j, fn in enumerate(data_fns):

        adata = sc.read_h5ad(fn, "r")
        idx_braak = np.where(
            (adata.uns["donor_cell_count"] >= min_cell_count) * (adata.uns["donor_BRAAK_AD"] > -1)
        )[0]
        idx_dementia = np.where(
            (adata.uns["donor_cell_count"] >= min_cell_count) * (adata.uns["donor_Dementia"] > -1)
        )[0]

        for n in range(n_repeats):
            idx0 = np.random.choice(idx_braak, len(idx_braak), replace=True)
            idx1 = np.random.choice(idx_dementia, len(idx_dementia), replace=True)

            braak_scores[j, n], _ = stats.pearsonr(
                adata.uns["donor_pred_BRAAK_AD"][idx0],
                adata.uns["donor_BRAAK_AD"][idx0]
            )

            dementia_scores[j, n] = classification_score(
                adata.uns["donor_pred_Dementia"][idx1],
                adata.uns["donor_Dementia"][idx1]
            )

    return braak_scores, dementia_scores

def model_accuracy(data_fns, cell_names, n_repeats = 20_000, min_cell_count = 5, save_fig_fn = None):

    """Generates Figure 7b: model accuracy"""
    braak_scores, dementia_scores = bootstrap_accuracy(data_fns, n_repeats, min_cell_count)

    scale = 1.5
    fig, ax1 = plt.subplots(1, 1, figsize=(3 * scale, 1.75 * scale))
    N = len(data_fns)

    color = 'g'
    # ax1.set_xlabel('Cell Class')
    ax1.set_ylabel('Dementia class. \n accurary', color=color)
    u = np.mean(dementia_scores, axis=1)
    sd = np.std(dementia_scores, axis=1)
    ax1.bar(np.arange(N) - 0.166, u, yerr=sd, color=color, width=0.333, edgecolor='k', linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(np.arange(N), [])
    ax1.set_ylim([0.5, 0.72])
    ax1.set_yticks([0.5, 0.6, 0.7])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'm'
    ax2.set_ylabel('Braak Pearsonr R', color=color)  # we already handled the x-label with ax1
    u = np.mean(braak_scores, axis=1)
    sd = np.std(braak_scores, axis=1)
    ax2.bar(np.arange(N) + 0.166, u, yerr=sd, color=color, width=0.333, edgecolor='k', linewidth=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0, 0.2, 0.4, 0.6])
    ax2.set_ylim([0, 0.6])
    ax1.set_xticks(np.arange(N), cell_names, rotation=-45, ha="left")
    # ax2.grid(axis='y')
    if save_fig_fn is not None:
        plt.savefig(save_fig_fn)
    plt.tight_layout()
    plt.show()


class TrajectoryInference:

    def __init__(self, data_fn):
        """Class to calculate Monocle3 and Palantir pseudotime trajectories
        Used for Figure 7c and Supp Figure 27"""

        self.adata = sc.read_h5ad(data_fn)
        self._starting_indices()

    def _starting_indices(self):
        """define satarting indices for Monocle and Palantir, corresponding to little disease burden
        We found that certain areas of UMAP space (defined below) produce the strongest correlations
        between Braak and pseudotime"""

        self.starting_idx = np.where(
            (self.adata.obs.Braak.values == 0) *
            (self.adata.obs.dementia.values == 0) *
            (self.adata.obs.MCI.values == 0) *
            (self.adata.obs.CERAD.values == 1) *
            (self.adata.obs.age.values <= 70) *
            (self.adata.obsm["X_umap"][:, 0] > 19) *
            (self.adata.obsm["X_umap"][:, 1] > 1)
        )[0]

    def compute_monocle3(self, n_max_repeats = 500):

        pseudotime = []
        correlations = []

        if len(self.starting_idx) > n_max_repeats:
            start_idx = np.random.choice(self.starting_idx, n_max_repeats, replace = False)
        else:
            start_idx = self.starting_idx

        for i, start in enumerate(start_idx):

            # calculate graph (stochastic!)
            projected_points, mst, centroids = learn_graph(
                matrix=self.adata.obsm["X_umap"],
                clusters=None,
            )

            pt = order_cells(
                self.adata.obsm["X_umap"],
                centroids,
                mst=mst,
                projected_points=projected_points,
                root_cells=start,
            )
            pseudotime.append(pt)
            idx = np.where(self.adata.obs.Braak.values > -1)[0]
            r, _ = stats.pearsonr(pt[idx], self.adata.obs.Braak.values[idx])
            correlations.append(r)
            print(f"Iteration {i} Pearsonr R = {r:1.3f} max R = {np.max(correlations):1.3f},")

        pseudotime = np.stack(pseudotime)
        j = np.argmax(correlations)
        self.adata.obs["Monocle3 pseudotime"] = pseudotime[j, :]

    def compute_palantir(self, n_max_repeats=500):

        pseudotime = []
        correlations = []

        if len(self.starting_idx) > n_max_repeats:
            start_idx = np.random.choice(self.starting_idx, n_max_repeats, replace=False)
        else:
            start_idx = self.starting_idx

        adata_plr = self.adata.copy()
        palantir.utils.run_diffusion_maps(adata_plr, pca_key="X_pca_regressed_harmony", n_components=5)
        palantir.utils.determine_multiscale_space(adata_plr)

        for i, start in enumerate(start_idx):

            # calculate graph (stochastic!)
            plr = palantir.core.run_palantir(
                adata_plr,
                adata_plr.obs.index[start], # doesn't vary between different staring cells.
                num_waypoints=500,
                # terminal_states=[adata_new.obs.index[i] for i in end_sample] # adding this doesn't seem to change the results
            )

            pseudotime.append(plr.pseudotime.values)
            idx = np.where(self.adata.obs.Braak.values > -1)[0]
            r, _ = stats.pearsonr(plr.pseudotime.values[idx], self.adata.obs.Braak.values[idx])
            correlations.append(r)
            print(f"Iteration {i} Pearsonr R = {r:1.3f} max R = {np.max(correlations):1.3f},")

        pseudotime = np.stack(pseudotime)
        j = np.argmax(correlations)
        self.adata.obs["Palantir pseudotime"] = pseudotime[j, :]

    def violin_plot_data(self, key, braak_vals, braak_key="Braak"):
        vals = []
        for b in braak_vals:
            a = self.adata[self.adata.obs[braak_key] == b]
            vals.append(a.obs[key].values)
        return vals

    def _violin_plot_data(self, key, braak_vals, braak_key="Braak"):
        vals = []
        for b in braak_vals:
            a = self.adata[self.adata.obs[braak_key] == b]
            vals.append(a.obs[key].values)
        return vals

    def generate_figure(self, save_fig_fn = None):

        f, ax = plt.subplots(2, 4, figsize=(10, 5))

        x = self.adata.obsm["X_umap"]
        y = self.adata.obs["Braak"].values
        idx = np.where(~np.isnan(y) * (y >= 0))[0]

        vals = [
            self.adata.obs["Braak"].values,
            self.adata.obs["pred_BRAAK_AD"].values,
            self.adata.obs["Monocle3 pseudotime"].values,
            self.adata.obs["Palantir pseudotime"].values,
        ]

        for n in range(4):
            y = vals[n]
            ax[0, n].hexbin(x[:, 0], x[:, 1], y, gridsize=50)
            divider = make_axes_locatable(ax[0, n])
            cax = divider.append_axes('right', size='4%', pad=0.02)
            norm = matplotlib.colors.Normalize(vmin=np.min(y), vmax=np.max(y))
            cbar = f.colorbar(cm.ScalarMappable(norm=norm), cax=cax, orientation='vertical')

        for i in range(4):
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
            ax[0, i].set_xlabel("UMAP 1")
            ax[0, i].set_ylabel("UMAP 2")

        ax[0, 0].set_title("Braak")
        ax[0, 1].set_title("Disease psuedotime")
        ax[0, 2].set_title("Monocle3")
        ax[0, 3].set_title("Palantir")

        y = self.adata.obs["pred_BRAAK_AD"].values
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        self.adata.obs["pred_BRAAK_AD_normalized"] = y

        terms = ["pred_BRAAK_AD_normalized", "Monocle3 pseudotime", "Palantir pseudotime"]
        labels = ["Disease psuedotime", "Monocle3", "Palantir"]
        for n in range(3):
            vals = self._violin_plot_data(terms[n], np.arange(7), braak_key="Braak")
            ax[1, n].violinplot(vals, np.arange(7))
            r, _ = stats.pearsonr(self.adata.obs[terms[n]][idx], self.adata.obs.Braak.values[idx])
            ax[1, n].set_title(f"{labels[n]}, R={r:1.3f}", fontsize=9)
            ax[1, n].set_xlabel("Braak", fontsize=9)
            ax[1, n].set_ylabel("Pseudotime", fontsize=9)

        ax[1, 3].remove()

        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()



class CellTrajectories:

    def __init__(
        self,
        data_fns,
        cell_names = ["EN", "IN", "Astro", "Immune", "Oligo", "OPC", "Mural", "Endo"],
        min_cell_count = 5,
        alpha = 4,
        edge = 0,
    ):

        """XXXXXXXXXXXXXX"""

        self.data_fns = data_fns
        self.cell_names = cell_names
        self.min_cell_count = min_cell_count
        self.alpha = alpha
        self.edge = edge

        self._get_gene_names()

    def _get_gene_names(self):

        adata = sc.read_h5ad(self.data_fns[0], "r")
        self.gene_names = adata.uns["gene_name"]

    def caclulate_cell_trajectories(self, n_components = 20):

        self.cell_traj = {}

        for fn, name in zip(self.data_fns, self.cell_names):
            print(f"Calculating trajectories for {name}...")
            adata = sc.read_h5ad(fn, "r")
            cell_count = adata.uns["donor_cell_count"]

            idx = (cell_count >= self.min_cell_count)
            pred_dementia = adata.uns["donor_pred_Dementia"][idx]
            pred_braak = adata.uns["donor_pred_BRAAK_AD"][idx]

            idx_sort = np.argsort(pred_braak)
            pred_braak = pred_braak[idx_sort]
            pred_dementia = pred_dementia[idx_sort]
            donor_gene_means = adata.uns["donor_gene_means"][idx][idx_sort, :]
            gamma = self.alpha / np.var(pred_braak)

            self.cell_traj[name] = {"pred_braak": pred_braak, "gene_exp": donor_gene_means}

            self.cell_traj[name]["pred_braak_traj"], self.cell_traj[name]["gene_exp_traj"] = calculate_trajectories_single(
                pred_braak,
                donor_gene_means,
                gamma,
                neighbors=None,
            )

            self.cell_traj[name]["resilience_traj"] = calculate_trajectory_residuals(
                pred_braak,
                pred_dementia,
                donor_gene_means,
                gamma,
                neighbors=None,
            )

            if self.edge > 0:
                self.cell_traj[name]["pred_braak_traj"] = self.cell_traj[name]["pred_braak_traj"][self.edge:-self.edge]
                self.cell_traj[name]["gene_exp_traj"] = self.cell_traj[name]["gene_exp_traj"][self.edge:-self.edge, :]
                self.cell_traj[name]["resilience_traj"] = self.cell_traj[name]["resilience_traj"][self.edge:-self.edge, :]

            pca = PCA(n_components=n_components)
            self.cell_traj[name]["pca_gene_exp_traj"]  = pca.fit_transform(self.cell_traj[name]["gene_exp_traj"])

    def plot_example_traj(self, gene_name = "NAV2", cell_name = "Immune", save_fig_fn = None):

        f, ax = plt.subplots(1, 1, figsize=(3.75, 3))

        i = np.where(np.array(self.gene_names ) == "NAV2")[0][0]
        y = self.cell_traj[cell_name]
        ax.plot(y["pred_braak"], y["gene_exp"][:, i], 'k.', markersize=3, label="Donor-averaged expression")
        ax.plot(y["pred_braak_traj"], y["gene_exp_traj"][:, i], 'r.', markersize=4, label="Smoothed fit")
        ax.set_xlabel("Predicted Braak")
        ax.set_ylabel("Mean expression")
        ax.set_yticks([0, 1, 2])
        ax.legend(fontsize=6)
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()

    def calculate_nonlinearity(self):

        explained_var_piecewise = []
        explained_var_single = []

        for name in self.cell_names:
            x = self.cell_traj[name]["pred_braak_traj"]
            y = self.cell_traj[name]["pca_gene_exp_traj"]
            _, ex_var_piece, _, _ = fit_piecewise_search(
                x, y, time_resolution=1, no_jump=True, max_time=None,
            )
            _, _, _, ex_var_single = fit_single(x, y)

            explained_var_piecewise.append(ex_var_piece)
            explained_var_single.append(ex_var_single)

        return np.stack(explained_var_piecewise), np.stack(explained_var_single)

    def plot_nonlinearity(self, save_fig_fn = None):

        explained_var_piecewise, explained_var_single = self.calculate_nonlinearity()

        f, ax = plt.subplots(2, 1, figsize=(3.75, 4), sharex=True)
        diff = explained_var_piecewise - explained_var_single
        ax[0].bar(np.arange(8) - 0.166, explained_var_single, width=0.33, label="Single fit")
        ax[0].bar(np.arange(8) + 0.166, explained_var_piecewise, width=0.33, label="Piecewise fit")
        ax[0].set_ylim([0.6, 1.0])
        ax[0].set_ylabel("Explained variance", fontsize=9)
        ax[0].legend(fontsize=8)
        ax[1].bar(np.arange(8), diff, color="darkgrey")
        # ax[1].set_ylabel("Difference in \n explained variance", fontsize=9)
        ax[1].set_ylabel("Nonlinearity index", fontsize=9)
        ax[1].set_xticks(np.arange(8), self.cell_names, fontsize=9, rotation=-45, ha="left")
        ax[1].set_yticks([0, 0.1, 0.2])

        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()

    def output_slopes_for_zenith(self, zenith_save_path, suffix = "", cell_names = None):

        """Outputs slopes/resilience to csv file, can then be used for Zenith analysis"""
        cell_names = self.cell_names if cell_names is None else cell_names

        for name in cell_names:
            save_fn = os.path.join(zenith_save_path, f"{name}_zenith_input.csv")

            slopes = {}
            n_slopes = len(self.cell_traj[name][f"slopes_{suffix}"])
            for n in range(n_slopes):
                slopes[f"Braak{n}"] = self.cell_traj[name][f"slopes_{suffix}"][n, :]
            for n in range(n_slopes):
                slopes[f"Resilience{n}"] = self.cell_traj[name][f"resilience_{suffix}"][n, :]

            genes = copy.deepcopy(list(self.gene_names))
            for n, g in enumerate(genes):
                if n == 0:
                    continue
                if g in genes[:n]:
                    genes[n] = g + "-1"
            output_zenith(slopes, save_fn, genes)

    def calculate_slopes(
        self,
        time_pts_donors = None,
        time_pts_pct = None,
        suffix = "",
        cell_names = None, # if None, just use default stored in self.cell_names
    ):

        """Calculate slopes with respect to Braak and resilience scores"""

        # time_pts_donors = e.g. [[0, 60], [20, 80], [40, 100],...]
        # time_pts_pct = e.g. [[0, 0.2], [0.2, 1.0]]
        cell_names = self.cell_names if cell_names is None else cell_names

        assert (time_pts_donors is None) ^ (time_pts_pct is None), "Either time_pts_donors or time_pts_pct must be defined "

        for name in cell_names:

            N = len(self.cell_traj[name]["pred_braak_traj"])
            time_pts = []
            if time_pts_pct is not None:
                for t in time_pts_pct:
                    time_pts.append([int(t[0] * N), int(t[1] * N)])
            else:
                for t in time_pts_donors:
                    if t[1] <= N:
                        time_pts.append([t[0], t[1]])

            self.cell_traj[name][f"slopes_{suffix}"], self.cell_traj[name][f"resilience_{suffix}"], _ = calculate_slopes(
                time_pts,
                self.cell_traj[name]["pred_braak_traj"],
                self.cell_traj[name]["gene_exp_traj"],
                self.cell_traj[name]["resilience_traj"],
            )

            self.cell_traj[name][f"braak_mid_time_pts_{suffix}"] = [
                self.cell_traj[name]["pred_braak_traj"][(t[0] + t[1]) // 2] for t in time_pts
            ]

    def plot_mean_traj(self, cell_name, top_k, xlabel, ax, suffix="early_late"):

        slopes = self.cell_traj[cell_name][f"slopes_{suffix}"]
        preds  = self.cell_traj[cell_name][f"pred_braak_traj"]
        traj = self.cell_traj[cell_name]["gene_exp_traj"]

        idx = np.argsort(preds)
        preds_sorted = preds[idx]
        traj_sorted = traj[idx]

        fs = 7
        suffix = ["Early increase", "Early decrease", "Late increase", "Late decrease"]
        for n, x in enumerate([slopes[0, :], slopes[1, :]]):
            idx_top = np.argsort(x)[::-1][:top_k]
            idx_bottom = np.argsort(x)[:top_k]

            for m, idx in enumerate([idx_top, idx_bottom]):
                y = traj_sorted[:, idx]
                print(y.shape)
                u0 = np.min(y, axis=0, keepdims=True)
                u1 = np.max(y, axis=0, keepdims=True) - u0

                y = (y - u0) / u1
                u = np.mean(y, axis=1)
                sd = np.std(y, axis=1)  # / np.sqrt(top_k)
                time = (preds_sorted - np.min(preds_sorted)) / (np.max(preds_sorted) - np.min(preds_sorted))
                print("AA", u.shape, time.shape)
                ax.fill_between(time, u - sd, u + sd, alpha=0.1)
                ax.plot(time, u, linewidth=3, label=suffix[m + 2 * n])
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels([0, 1], fontsize=fs)
                ax.set_yticklabels([0, 1], fontsize=fs)
                ax.set_xlabel(xlabel, fontsize=fs)
                ax.set_ylabel("Normalized mean expression", fontsize=fs)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

        ax.legend(fontsize=7, loc='upper right')

    def plot_gene_scores_specified(
        self,
        ax,
        fig,
        cell_name="Immune",
        suffix="early_late",
        top_k=5,
        normalize=True,
        interpolate_traj=True,
        edge_num_exclude=10,
    ):

        slopes = self.cell_traj[cell_name][f"slopes_{suffix}"]
        pred_braak = self.cell_traj[cell_name][f"pred_braak_traj"]
        braak_traj = self.cell_traj[cell_name]["gene_exp_traj"]
        idx = np.argsort(pred_braak)
        pred_braak_sorted = pred_braak[idx]
        braak_traj_sorted = braak_traj[idx]


        fs = 7
        idx = []
        gene_list = exclusive_gene_lists(slopes, top_k=250)
        for i in range(4):
            idx += gene_list[i][:top_k]


        if interpolate_traj:
            N = 100
            z = np.zeros((N, len(idx)))
            for m, j in enumerate(idx):
                """
                x0 = np.linspace(np.min(pred_braak), np.max(pred_braak), N)
                z[:N, m] = np.interp(x0, pred_braak, braak_traj[:, j])
                x0 = np.linspace(np.min(pred_dementia), np.max(pred_dementia), N)
                z[N:, m]  = np.interp(x0, pred_dementia, dementia_traj[:, j])

                """
                x0 = np.linspace(np.min(pred_braak_sorted), np.max(pred_braak_sorted), N + 1)
                y = np.interp(x0, pred_braak_sorted, braak_traj_sorted[:, j])
                z[:N, m] = np.diff(y) / np.diff(x0)
                # x0 = np.linspace(np.min(pred_braak_sorted), np.max(pred_braak_sorted), N)
                # y = np.interp(x0, pred_braak_sorted, resilience_sorted[:, j])
                # z[N:, m] = y

            # z -= np.mean(z, axis=0, keepdims=True)
            # z /= np.max(z, axis=0, keepdims=True)
        else:
            z = np.vstack((slopes_br[:, idx], slopes_dm[:, idx]))

        m = np.max(np.abs(z))
        sd = np.std(z)

        im = ax.imshow(z, vmin=-5 * sd, vmax=5 * sd, aspect="auto", cmap="bwr", interpolation="none")
        # ax.plot([-0.5, 4*top_k-0.5], [N, N], 'k-')
        ax.set_xlim([-0.5, 4 * top_k - 0.5])
        genes = self.gene_names[idx].tolist()
        for n, g in enumerate(genes):
            if g.startswith("ENS"):
                genes[n] = g[:3] + "..." + g[-4:]
        ax.set_xticks(range(len(idx)), genes, rotation=-45, fontsize=fs, ha="left")
        for n, xtick in enumerate(ax.get_xticklabels()):
            xtick.set_color(colors[n // top_k])
        ax.set_yticks([50], ["BRAAK"], fontsize=fs, rotation=90)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=3.0)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='1%', pad=0.01)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=fs)

    def plot_gene_summary(
        self,
        cell_name="Immune",
        zenith_fns=None,
        magma_fns=None,
        top_k=250,
        xlabel="Disease pseudotime",
        save_fig_fn=None,
        n_pathways=5,
        top_k_genes=20,
    ):
        fig = plt.figure(constrained_layout=True, figsize=(10.5, 5))
        gs = gridspec.GridSpec(10, 10, figure=fig)

        bbox = [0, 0, 1, 1]
        ax0 = fig.add_subplot(gs[:6, :3])
        self.plot_mean_traj(cell_name, top_k, xlabel, ax0)

        dataframes = process_zenith_dfs(zenith_fns, paths_per_df=n_pathways)
        ax1 = fig.add_subplot(gs[:6, 5:7])
        plot_zenith_results(dataframes, ax1, fig, paths_per_df=n_pathways)

        magma_scores, magma_titles = process_magma_dfs(magma_fns)
        ax1 = fig.add_subplot(gs[:6, 9])
        plot_magma_results(magma_scores, magma_titles, ax1, fig)

        ax1 = fig.add_subplot(gs[7:, :])

        self.plot_gene_scores_specified(
            ax1,
            fig,
            top_k=top_k_genes,
            normalize=True,
            interpolate_traj=True,
        )

        fig.get_layout_engine().set(w_pad=0.25, h_pad=0.25, hspace=0.1, wspace=0.1)
        # plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()


class GlobalTrajectory:

    def __init__(self, data_fns, min_cell_count = 5, alpha = 4, edge = 0):

        """PCA plot for Supplementary Figure 16, identify transition times"""

        self.data_fns = data_fns
        self.min_cell_count = min_cell_count
        self.alpha = alpha
        self.edge = 0

        self._get_gene_count()
        self._get_elibgible_donors()
        self._concat_data()

    def _get_gene_count(self):
        adata = sc.read_h5ad(self.data_fns[0], "r")
        self.n_genes = adata.uns["donor_gene_means"].shape[1]
        print(f"Number of genes: {self.n_genes}")

    def _get_elibgible_donors(self):

        # discover which donors have at least the minimum number of cells for all eight cell classes
        donor_cell_counts = {}
        for fn in self.data_fns:
            adata = sc.read_h5ad(fn, "r")

            for donor, n in zip(adata.uns["donors"], adata.uns["donor_cell_count"]):
                if not donor in donor_cell_counts.keys():
                    donor_cell_counts[donor] = [n]
                else:
                    donor_cell_counts[donor].append(n)

        self.eligible_donors = []
        for donor, counts in donor_cell_counts.items():
            if np.sum(np.stack(counts) >= self.min_cell_count) == len(self.data_fns):
                self.eligible_donors.append(donor)

        print(f"Number of eligible donors: {len(self.eligible_donors)}")

    def _concat_data(self):

        fields = ["pred_BRAAK_AD", "BRAAK_AD"]
        n_donors = len(self.eligible_donors)
        self.traj_data = {"gene_exp": np.zeros((n_donors, len(self.data_fns) * self.n_genes), dtype=np.float32)}
        for k in fields:
            self.traj_data[k] = np.zeros(n_donors)


        for i, fn in enumerate(self.data_fns):
            adata = sc.read_h5ad(fn, "r")
            n_genes = len(adata.var)
            gene_data_donor = np.zeros((n_donors, self.n_genes), dtype=np.float32)

            for n, donor in enumerate(self.eligible_donors):
                idx_donor = np.where(np.array(adata.uns["donors"]) == donor)[0][0]
                i0 = self.n_genes * i
                i1 = self.n_genes * (i + 1)
                self.traj_data["gene_exp"][n, i0: i1] = adata.uns["donor_gene_means"][idx_donor]
                for k in fields:
                    val = float(adata.uns[f"donor_{k}"][idx_donor])
                    self.traj_data[k][n] += val / len(self.data_fns)

        idx = self.traj_data["BRAAK_AD"] > - 1
        r, _ = stats.pearsonr(self.traj_data["pred_BRAAK_AD"][idx], self.traj_data["BRAAK_AD"][idx])
        print(f"Correlation between actual and predicted Braak: {r:1.4f}")


    def calculate_trajectories(self, k = "pred_BRAAK_AD"):

        gamma = self.alpha / np.var(self.traj_data[k])
        idx_sort = np.argsort(self.traj_data[k])

        self.traj_data[f"filtered_{k}"],  self.traj_data[f"filtered_gene_exp"] = calculate_trajectories_single(
            self.traj_data[k][idx_sort], self.traj_data["gene_exp"][idx_sort, :], gamma, neighbors = None,
        )

        if self.edge > 0:
            self.traj_data[f"filtered_{k}"] = self.traj_data[f"filtered_{k}"][self.edge:-self.edge]
            self.traj_data[f"filtered_gene_exp"] = self.traj_data[f"filtered_gene_exp"][self.edge:-self.edge, :]

    def calculate_pca_transition_points(self, k = "pred_BRAAK_AD", n_components = 4, time_resolution = 1):

        self.traj_data["pca"] = PCA(n_components=n_components)
        self.traj_data[f"pca_filtered_gene_exp"] = self.traj_data["pca"].fit_transform(
            self.traj_data[f"filtered_gene_exp"]
        )

        # calculate transition points t0 and t1
        self.traj_data["t1"], _, _, _ = fit_piecewise_search(
            self.traj_data[f"filtered_{k}"],
            self.traj_data["pca_filtered_gene_exp"],
            time_resolution=time_resolution,
            max_time=None,
            no_jump=True,
        )
        t1 = self.traj_data["t1"]
        self.traj_data["t0"], _, _, _ = fit_piecewise_search(
            self.traj_data[f"filtered_{k}"][:t1],
            self.traj_data["pca_filtered_gene_exp"][:t1, :],
            time_resolution=time_resolution,
            max_time=None,
            no_jump=True,
        )

        self.traj_data["t0_pct"] = self.traj_data["t0"] / len(self.eligible_donors)
        self.traj_data["t1_pct"] = self.traj_data["t1"] / len(self.eligible_donors)

        print(f" Transition times: donor {self.traj_data['t0']} and donor {self.traj_data['t1']}")

    def plot_pca_figure(self, k = "pred_BRAAK_AD", save_fig_fn = None):

        """PCA plot for Supplementary Figure 16"""

        fig, ax = plt.subplots(1, 4, figsize=(7, 2))
        fs = 9
        pcs = [[0, 1], [0, 2], [1, 2]]

        for n, pc in enumerate(pcs):

            p = self.traj_data[f"filtered_{k}"]
            y = self.traj_data["pca_filtered_gene_exp"]
            ax[n].scatter(y[:, pc[0]], y[:, pc[1]], s=5, c=p, cmap='viridis')

            ax[n].set_xlabel(f"PC {pc[0] + 1}", fontsize=fs)
            ax[n].set_ylabel(f"PC {pc[1] + 1}", fontsize=fs)
            ax[n].spines['top'].set_visible(False)
            ax[n].spines['right'].set_visible(False)

            t0 = self.traj_data["t0"]
            t1 = self.traj_data["t1"]
            ax[n].plot(y[t0, pc[0]], y[t0, pc[1]], 'm.', markersize=12)
            ax[n].plot(y[t1, pc[0]], y[t1, pc[1]], 'c.', markersize=12)

        divider = make_axes_locatable(ax[n])
        cax = divider.append_axes('right', size='4%', pad=0.02)
        norm = matplotlib.colors.Normalize(vmin=np.min(p), vmax=np.max(p))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=fs)

        m = len(pcs)
        ax[len(pcs)].bar(np.arange(1, 4), self.traj_data["pca"].explained_variance_ratio_[:3], color="darkgrey")
        ax[len(pcs)].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax[len(pcs)].set_ylim([0, 0.8])
        ax[len(pcs)].set_xlabel("PC")
        ax[len(pcs)].set_ylabel("Explained variance", fontsize=fs)

        ax[len(pcs)].spines['top'].set_visible(False)
        ax[len(pcs)].spines['right'].set_visible(False)

        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()


class PathEnrichmentAllCells:

    def __init__(self, input_path, FDR_cutoff = 0.01, min_genes = 10, max_genes = 250):

        self.FDR_cutoff = FDR_cutoff
        self.min_genes = min_genes
        self.max_genes = max_genes

        self._get_input_files(input_path)
        self._load_data()
        self._filter_data()

    def _get_input_files(self, input_path):

        fns = os.listdir(input_path)
        self.fns = [os.path.join(input_path, fn) for fn in fns if not "lock" in fn]

    def _load_data(self):

        dataframes = []
        for fn in self.fns:
            df = pd.read_csv(fn)
            df["fn"] = os.path.split(fn)[-1]
            go_ids = []
            pathways = []
            for g in df["Unnamed: 0"]:
                g1 = f"GO:{g[2:9]}"
                go_ids.append(g1)
                p = g.split(":")[1][1:]
                pathways.append(p)
            df["go_id"] = go_ids
            df["pathway"] = pathways
            dataframes.append(df.copy())

        self.df_full = pd.concat(dataframes, axis=0)
        self.df_full = self.df_full.sort_values(["FDR"])
        self.df_full = self.df_full.reset_index(None)

    def _filter_data(self):

        # save_fn can be used ot save results as csv for rrvgo analysis
        data = {
            "go_id": [],
            "pathway": [],
            "FDR": [],
            "direction": [],
            "delta": [],
            "ngenes": [],
            "FDR_resilience": [],
        }
        for g, p, f, d, dl, n, filename in zip(
                self.df_full.go_id.values,
                self.df_full["Unnamed: 0"].values,
                self.df_full.FDR.values,
                self.df_full.Direction.values,
                self.df_full.delta.values,
                self.df_full.NGenes.values,
                self.df_full.fn.values
        ):
            if f > self.FDR_cutoff:
                continue
            if n < self.min_genes or n > self.max_genes:
                continue

            if g in data["go_id"]:
                j = np.where(np.array(data["go_id"]) == g)[0][0]
                if f < data["FDR"][j]:
                    data["FDR"][j] = f
                if f < data["FDR_resilience"][j] and "resilience" in filename.lower():
                    data["FDR_resilience"][j] = f
            else:
                data["go_id"].append(g)
                data["pathway"].append(p.split(":")[1][1:])
                data["FDR"].append(f)
                data["FDR_resilience"].append(f if "resilience" in filename.lower() else 1.0)
                data["direction"].append(d)
                data["delta"].append(dl)
                data["ngenes"].append(n)

        self.df_filtered = pd.DataFrame(data=data)
        self.df_filtered = figure_utils.remove_bad_terms( self.df_filtered)
        self.df_filtered =  self.df_filtered.sort_values(["FDR"])
        self.df_filtered =  self.df_filtered.reset_index(None)

    def save_filtered_data(self, save_fn):
        self.df_filtered.to_csv(save_fn)

    def _generate_data_time_pts(self, pathways):

        df_full = self.df_full[self.df_full.pathway.isin(pathways)]
        fns = df_full.fn.unique()
        n_time_pts = len(fns)
        n_paths = len(pathways)

        self.fdr = np.zeros((n_paths, n_time_pts)) # [pathway, time pts]
        self.z_score = np.zeros((n_paths, n_time_pts))

        for i, p in enumerate(pathways):
            df_rows = df_full[df_full.pathway == p]
            df_rows = df_rows.reset_index()
            for n in range(n_time_pts):
                for j in range(len(df_rows)):
                    if f"time{n+1}.csv" in df_rows.loc[j].fn:
                        self.fdr[i, n] = - np.sign(df_rows.loc[j].delta) * np.log10(df_rows.loc[j].FDR)
                        self.z_score[i, n] = df_rows.loc[j].delta / df_rows.loc[j].se


    def _generate_dataframes_time_pts(
        self,
        pathways,
        time_pts = None,
        cluster_pathways = False,
        n_interp_pts = 100,
    ):

        n_paths = len(pathways)
        new_pathways = []
        for p in pathways:
            new_pathways.append(figure_utils.condense_pathways(p))

        if cluster_pathways:
            Z = linkage(np.reshape(self.z_score, (self.z_score.shape[0], -1)), method='ward')
            s = dendrogram(Z, no_plot=True)
            idx = [s["leaves"][i] for i in range(n_paths)]
            new_pathways = [new_pathways[i] for i in idx]
            self.z_score = self.z_score[idx]
            self.fdr = self.fdr[idx]

        time_pts = self.fdr.shape[1] if time_pts is None else time_pts
        t_new = np.linspace(time_pts[0], time_pts[-1], n_interp_pts)
        n_paths = len(pathways)

        fdr_interp = np.zeros((n_paths, n_interp_pts))
        z_score_interp = np.zeros((n_paths, n_interp_pts))

        for n in range(n_paths):
            fdr_interp[n, :] = np.interp(t_new, time_pts, self.fdr[n, :])
            z_score_interp[n, :] = np.interp(t_new, time_pts, self.z_score[n, :])

        self.df_fdr = pd.DataFrame(fdr_interp, index=new_pathways, columns=t_new)
        self.df_z_score = pd.DataFrame(z_score_interp, index=new_pathways, columns=t_new)
        self.time_pts_interp = t_new


    def _generate_early_late_data(
        self,
        pathways,
        cell_names = ["EN", "IN", "Astro", "Immune", "Oligo", "OPC", "Mural", "Endo"],
    ):

        df_full = self.df_full[self.df_full.pathway.isin(pathways)]

        n_paths = len(pathways)
        n_cells = len(cell_names)
        self.fdr = np.zeros((n_paths, 2, 2, n_cells)) # [pathway, early/late, Braak/Resilience, Cell]
        self.z_score = np.zeros((n_paths, 2, 2, n_cells))

        for i, p in enumerate(pathways):
            # df_row = df[df.pathway == p]
            df_full_rows = df_full[df_full.pathway == p]
            df_full_rows = df_full_rows.reset_index()
            for n0, c in enumerate(cell_names):
                for n1, s in enumerate(["braak", "resilience"]):
                    for j in range(len(df_full_rows)):
                        if s in df_full_rows.loc[j].fn.lower() and c in df_full_rows.loc[j].fn:
                            m = 0 if "early" in df_full_rows.loc[j].fn.lower() else 1
                            sign = -1 if "resilience" in df_full_rows.loc[j].fn.lower() else 1
                            self.fdr[i, m, n1, n0] = - np.sign(df_full_rows.loc[j].delta) * np.log10(df_full_rows.loc[j].FDR)
                            self.z_score[i, m, n1, n0] = sign * df_full_rows.loc[j].delta / df_full_rows.loc[j].se


    def _generate_early_late_dataframes(
        self,
        pathways,
        cell_names = ["EN", "IN", "Astro", "Immune", "Oligo", "OPC", "Mural", "Endo"],
        cluster_pathways = False,
    ):

        n_paths = len(pathways)
        new_pathways = []
        for p in pathways:
            new_pathways.append(figure_utils.condense_pathways(p))

        if cluster_pathways:
            Z = linkage(np.reshape(self.z_score, (self.z_score.shape[0], -1)), method='ward')
            s = dendrogram(Z, no_plot=True)
            idx = [s["leaves"][i] for i in range(n_paths)]
            new_pathways = [new_pathways[i] for i in idx]
            self.z_score = self.z_score[idx]
            self.fdr = self.fdr[idx]

        self.df_fdr_early = pd.DataFrame(
            data=np.reshape(self.fdr[:, 0], (n_paths, -1)), index=new_pathways, columns=2 * cell_names,
        )
        self.df_zscore_early = pd.DataFrame(
            data=np.reshape(self.z_score[:, 0], (n_paths, -1)), index=new_pathways, columns=2 * cell_names,
        )
        self.df_fdr_late = pd.DataFrame(
            data=np.reshape(self.fdr[:, 1], (n_paths, -1)), index=new_pathways, columns=2 * cell_names,
        )
        self.df_zscore_late = pd.DataFrame(
            data=np.reshape(self.z_score[:, 1], (n_paths, -1)), index=new_pathways, columns=2 * cell_names,
        )

    def generate_sliding_window_figure(
        self,
        pathways,
        time_pts=None,
        pred_braak_traj=None,
        transition_pts=None,
        cluster_pathways=False,
        figsize=(6.0, 6.0),
        save_fig_fn=None,
    ):

        self._generate_data_time_pts(pathways)
        self._generate_dataframes_time_pts(pathways, time_pts=time_pts, cluster_pathways=cluster_pathways)

        f, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
        max_val = 8
        sns.heatmap(
            self.df_z_score,
            ax=ax,
            cmap='RdBu_r',
            center=True,
            vmin=-max_val,
            vmax=max_val,
            cbar=False,
            xticklabels=False,
        )


        if transition_pts is not None and pred_braak_traj is not None:
            t0, t1 = transition_pts[0], transition_pts[1]
            t0 = int(t0 * len(pred_braak_traj))
            t1 = int(t1 * len(pred_braak_traj))
            t0 = np.argmin(np.abs(self.time_pts_interp - pred_braak_traj[t0]))
            t1 = np.argmin(np.abs(self.time_pts_interp - pred_braak_traj[t1]))
            ax.vlines([t0], 0, len(pathways), linewidth=0.5, colors='k')
            ax.vlines([t1], 0, len(pathways), linewidth=0.5, colors='k')


        # ax[0].hlines([n_protective], 0, 100, linewidth=0.5, colors='k')
        # ax[1].hlines([n_protective], 0, 100, linewidth=0.5, colors='k')

        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel("Disease pseudotime")
        # ax[0].set_xticksminor([t0, t1])
        # ax[0].set_xticks([ 0t0, t1], ["0", "2.33", "2.98", "99"])
        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)


    def generate_early_late_figure(
        self,
        pathways,
        cell_names = ["EN", "IN", "Astro", "Immune", "Oligo", "OPC", "Mural", "Endo"],
        cluster_pathways = False,
        figsize=(6.0, 6.0),
        save_fig_fn = None,
    ):

        self._generate_early_late_data(pathways, cell_names)
        self._generate_early_late_dataframes(pathways, cell_names, cluster_pathways=cluster_pathways)

        # fig, ax = plt.subplots(figsize=(13, 12))

        # fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        m = 10

        # Create colormap
        cmap0 = cm.get_cmap('RdBu_r')  # replace with your preferred colormap
        cmap1 = cm.get_cmap('PiYG')  # replace with your preferred colormap

        norm = Normalize(vmin=-m, vmax=m)

        n_paths = self.df_fdr_early.shape[0]
        n_cells = self.df_fdr_early.shape[1]

        # Iterate over each cell
        for i in range(n_paths):
            for j in range(n_cells):

                cmap = cmap0 if j < 8 else cmap1
                # Create polygon coordinates
                polygon_coords = [(j, i), (j + 1, i), (j, i + 1), (j + 1, i + 1)]

                # Create two polygons for each half of the cell
                upper_triangle = patches.Polygon(polygon_coords[:3], closed=True, lw=0.2,
                                                 facecolor=cmap(norm(self.df_zscore_early.iloc[i, j])))
                lower_triangle = patches.Polygon(polygon_coords[1:], closed=True, lw=0.2,
                                                 facecolor=cmap(norm(self.df_zscore_late.iloc[i, j])))
                ax.add_patch(upper_triangle)
                ax.add_patch(lower_triangle)

                if np.abs(self.df_fdr_early.iloc[i, j]) > -np.log10(0.05):
                    ax.text(j + 0.33, i + 0.4, "*", ha="center", va="center", color="w", fontsize=6)
                if np.abs(self.df_fdr_late.iloc[i, j]) > -np.log10(0.05):
                    ax.text(j + 0.66, i + 0.8, "*", ha="center", va="center", color="w", fontsize=6)

        ax.set_xlim(0, n_cells)
        ax.set_ylim(0, n_cells)
        ax.set_xticks(np.arange(n_cells) + 0.5, minor=False)
        ax.set_yticks(np.arange(n_paths) + 0.5, minor=False)
        ax.set_xticklabels(self.df_zscore_early.columns, minor=False, rotation=-45, fontsize=9, ha="left")
        ax.set_yticklabels(self.df_zscore_early.index, minor=False, fontsize=8)

        ax.vlines([len(cell_names)], 0, 86, linewidth=1, colors='k')
        ax.vlines([2 * len(cell_names)], 0, 86, linewidth=1, colors='k')
        plt.setp(ax.get_yticklabels(), fontsize=7)
        ax.set_title("Disease pseudotime     Resilience", fontsize=9)

        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)
        plt.show()


def output_zenith(slope_dict, save_fn, genes):
    """Out a csv of genes with associtaed scores as input for Zenith gene enrichment"""

    data = {"genes": []}
    for k in slope_dict.keys():
        data[k] = []

    for i in range(len(genes)):
        data["genes"].append(genes[i])
        for k, v in slope_dict.items():
            data[k].append(v[i])

    df = pd.DataFrame(data=data)
    df.to_csv(save_fn)

def calculate_slopes(time_pts, pred_braak_traj, genes_braak_traj, res):

    n_time, n_genes = genes_braak_traj.shape

    m = len(time_pts)
    slopes = np.zeros((m, n_genes))
    resilience = np.zeros((m, n_genes))
    explained_var = np.zeros((m, n_genes))

    for i, break_point in enumerate(time_pts):
        t0 = break_point[0]
        t1 = break_point[1]
        slopes[i, :], _, _, explained_var[i, :] = fit_single(
            pred_braak_traj[t0:t1], genes_braak_traj[t0:t1, :]
        )
        resilience[i, :] = np.mean(res[t0:t1, :], axis=0)

    return slopes, resilience, explained_var

def calculate_trajectory_residuals(pred_x, pred_y, gene_vals, gamma, neighbors=None):

    """Used to determine whether genes are protective or damaging
    pred_x = e.g. predicted Braak values
    pred_y = e.g. predicted Dementia values"""

    # in case these are actual values
    pred_x[pred_x < -1] = np.nan
    pred_y[pred_y < -1] = np.nan

    resids = []

    for i in range(len(pred_x)):
        delta = (pred_x - pred_x[i]) ** 2
        w = np.exp(-gamma * delta)
        if neighbors is not None:
            idx_min = np.argsort(delta)[:neighbors]
            w *= 0.0
            w[idx_min] = 1.0
        w[i] = 0
        w /= np.sum(w)
        local_y = np.sum(w * pred_y, axis=0)
        local_gene_vals = np.sum(w[:, None] * gene_vals, axis=0)
        y_resid = pred_y[i] - local_y
        gene_resid = gene_vals[i, :] - local_gene_vals
        resids.append(y_resid * gene_resid)

    return  np.stack(resids)

def calculate_trajectories_single(x, y, gamma, neighbors=None):

    # x = e.g. predicted BRAAK or Dementia
    # y = e.g. gene expression
    x_smooth = []
    y_smooth = []

    for i in range(len(x)):
        delta = (x - x[i]) ** 2
        w = np.exp(-gamma * delta)[:, None]
        if neighbors is not None:
            idx_min = np.argsort(delta)[:neighbors]
            w *= 0.0
            w[idx_min] = 1.0
        w /= np.sum(w)

        local_x = np.sum(w[:, 0] * x, axis=0)
        local_y = np.sum(w * y[:, :], axis=0)

        x_smooth.append(local_x)
        y_smooth.append(local_y)

        #resid.append((x[i] - local_x) * (y[i, :] - local_y))

    return np.stack(x_smooth), np.stack(y_smooth)

def fit_piecewise_search(x, y, time_resolution=1, no_jump=False, max_time=None, min_time=None):

    n_time = y.shape[0]
    ex_var = []
    error = []
    max_time = n_time - 2 * time_resolution if max_time is None else max_time
    min_time = 2 * time_resolution if min_time is None else min_time
    y_hat = []

    for t in range(min_time, max_time, time_resolution):
        early_time = np.arange(t)
        late_time = np.arange(t, n_time)
        _, resid, y0 = fit_piecewise(x, y, early_time, late_time, no_jump=no_jump)
        # ex_var.append(np.mean(1 - np.var(resid, axis=0) / np.var(y, axis=0)))
        err = np.sum(np.mean(resid ** 2, axis=0))
        var = np.sum(np.var(y, axis=0))
        ex_var.append(1 - err / var)
        error.append(err)
        y_hat.append(y0)

    n = np.argmax(np.stack(ex_var))
    t = n * time_resolution + min_time

    return t,ex_var[n], error[n], y_hat[n]

def fit_single(x, y):

    n_time, n_features = y.shape
    slopes = np.zeros((n_features), dtype=np.float32)
    resid = np.zeros((n_time, n_features), dtype=np.float32)

    for n in range(n_features):
        curve_coefs, _ = optimize.curve_fit(curve, x, y[:, n])
        slopes[n] = curve_coefs[1]
        y_hat = curve(x, *curve_coefs)
        resid[:, n] = y[:, n] - y_hat

    err = np.sum(np.mean(resid ** 2, axis=0))
    var = np.sum(np.var(y, axis=0))
    ex_var = 1 - err / var

    return slopes, resid, y_hat, ex_var

def curve(x, a0, a1):
    return a0 + a1 * x

def curve_no_int(x, a0):
    return a0 * x

def fit_piecewise(x, y, idx_early, idx_late, no_jump=False, n_bootstrap=None):

    # no_jump ensure that the piece wise fit doesn't jump at the transition point
    n_time, n_features = y.shape
    time0 = x[idx_early]
    time1 = x[idx_late]

    slopes = np.zeros((2, n_features), dtype=np.float32)
    resid = np.zeros((n_time, n_features), dtype=np.float32)
    y_hat = np.zeros((n_time, n_features), dtype=np.float32)

    for n in range(n_features):

        curve_coefs, _ = optimize.curve_fit(curve, time0, y[idx_early, n])
        slopes[0, n] = curve_coefs[1]
        y1 = curve(time0, *curve_coefs)
        resid[idx_early, n] = y[idx_early, n] - y1
        y_hat[idx_early, n] = y1

        # if no_jump is True, will be ensure there's no jump at the transition
        baseline = curve(time1[0], *curve_coefs) if no_jump else 0
        t0 = time1[0] if no_jump else 0
        curve_fn = curve_no_int if no_jump else curve

        curve_coefs, _ = optimize.curve_fit(curve_fn, time1 - t0, y[idx_late, n] - baseline)
        y1 = curve_fn(time1 - t0, *curve_coefs) + baseline
        slopes[1, n] = curve_coefs[0]
        resid[idx_late, n] = y[idx_late, n] - y1
        y_hat[idx_late, n] = y1

    return slopes, resid, y_hat

def process_zenith_dfs(zenith_fns, paths_per_df=5):
    dataframes = []

    for fn in zenith_fns:
        df = pd.read_csv(fn)

        data = {"pathway": [], "Direction": [], "FDR": [], "ngenes": []}
        for p, d, f, n in zip(df["Unnamed: 0"].values, df["Direction"].values, df["FDR"].values, df["NGenes"]):
            j = p.find(":")
            p = p[j + 1:]
            if n < 10 or n > 200:
                continue
            include = True
            for b in figure_utils.bad_path_words:
                if b in p:
                    include = False
            if not include:
                continue

            data["pathway"].append(p)
            data["Direction"].append(d)
            data["FDR"].append(np.float32(f))
            data["ngenes"].append(n)

        df = pd.DataFrame(data=data)
        df0 = df[df.Direction == "Up"]
        df0 = df0[:paths_per_df]
        df1 = df[df.Direction == "Down"]
        df1 = df1[:paths_per_df]
        df0 = df0.drop(["Direction"], axis=1)
        df1 = df1.drop(["Direction"], axis=1)

        dataframes.append(df0)
        dataframes.append(df1)

    return dataframes

def process_magma_dfs(magma_fns):
    dataframes = []
    scores = []
    names = []

    for fn in magma_fns:
        df, titles = get_magma_results(fn)
        dataframes.append(df)

    scores = np.zeros((len(titles), len(magma_fns)))
    for i, df in enumerate(dataframes):
        for n in range(len(titles)):
            scores[n, i] = np.float32(df["P"].values[n])

    return scores, titles

def get_magma_results(magma_fn):

    df = pd.read_csv(magma_fn)
    data = {"GWAS": [], "NGENES": [], "P": []}
    data = {"GWAS": [], "P": []}
    # print(df.Gwas.values)
    titles = []
    for k in figure_utils.gwas_dict.keys():
        for g, n, p in zip(df.Gwas.values, df.NGENES.values, df.P.values):
            if k == g:
                data["GWAS"].append(g)
                data["P"].append("{:.2e}".format(p))
                titles.append(figure_utils.gwas_dict[g])
                # data["NGENES"].append(n)

    return pd.DataFrame(data), titles

def plot_magma_results(scores, titles, ax, fig):
    fs = 7
    N = scores.shape[1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.02)
    im = ax.imshow(-np.log10(scores), clim=(0, 5), aspect="auto")
    labels = ["Early inc.", "Early dec.", "Late inc.", "Late dec."]
    if N > 4:
        labels += ["Early protect.", "Early damag.", "Late protect.", "Late damag."]

    ax.set_xticks(np.arange(N), labels=labels, rotation=-45, fontsize=fs, ha="left")
    ax.set_yticks(np.arange(len(titles)), labels=titles, fontsize=fs)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=(0, 5))
    cbar.ax.tick_params(labelsize=fs)
    # fig.colorbar(im, ax=ax)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    for i in range(len(titles)):
        for j in range(N):
            if scores[i, j] < 0.001:
                text = "*"
            elif scores[i, j] < 0.01:
                text = "*"
            elif scores[i, j] < 0.05:
                text = "*"
            else:
                text = ""
            text = ax.text(j, i, text, ha="center", va="center", color="w", fontsize=6)

def plot_zenith_results(dataframes, ax, fig, paths_per_df=6):
    fs = 6
    suffix = ["Early increase", "Early decrease", "Late increase", "Late decrease"]
    pathways = []
    scores = []
    count = 0
    ax.plot([-np.log10(0.05), -np.log10(0.05)], [0.5 - paths_per_df * 4, 0.5], 'k--')
    # ax.plot([-np.log10(0.01), -np.log10(0.01)], [0.5-paths_per_df * 4, 0.5], 'k--')
    ax.set_xlabel("-log10 FDR", fontsize=fs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.02)
    ng = []
    for m, df in enumerate(dataframes):
        for n in range(paths_per_df):
            ngenes = np.minimum(199, df["ngenes"].values[n])
            ng.append(ngenes)
    max_genes = np.max(ng)
    norm = matplotlib.colors.Normalize(vmin=10.0, vmax=max_genes)

    for m, df in enumerate(dataframes):
        for n in range(paths_per_df):
            ngenes = np.minimum(199, df["ngenes"].values[n])
            p = df["pathway"].values[n]
            p = figure_utils.condense_pathways(p)
            pathways.append(p)
            scores.append(df["FDR"].values[n])
            rgba_color = np.array(plt.cm.viridis(norm(ngenes), bytes=True)).reshape(-1, 4) / 255
            bar = ax.barh(-count, - np.log10(df["FDR"].values[n]), label=suffix[m], color=rgba_color)
            count += 1
    scores = np.array(scores)[:, None]
    scores = np.tile(scores, (1, 2))

    # cmap = plt.imshow(np.reshape(ng, (5, -1)), clim=(10, 60))
    ax.set_yticks(np.arange(0, - paths_per_df * 4, -1), pathways, fontsize=fs)
    ax.set_xlim([0, 10])
    ax.set_xticks([0, 10])
    # ax.set_yticks(np.arange(len(pathways)), labels=pathways)
    for n, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(colors[n // paths_per_df])

    ax.yaxis.set_tick_params(length=0)
    ax.set_ylim([0.25 - paths_per_df * 4, 0.75])
    # fig.colorbar(bar, cax=cax, orientation='vertical', ticks=[10, 60])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation='vertical',
                        ticks=[10, max_genes])
    cbar.ax.tick_params(labelsize=fs)

def exclusive_gene_lists(slopes, top_k=250, exclusive=True):
    idx = []
    idx.append(np.argsort(slopes[0, :])[::-1])
    idx.append(np.argsort(slopes[0, :]))
    idx.append(np.argsort(slopes[1, :])[::-1])
    idx.append(np.argsort(slopes[1, :]))
    gene_list = [[] for _ in range(4)]

    excluded_genes = []
    for n in range(2 * top_k):
        for i in range(4):
            if not idx[i][n] in excluded_genes and len(gene_list[i]) < top_k:
                gene_list[i].append(idx[i][n])
                if exclusive:
                    excluded_genes.append(idx[i][n])
    return gene_list