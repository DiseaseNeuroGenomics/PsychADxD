from typing import Optional
import os, sys
import pickle
import copy
from constants import bad_path_words
import scanpy as sc
import anndata as ad
import copy
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
matplotlib.rcParams['pdf.fonttype'] = 42



gwas_dict = {
    "alzBellenguezNoApoe": "AD_2022_Bellenguez",
    "ms": "MS_2019_IMSGC",
    "pd_without_23andMe": "PD_2019_Nalls",
    "migraines_2021": "Migraines_2021_Donertas",
    "als2021": "ALS_2021_vanRheenen",
    "stroke": "Stroke_2018_Malik",
    "epilepsyFocal": "Epilepsy_2018_ILAECCE",

    "sz3": "SCZ_2022_Trubetskoy",
    "bip2": "BD_2021_Mullins",
    "asd": "ASD_2019_Grove",
    "adhd_ipsych": "ADHD_2023_Demontis",
    "mdd_ipsych": "MDD_2023_AlsBroad",
    "ocd": "OCD_2018_IOCDF_GC",
    "insomn2": "Insomnia_2019_Jansen",
    "alcohilism_2019": "Alcoholism_2019_SanchezRoige",
    "tourette": "Tourettes_2019_Yu",
    "intel": "IQ_2018_Savage",
    "eduAttainment": "Education_2018_Lee",
}

bad_path_words = [
    "cardiac",
    "pulmonary",
    "vocalization",
    "auditory stimulus",
    "learning",
    "memory",
    "social behavior",
    "locomotory",
    "nervous system process",
    "recognition",
    "forelimb",
    "hindlimb",
    "startle",
    "dosage",
    "substantia nigra development",
    "retina",
    "optic",
]

meta_fn = "/home/masse/work/data/mssm_rush/metadata_slim.pkl"
meta = pickle.load(open(meta_fn, "rb"))
gene_names = meta["var"]["gene_name"]
sex_chrom = (meta["var"]["gene_chrom"] == "X") + (meta["var"]["gene_chrom"] == "Y")
ribosomal = meta["var"]["ribosomal"]
mitochondrial = meta["var"]["mitochondrial"]
protein_coding = meta["var"]["protein_coding"]
protein_coding_genes = gene_names[protein_coding]


gene_dict = {"gene_convert" : {}}
for g, gene_id in zip(meta["var"]["gene_name"], meta["var"]["gene_id"]):
    gene_dict["gene_convert"][g] = gene_id


def output_zenith(
        pred_braak,
        pred_dementia,
        traj_braak,
        traj_dementia,
        traj_resilience,
        time_cutoff,
        save_fn,
        protein_coding_only=True,
        non_protein_coding_only=False,
        threshold=0.01,
        n_bootstrap=None,
):
    assert not (protein_coding_only and non_protein_coding_only), "only one can be true"

    idx = np.argsort(pred_braak)
    idx_early = idx[:time_cutoff]
    idx_late = idx[time_cutoff:]

    slopes_br, resid_br, _ = fit_piecewise(pred_braak, traj_braak, idx_early, idx_late)
    slopes_dm, resid_dm, _ = fit_piecewise(pred_dementia, traj_dementia, idx_early, idx_late)

    mean_exp = np.mean(traj_braak, axis=0)

    r = np.mean(traj_resilience[:, :], axis=0)
    r_early = np.mean(traj_resilience[idx_early, :], axis=0)
    r_late = np.mean(traj_resilience[idx_late, :], axis=0)

    create_zenith_df(
        copy.deepcopy(-r),
        copy.deepcopy(-r_early),
        copy.deepcopy(-r_late),
        copy.deepcopy(slopes_br),
        copy.deepcopy(slopes_dm),
        mean_exp,
        save_fn,
        protein_coding_only=protein_coding_only,
        non_protein_coding_only=non_protein_coding_only,
        threshold=threshold,
    )


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x


def subject_averaged_scores(x, y, subject):
    x_sub = []
    y_sub = []

    for s in np.unique(subject):
        idx = np.where(subject == s)[0]
        if len(idx) == 0:
            continue
        x_sub.append(np.mean(x[idx]))
        y_sub.append(np.mean(y[idx]))

    return np.stack(x_sub), np.stack(y_sub)


def explained_var(x_pred, x_real, discrete=False):
    idx = ~np.isnan(x_real) * (x_real > -99)
    x = x_real[idx]
    if discrete:
        n = x_pred.shape[1]
        w = np.arange(n)
        y_pred_cont = np.sum(w[None, :] * x_pred, axis=1)
        y = y_pred_cont[idx]
    else:
        y = x_pred[idx]
        ex_var = 1 - np.nanvar(x - y) / np.nanvar(x)
        r, _ = stats.pearsonr(x, y)

    return ex_var, r


def braak_score_boot(x_pred, x_real, n_bootstrap=1_000):
    idx = ~np.isnan(x_real) * (x_real > -99)
    x_real = x_real[idx]
    x_pred = x_pred[idx]
    acc = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x_pred), size=len(x_pred), replace=True)
        r, _ = stats.pearsonr(x_real[idx], x_pred[idx])
        acc.append(r)
    acc = np.stack(acc)
    return np.mean(acc), np.std(acc), np.mean(acc <= 0)


def classification_score(x_pred, x_real):
    idx = ~np.isnan(x_real) * (x_real > -99)
    s0 = np.sum((x_real[idx] == 0) * (x_pred[idx] < 0.5)) / np.sum(x_real[idx] == 0)
    s1 = np.sum((x_real[idx] > 0.99) * (x_pred[idx] >= 0.5)) / np.sum(x_real[idx] > 0.99)
    return (s0 + s1) / 2


def dementia_score_boot(x_pred, x_real, n_bootstrap=1_000):
    idx = ~np.isnan(x_real) * (x_real > -99)
    x_real = x_real[idx]
    x_pred = x_pred[idx]
    acc = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x_pred), size=len(x_pred), replace=True)
        s0 = np.sum((x_real[idx] == 0) * (x_pred[idx] < 0.5)) / np.sum(x_real[idx] == 0)
        s1 = np.sum((x_real[idx] > 0.99) * (x_pred[idx] >= 0.5)) / np.sum(x_real[idx] > 0.99)
        acc.append((s0 + s1) / 2)
    acc = np.stack(acc)
    return np.mean(acc), np.std(acc), np.mean(acc <= 0.5)


def create_zenith_df(
        resilience,
        resilience_early,
        resilience_late,
        slopes_br,
        slopes_dm,
        mean_exp,
        save_fn,
        protein_coding_only=True,
        non_protein_coding_only=False,
        threshold=0.01,
):
    if len(resilience) == len(gene_names):
        pc = copy.deepcopy(protein_coding)
        gn = copy.deepcopy(gene_names)
    elif len(resilience) == len(gene_names_pxr):
        pc = copy.deepcopy(protein_coding_pxr)
        gn = copy.deepcopy(gene_names_pxr)

    data = {
        "genes": [],
        "resilience": [],
        "resilience_early": [],
        "resilience_late": [],
        "early_braak": [],
        "late_braak": [],
        "braak": [],
        "early_dementia": [],
        "late_dementia": [],
        "dementia": [],
    }

    for i in range(len(gn)):
        if not pc[i] and protein_coding_only:
            continue
        elif pc[i] and non_protein_coding_only:
            continue
        if mean_exp[i] < threshold:
            continue

        if gn[i] in data["genes"]:
            continue

        data["genes"].append(gn[i])
        data["resilience"].append(resilience[i])
        data["resilience_early"].append(resilience_early[i])
        data["resilience_late"].append(resilience_late[i])
        data["early_braak"].append(slopes_br[0, i])
        data["late_braak"].append(slopes_br[1, i])
        data["braak"].append(slopes_br[2, i])
        data["early_dementia"].append(slopes_dm[0, i])
        data["late_dementia"].append(slopes_dm[1, i])
        data["dementia"].append(slopes_dm[2, i])

    df = pd.DataFrame(data=data)
    df.to_csv(save_fn)


def get_magma_results(magma_fn):
    magma_dir = "/home/masse/work/mssm/psuedotime/magma_output_0319"
    fn = os.path.join(magma_dir, magma_fn)
    df = pd.read_csv(fn)
    data = {"GWAS": [], "NGENES": [], "P": []}
    data = {"GWAS": [], "P": []}
    # print(df.Gwas.values)
    titles = []
    for k in gwas_dict.keys():
        for g, n, p in zip(df.Gwas.values, df.NGENES.values, df.P.values):
            if k == g:
                data["GWAS"].append(g)
                data["P"].append("{:.2e}".format(p))
                titles.append(gwas_dict[g])
                # data["NGENES"].append(n)

    return pd.DataFrame(data), titles


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
            for b in bad_path_words:
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


def condense_pathways(pathway):
    pathway = pathway.split()

    for n, p in enumerate(pathway):
        p = copy.deepcopy(p.lower())
        if p == "positive":
            pathway[n] = "pos."
        elif p == "negative":
            pathway[n] = "neg."
        elif p == "regulation":
            pathway[n] = "reg."
        elif p == "response":
            pathway[n] = "resp."
        elif p == "neurotransmitter":
            pathway[n] = "neurotrans."
        elif p == "neurotransmitter":
            pathway[n] = "neurotrans."
        elif p == "modulation":
            pathway[n] = "mod."
        elif p == "differentiation":
            pathway[n] = "diff."
        elif p == "biosynthetic":
            pathway[n] = "biosynth."
        elif p == "mitochondrial":
            pathway[n] = "mito."
        elif p == "nitric-oxide":
            pathway[n] = "NO"
        elif p == "glutamate":
            pathway[n] = "GLU"
        elif p == "glutamatergic":
            pathway[n] = "GLU"
        elif p == "homeostasis":
            pathway[n] = "homeo."
        elif p == "signaling":
            pathway[n] = "sign."
        elif p == "exocytosis":
            pathway[n] = "exocyt."
        elif p == "colony-stimulating":
            pathway[n] = "colony-stim."
        elif p == "derived":
            pathway[n] = "der."
        elif p == "multicellular":
            pathway[n] = "multicell."
        elif p == "presentation":
            pathway[n] = "pres."
        elif p == "organismal-level":
            pathway[n] = "org.-level"
        elif p == "proliferation":
            pathway[n] = "prolif."
        elif p == "homeostasis":
            pathway[n] = "homeo."
        elif p == "t-helper":
            pathway[n] = "T-help."
        elif p == "immune":
            pathway[n] = "imm.."
        elif p == "helper":
            pathway[n] = "help."
        elif p == "ligand-gated":
            pathway[n] = "lig.-gated"
        elif p == "macrophage":
            pathway[n] = "macrophg."
        elif p == "associated":
            pathway[n] = "ass."
        elif p == "transport":
            pathway[n] = "trans."
        elif p == "synthesis":
            pathway[n] = "synth."
        elif p == "contraction":
            pathway[n] = "contract."
        elif p == "migration":
            pathway[n] = "migrat."
        elif p == "receptor":
            pathway[n] = "recept."
        elif p == "nucleotide":
            pathway[n] = "nucleot."
        elif p == "pathway":
            pathway[n] = "path."
        elif p == "complex":
            pathway[n] = "comp."
        elif p == "protein":
            pathway[n] = "prot."
        elif p == "assembly":
            pathway[n] = "assemb."
        elif p == "organization":
            pathway[n] = "org."
        elif p == "membrane":
            pathway[n] = "memb."
        elif p == "transmission":
            pathway[n] = "trans."
        elif p == "transmission,":
            pathway[n] = "trans.,"
        elif p == "stimulus":
            pathway[n] = "stim."
        elif p == "mineralocorticoid":
            pathway[n] = "mineralcort."
        elif p == "chaperone-mediated":
            pathway[n] = "chaperone-med."
    for n in range(len(pathway) - 1):
        if pathway[n] == "calcium" and pathway[n + 1] == "ion":
            pathway[n] = "Ca2+"
            pathway[n + 1] = ""
        elif pathway[n] == "iron" and pathway[n + 1] == "ion":
            pathway[n] = "Fe2+"
            pathway[n + 1] = ""
        elif pathway[n] == "manganese" and pathway[n + 1] == "ion":
            pathway[n] = "Mn2+"
            pathway[n + 1] = ""
        elif pathway[n] == "calcium" and "ion-" in pathway[n + 1]:
            pathway[n] = "Ca2+ " + pathway[n + 1][4:]
            pathway[n + 1] = ""
        elif pathway[n] == "G" and "protein-coupled" in pathway[n + 1]:
            pathway[n] = "GPC"
            pathway[n + 1] = ""

    pathway = " ".join(pathway)
    pathway = pathway.split()
    return " ".join(pathway)


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
            p = condense_pathways(p)
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


def plot_gene_scores_specified(
        slopes,
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        resilience,
        ax,
        fig,
        top_k=5,
        normalize=True,
        interpolate_traj=True,
        edge_num_exclude=10,
):
    idx = np.argsort(pred_braak)
    if edge_num_exclude > 0:
        idx = idx[edge_num_exclude:-edge_num_exclude]
    pred_braak_sorted = pred_braak[idx]
    braak_traj_sorted = braak_traj[idx]
    resilience_sorted = resilience[idx]

    print("R", resilience_sorted.shape, braak_traj_sorted.shape)

    fs = 7
    idx = []
    gene_list = exclusive_gene_lists(slopes, top_k=250)
    for i in range(4):
        idx += gene_list[i][:top_k]

    print("GENES")
    print(idx)

    print(f"Number of unique genes: {len(np.unique(idx))}")

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
    genes = gene_names[idx].tolist()
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


def plot_mean_traj(slopes, preds, traj, top_k, xlabel, ax):
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


def plot_summary_all(
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        zenith_br_fns=None,
        zenith_dm_fns=None,
        magma_br_fns=None,
        magma_dm_fns=None,
        top_k=250,
        savefig_fn=None,
):
    fig = plt.figure(constrained_layout=True, figsize=(0.8 * 10.5, 3 + 0.8 * 11.3))
    gs = gridspec.GridSpec(23, 10, figure=fig)

    bbox = [0, 0, 1, 1]
    ax0 = fig.add_subplot(gs[:6, :3])
    plot_mean_traj(slopes_br, pred_braak, braak_traj, top_k, "Normalized predicted Braak", ax0)

    n = 6
    dataframes = process_zenith_dfs(zenith_br_fns, paths_per_df=n)
    ax1 = fig.add_subplot(gs[:6, 5:7])
    plot_zenith_results(dataframes, ax1, fig, paths_per_df=n)

    magma_scores, magma_titles = process_magma_dfs(magma_br_fns)
    ax1 = fig.add_subplot(gs[:6, 9])
    plot_magma_results(magma_scores, magma_titles, ax1, fig)

    ax1 = fig.add_subplot(gs[7:10, :])
    plot_gene_scores_specified(
        slopes_br,
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        resilience,
        ax1,
        fig,
        top_k=20,
        normalize=True,
        interpolate_traj=True,
    )

    ax0 = fig.add_subplot(gs[12:18, :3])
    plot_mean_traj(slopes_dm, pred_dementia, dementia_traj, top_k, "Predicted Dementia", ax0)

    dataframes = process_zenith_dfs(zenith_dm_fns, paths_per_df=n)
    ax1 = fig.add_subplot(gs[12:18, 5:7])
    plot_zenith_results(dataframes, ax1, fig, paths_per_df=n)

    magma_scores, magma_titles = process_magma_dfs(magma_dm_fns)
    ax1 = fig.add_subplot(gs[12:18, 9])
    plot_magma_results(magma_scores, magma_titles, ax1, fig)

    ax1 = fig.add_subplot(gs[19:, :])
    plot_gene_scores_specified(
        slopes_dm,
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        resilience,
        ax1,
        fig,
        top_k=20,
        normalize=True,
        interpolate_traj=True,
    )

    plt.savefig(savefig_fn)
    plt.show()


def plot_gene_summary(
        slopes,
        preds,
        traj,
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        resilience,
        zenith_fns=None,
        magma_fns=None,
        top_k=250,
        xlabel="Normalized predicted BRAAK",
        savefig_fn=None,
        n_pathways=5,
        top_k_genes=20,
):
    fig = plt.figure(constrained_layout=True, figsize=(10.5, 5))
    gs = gridspec.GridSpec(10, 10, figure=fig)

    bbox = [0, 0, 1, 1]
    ax0 = fig.add_subplot(gs[:6, :3])
    plot_mean_traj(slopes, preds, traj, top_k, xlabel, ax0)

    dataframes = process_zenith_dfs(zenith_fns, paths_per_df=n_pathways)
    ax1 = fig.add_subplot(gs[:6, 5:7])
    plot_zenith_results(dataframes, ax1, fig, paths_per_df=n_pathways)

    magma_scores, magma_titles = process_magma_dfs(magma_fns)
    ax1 = fig.add_subplot(gs[:6, 9])
    plot_magma_results(magma_scores, magma_titles, ax1, fig)

    ax1 = fig.add_subplot(gs[7:, :])

    plot_gene_scores_specified(
        slopes,
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        resilience,
        ax1,
        fig,
        top_k=top_k_genes,
        normalize=True,
        interpolate_traj=True,
    )

    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.25, hspace=0.1, wspace=0.1)
    # plt.tight_layout()
    plt.savefig(savefig_fn)
    plt.show()


def plot_gene_summary_v2(
        slopes,
        preds,
        traj,
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        resilience,
        zenith_fns=None,
        magma_fns=None,
        top_k=250,
        xlabel="Normalized predicted BRAAK",
        savefig_fn=None,
        n_pathways=5,
        top_k_genes=20,
):
    fig = plt.figure(constrained_layout=True, figsize=(10.5, 5.5))
    gs = gridspec.GridSpec(11, 11, figure=fig)

    bbox = [0, 0, 1, 1]
    ax0 = fig.add_subplot(gs[:6, :3])
    plot_mean_traj(slopes, preds, traj, top_k, xlabel, ax0)

    dataframes = process_zenith_dfs(zenith_fns[:2], paths_per_df=n_pathways)
    ax1 = fig.add_subplot(gs[:6, 5:7])
    plot_zenith_results(dataframes, ax1, fig, paths_per_df=n_pathways)

    dataframes = process_zenith_dfs(zenith_fns[2:], paths_per_df=n_pathways)
    ax1 = fig.add_subplot(gs[:6, 9:11])
    plot_zenith_results(dataframes, ax1, fig, paths_per_df=n_pathways)

    magma_scores, magma_titles = process_magma_dfs(magma_fns)
    ax1 = fig.add_subplot(gs[7:, 1:3])
    plot_magma_results(magma_scores, magma_titles, ax1, fig)

    ax1 = fig.add_subplot(gs[7:, 3:])

    plot_gene_scores_specified(
        slopes,
        slopes_br,
        slopes_dm,
        pred_braak,
        pred_dementia,
        braak_traj,
        dementia_traj,
        resilience,
        ax1,
        fig,
        top_k=top_k_genes,
        normalize=True,
        interpolate_traj=True,
    )

    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.25, hspace=0.1, wspace=0.1)
    # plt.tight_layout()
    plt.savefig(savefig_fn)
    plt.show()


def output_trajectories(
        cell_type,
        model_fn,
        time_cutoff,
        n_edge_exclude=10,
        alpha=4,
        min_cell_count=5,
        n_boostrap=None,
):

    gene_type = "gene_means"
    adata = sc.read_h5ad(model_fn)

    idx = (
        ~np.isnan(np.sum(adata.uns["donor_gene_means"], axis=1)) *
        ~np.isnan(adata.uns["donor_pred_BRAAK_AD"]) *
        (adata.uns["donor_cell_count"] >= min_cell_count)
    )

    gamma0 = alpha / np.var(adata.uns["donor_pred_BRAAK_AD"][idx])
    gamma1 = alpha / np.var(adata.uns["donor_pred_Dementia"][idx])

    pred_braak, pred_dementia, braak_traj, dementia_traj, resilience_traj, resid_dementia = traj_wrapper(
        copy.deepcopy(adata.uns["donor_gene_means"][idx]),
        copy.deepcopy(adata.uns["donor_pred_BRAAK_AD"][idx]),
        copy.deepcopy(adata.uns["donor_pred_Dementia"][idx]),
        gamma0,
        gamma1,
        neighbors=None,
        gene_type=gene_type,
        normalize_counts=False,
        log_counts=False,
        calculate_residual=True,
        log_prior=1.0,
        n_edge_exclude=n_edge_exclude,
    )

    # sort based on predicted Braak
    idx = np.argsort(pred_braak)
    idx_early = idx[:time_cutoff]
    idx_late = idx[time_cutoff:]

    slopes_br, resid_br, _ = fit_piecewise(pred_braak, braak_traj, idx_early, idx_late)
    slopes_dm, resid_dm, _ = fit_piecewise(pred_dementia, dementia_traj, idx_early, idx_late)

    slopes_br[:, ~protein_coding] = 0.0
    slopes_dm[:, ~protein_coding] = 0.0

    resilience_traj[:, ~protein_coding] = 0.0
    resilience = np.zeros_like(slopes_br)
    resilience[0, :] = np.mean(resilience_traj[idx_early, :], axis=0)
    resilience[1, :] = np.mean(resilience_traj[idx_late, :], axis=0)

    return pred_braak, pred_dementia, braak_traj, dementia_traj, resilience_traj, slopes_br, slopes_dm, resilience


def curve(x, a0, a1):
    return a0 + a1 * x

class StepwiseFit:

    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, x, y):
        r0 = stats.linregress(x[x < self.threshold], y[x < self.threshold])
        r1 = stats.linregress(x[x >= self.threshold], y[x >= self.threshold])
        IN_log_norm_full_20k_glc1_simple_drop_no_domain_ep2_3
        self.f0 = lambda u: r0[0] * u + r0[1]
        self.f1 = lambda u: r1[0] * u + r1[1]

    def map(self, x):
        y = np.zeros_like(x)
        idx0 = x < self.threshold
        idx1 = x >= self.threshold
        y[idx0] = self.f0(x[idx0])
        y[idx1] = self.f1(x[idx1])
        return y


def piecewise_search_wrapper(x, y):
    n_time, n_genes = y.shape
    ex_var_single = np.zeros(n_genes)
    ex_var_double = np.zeros(n_genes)
    slopes = np.zeros((n_genes, 2))
    cutoff_time = np.zeros(n_genes)

    for n in range(n_genes):

        if n % 1000 == 0:
            print(f"Gene {n}")

        _, resid, _, _ = fit_single(x, y[:, n:n + 1])
        ex_var_single[n] = 1 - np.var(resid) / np.var(y[:, n])
        s, resid, _, cutoff_time[n] = fit_piecewise_search(x, y[:, n:n + 1])
        ex_var_double[n] = 1 - np.var(resid) / np.var(y[:, n])
        slopes[n, :] = np.squeeze(s)

    return ex_var_single, ex_var_double, slopes, cutoff_time


def fit_piecewise_search(x, y, time_resolution=10):
    n_time, n_genes = y.shape
    ex_var = []
    for t in range(time_resolution, n_time, time_resolution):
        early_time = np.arange(t)
        late_time = np.arange(t, n_time)
        _, resid, _ = fit_piecewise(x, y, early_time, late_time)
        ex_var.append(1 - np.var(resid) / np.var(y))

    t = (1 + np.argmax(np.stack(ex_var))) * time_resolution
    early_time = np.arange(t)
    late_time = np.arange(t, n_time)
    slopes, resid, y_hat = fit_piecewise(x, y, early_time, late_time)

    return slopes, resid, y_hat, t


def fit_single(x, y):
    n_time, n_genes = y.shape
    time = np.arange(0, n_time)
    slopes = np.zeros((n_genes), dtype=np.float32)
    resid = np.zeros((n_time, n_genes), dtype=np.float32)
    for n in range(n_genes):
        curve_coefs, _ = optimize.curve_fit(curve, x, y[:, n])
        slopes[n] = curve_coefs[1]
        y_hat = curve(x, *curve_coefs)
        resid[:, n] = y[:, n] - y_hat

    return slopes, resid, curve_coefs, y_hat


def fit_piecewise(x, y, idx_early, idx_late, n_bootstrap=None):
    n_time, n_genes = y.shape
    time0 = x[idx_early]
    time1 = x[idx_late]
    slopes = np.zeros((3, n_genes), dtype=np.float32)
    resid = np.zeros((n_time, n_genes), dtype=np.float32)
    y_hat = np.zeros((n_time, n_genes), dtype=np.float32)
    for n in range(n_genes):
        curve_coefs, _ = optimize.curve_fit(curve, time0, y[idx_early, n])
        slopes[0, n] = curve_coefs[1]
        y1 = curve(time0, *curve_coefs)
        resid[idx_early, n] = y[idx_early, n] - y1
        y_hat[idx_early, n] = y1

        curve_coefs, _ = optimize.curve_fit(curve, time1, y[idx_late, n])
        y1 = curve(time1, *curve_coefs)
        slopes[1, n] = curve_coefs[1]
        resid[idx_late, n] = y[idx_late, n] - y1
        y_hat[idx_late, n] = y1

        curve_coefs, _ = optimize.curve_fit(curve, x, y[:, n])
        slopes[2, n] = curve_coefs[1]

    if n_bootstrap is not None:
        slopes_boot = np.zeros((2, n_bootstrap, n_genes), dtype=np.float32)
        for m in range(n_bootstrap):
            print(f"Bootstrap number: {m}")
            for n in range(n_genes):
                y_shuffle = copy.deepcopy(y[idx_early, n])
                y_shuffle = np.random.permutation(y_shuffle)
                curve_coefs, _ = optimize.curve_fit(curve, time0, y_shuffle)
                slopes_boot[0, m, n] = curve_coefs[1]

                y_shuffle = copy.deepcopy(y[idx_late, n])
                y_shuffle = np.random.permutation(y_shuffle)
                curve_coefs, _ = optimize.curve_fit(curve, time1, y_shuffle)
                slopes_boot[1, m, n] = curve_coefs[1]

        u = np.mean(slopes_boot, axis=1)
        sd = np.std(slopes_boot, axis=1) + 1e-3
        slopes = (slopes - u) / sd

    return slopes, resid, y_hat

def create_ensembl_list(
        genes,
        fn,
        magma_dir="/home/masse/work/mssm/psuedotime/magma_input",
        csv_dir=None,
):

    gene_ensembl = []
    gene_names = []
    for g in genes:
        gene_names.append(g)
        try:
            gene_ensembl.append(gene_dict["gene_convert"][g])
        except:
            continue

    df = pd.DataFrame(gene_ensembl, columns=["colummn"])
    df.to_csv(f'{magma_dir}/{fn}.csv', index=False)
    if csv_dir is not None:
        df = pd.DataFrame(gene_names, columns=["colummn"])
        df.to_csv(f'{csv_dir}/{fn}.csv', index=False)

def bad_path_idx(pathways):
    idx = []
    for p in pathways:
        include = False
        for b in bad_path_words:
            if b in p:
                include = True
                break
        idx.append(include)
    return idx


def traj_wrapper(
        gene_scores,
        pred_braak,
        pred_dementia,
        gamma_braak,
        gamma_dementia,
        neighbors=None,
        gene_type="gene_means",
        normalize_counts=False,
        log_counts=False,
        calculate_residual=True,
        log_prior=1.0,
        n_edge_exclude=10,
):
    if normalize_counts:
        mean_total = 10_000
        gene_scores /= (np.sum(gene_scores, axis=1, keepdims=True) / mean_total)
    if log_counts:
        gene_scores = np.log(gene_scores + log_prior)

    p_braak, genes_braak, _ = calculate_trajectories_single(
        pred_braak, gene_scores, gamma_braak, neighbors=neighbors
    )
    p_dementia, genes_dementia, _ = calculate_trajectories_single(
        pred_dementia, gene_scores, gamma_dementia, neighbors=neighbors
    )

    if calculate_residual:
        resid_genes, smooth_resid_genes = calculate_trajectory_residuals(
            pred_braak, pred_dementia, gene_scores, gamma_braak
        )
    else:
        resid_genes = None
        smooth_resid_genes = None

    # sort braak and exclude edges
    idx_braak_sort = np.argsort(pred_braak)
    if n_edge_exclude > 0:
        idx_braak_sort = idx_braak_sort[n_edge_exclude:-n_edge_exclude]
    p_braak = p_braak[idx_braak_sort]
    genes_braak = genes_braak[idx_braak_sort]
    resid_genes = resid_genes[idx_braak_sort]
    smooth_resid_genes = smooth_resid_genes[idx_braak_sort]

    # sort dementia and exclude edges
    idx_braak_sort = np.argsort(p_dementia)
    if n_edge_exclude > 0:
        idx_braak_sort = idx_braak_sort[n_edge_exclude:-n_edge_exclude]
    p_dementia = p_dementia[idx_braak_sort]
    genes_dementia = genes_dementia[idx_braak_sort]

    return p_braak, p_dementia, genes_braak, genes_dementia, resid_genes, smooth_resid_genes


def calculate_trajectory_residuals(pred_x, pred_y, gene_vals, gamma):
    resids = []
    smooth_resids = []

    # in case these are actual values
    pred_x[pred_x < -9] = np.nan
    pred_y[pred_y < -9] = np.nan

    time_pts = np.linspace(np.min(pred_x), np.max(pred_x), 100)

    for i in range(len(pred_x)):
        delta = (pred_x - pred_x[i]) ** 2
        w = np.exp(-gamma * delta)
        w[i] = 0
        w /= np.sum(w)
        local_x = np.sum(w * pred_x, axis=0)
        local_y = np.sum(w * pred_y, axis=0)
        local_gene_vals = np.sum(w[:, None] * gene_vals, axis=0)
        y_resid = pred_y[i] - local_y
        gene_resid = gene_vals[i, :] - local_gene_vals
        resids.append(y_resid * gene_resid)

    # smmoth the residuals
    resids = np.stack(resids)
    for i in range(len(pred_x)):
        delta = (pred_x - pred_x[i]) ** 2
        w = np.exp(-gamma * delta)
        w /= np.sum(w)
        y = np.sum(w[:, None] * resids, axis=0)
        smooth_resids.append(y)

    smooth_resids = np.stack(smooth_resids)

    return resids, smooth_resids


def calculate_trajectory_residuals_old(pred_x, pred_y, gene_vals, gamma):
    resids = []
    resids_y = []

    # in case these are actual values
    pred_x[pred_x < -9] = np.nan
    pred_y[pred_y < -9] = np.nan

    time_pts = np.linspace(np.min(pred_x), np.max(pred_x), 100)

    for i in range(len(pred_x)):
        smoothed_x = []
        smoothed_y = []
        weights = []
        for t in time_pts:
            delta = (pred_x - t) ** 2
            w = np.exp(-gamma * delta)
            w[np.isnan(w)] = 0.0
            w[i] = 0.0
            w /= np.sum(w)
            weights.append(w)
            local_x = np.sum(w * pred_x, axis=0)
            local_y = np.sum(w * pred_y, axis=0)
            smoothed_x.append(local_x)
            smoothed_y.append(local_y)

        smoothed_x = np.stack(smoothed_x)
        smoothed_y = np.stack(smoothed_y)
        t_min = np.argmin((pred_x[i] - smoothed_x) ** 2)

        local_gene_vals = np.sum(weights[t_min][:, None] * gene_vals, axis=0)
        y_resid = pred_y[i] - smoothed_y[t_min]
        gene_resid = gene_vals[i, :] - local_gene_vals

        resids.append(y_resid * gene_resid)
        resids_y.append(y_resid)

    return np.stack(resids), np.stack(resids_y)


def calculate_trajectories_single(x, y, gamma, neighbors=None):
    x_smooth = []  # BRAAK or DM predictions
    y_smooth = []  # gene scores
    resid = []

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

        resid.append((x[i] - local_x) * (y[i, :] - local_y))

    return np.stack(x_smooth), np.stack(y_smooth), np.stack(resid)


def link_results(model_fn, df_meta, include_pxr=False, include_pathway_means=False):
    results = {}
    donors = np.array(df_meta.SubID.values)

    n_genes_pxr = len(gene_names_pxr)
    n_genes = len(gene_names)

    cell_level_attrs = ["pred_BRAAK_AD", "pred_Dementia", "cell_count"]
    if include_pathway_means:
        path_level_attrs = ["pathway_donor_means"]
    gene_level_attrs = ["gene_means", "gene_BRAAK_AD_zscore", "gene_Dementia_zscore"]
    donor_level_attrs = [
        "BRAAK_AD", "Dementia",
    ]

    for k in donor_level_attrs:
        results[k] = np.nan * np.ones((len(donors)), dtype=np.float32)
    for k in cell_level_attrs:
        results[k] = np.nan * np.ones((len(donors),), dtype=np.float32)
    if include_pathway_means:
        for k in path_level_attrs:
            results[k] = np.nan * np.ones((len(donors), n_paths), dtype=np.float32)

    k = "gene_means"
    results[k] = np.nan * np.ones((len(donors), n_genes), dtype=np.float32)
    if include_pxr:
        k = "px_r"
        results[k] = np.nan * np.ones((len(donors), n_genes_pxr), dtype=np.float32)

    adata = sc.read_h5ad(model_fn)

    for m, subid in enumerate(adata.uns["donors"]):

        j = np.where(subid == donors)[0][0]

        for k in donor_level_attrs:
            results[k][j] = adata.uns[f"donor_{k}"][m]

        for k in donor_level_attrs:
            results[k][j] = adata.uns[f"donor_{k}"][m]

        for k in cell_level_attrs:
            results[k][j] = adata.uns[f"donor_{k}"][m]

        if include_pathway_means:
            for k in path_level_attrs:
                results[k][j, :] = adata.uns[f"{k}"][m, :]

        k = "gene_means"
        s = adata.uns[f"donor_{k}"][m, :]
        results[k][j, :] = s

        if include_pxr:
            k = "px_r"
            s = adata.uns[f"donor_{k}"][subid]
            results[k][j, :] = s

    return results


def single_double_expalined_var(
        x,
        y,
        protein_coding_only=True,
        time_resolution=5,
        top_k=None,
        min_time=50,
        max_time=510,
        mean_gene_threshold=None,
        n_boot=None,
):
    if protein_coding_only:
        y = y[:, protein_coding]

    if mean_gene_threshold is not None:
        idx = np.mean(y, axis=0) > mean_gene_threshold
        print(f"Genes above threshold: {np.sum(idx)}")
        y = y[:, idx]
    elif top_k is not None:
        idx = np.argsort(np.var(y, axis=0))[-top_k:]
        y = y[:, idx]

    cutoff_times = np.arange(min_time, max_time, time_resolution)
    double_ex_var = []

    slopes, resid, _, _ = fit_single(x, y)
    var_resid = np.sum(np.var(resid, axis=0)) / np.sum(np.var(y, axis=0))
    single_ex_var = 1 - var_resid
    print("Single exaplined variance", single_ex_var)

    idx = np.argsort(x)
    for t in cutoff_times:
        idx_early = idx[:t]
        idx_late = idx[t:]
        slopes, resid, y_hat = fit_piecewise(x, y, idx_early, idx_late)
        var_resid = np.sum(np.var(resid, axis=0)) / np.sum(np.var(y, axis=0))
        double_ex_var.append(1 - var_resid)
        print(t, x[t], double_ex_var[-1])

    double_ex_var = np.stack(double_ex_var)
    boot_nonlinear_index = []
    if n_boot is not None:
        t_max = cutoff_times[np.argmax(double_ex_var)]
        idx_early = idx[:t_max]
        idx_late = idx[t_max:]
        print("t_max", t_max, len(idx_early), len(idx_late))
        for _ in range(n_boot):
            """
            idx_boot = np.random.choice(y.shape[0], size=y.shape[0], replace=True)
            x0 = x[idx_boot]
            y0 = y[idx_boot, :]
            idx = np.argsort(x0)
            x0 = x0[idx]
            y0 = y0[idx]
            idx_early = idx[:t_max]
            idx_late = idx[t_max:]
            """
            idx_boot = np.random.choice(y.shape[1], size=y.shape[1], replace=True)
            x0 = x
            y0 = y[:, idx_boot]

            _, resid, _, _ = fit_single(x0, y0)
            var_resid_linear = np.sum(np.var(resid, axis=0)) / np.sum(np.var(y0, axis=0))
            _, resid, _ = fit_piecewise(x0, y0, idx_early, idx_late)
            var_resid_nonlinear = np.sum(np.var(resid, axis=0)) / np.sum(np.var(y0, axis=0))
            boot_nonlinear_index.append(var_resid_linear - var_resid_nonlinear)
            print("boot", var_resid_linear, var_resid_nonlinear)

    return single_ex_var, double_ex_var, cutoff_times, boot_nonlinear_index


def single_double_expalined_var_boot(
        x,
        y,
        protein_coding_only=True,
        time_resolution=10,
        top_k=None,
        mean_gene_threshold=None,
        min_time=100,
        max_time=510,
        n_bootstrap=20,
):
    print("AAA", y.shape)
    if protein_coding_only:
        y = y[:, protein_coding]

    print("BBB", y.shape)

    boot_nonlinearity = []
    cutoff_times = np.arange(min_time, max_time, time_resolution)

    for _ in range(n_bootstrap):

        idx = np.random.choice(y.shape[0], size=y.shape[0], replace=True)
        x0 = x[idx]
        y0 = y[idx, :]

        if mean_gene_threshold is not None:
            idx_genes = np.mean(y0, axis=0) > mean_gene_threshold
            print(f"Genes above threshold: {np.sum(idx_genes)}")
            y0 = y0[:, idx_genes]
            print("CCC", y0.shape)
        elif top_k is not None:
            idx_genes = np.argsort(np.var(y0, axis=0))[-top_k:]
            y0 = y0[:, idx_genes]

        double_ex_var = []
        slopes, resid, _, _ = fit_single(x0, y0)
        var_resid = np.mean(np.var(resid, axis=0) / np.var(y0, axis=0))
        single_ex_var = 1 - var_resid

        for t in cutoff_times:
            idx = np.argsort(x0)
            x0 = x0[idx]
            y0 = y0[idx]
            idx_early = idx[:t]
            idx_late = idx[t:]
            slopes, resid, y_hat = fit_piecewise(x0, y0, idx_early, idx_late)
            var_resid = np.mean(np.var(resid, axis=0) / np.var(y0, axis=0))
            double_ex_var.append(1 - var_resid)

        double_ex_var = np.stack(double_ex_var)
        d = np.max(double_ex_var) - single_ex_var
        print(single_ex_var, np.max(double_ex_var))
        boot_nonlinearity.append(d)

    print("nonlinear std", np.std(np.stack(boot_nonlinearity)))

    return np.stack(boot_nonlinearity)


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


