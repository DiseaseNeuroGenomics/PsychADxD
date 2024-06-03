from typing import Any, List, Optional

import os
import pickle
import numpy as np
import scanpy as sc
from anndata.experimental.multi_files import AnnCollection


class CreateData:
    """Class to save the data in numpy memmap format
    Will save the gene expression using int16 to save space,
    but will create the function to normalize the data using
    the gene expression statistics"""

    def __init__(
        self,
        source_paths: List[str],
        target_path: str,
        min_genes_per_cell: int = 1000,
        min_percent_cells_per_gene: float = 0.0,
        protein_coding_only: bool = False,
    ):
        self.source_paths = source_paths
        self.target_path = target_path

        self.min_genes_per_cell = min_genes_per_cell
        self.min_percent_cells_per_gene = min_percent_cells_per_gene
        self.protein_coding_only = protein_coding_only

        self.obs_keys = [
            'CERAD', 'BRAAK_AD', 'BRAAK_PD', 'Dementia', 'AD', 'class', 'subclass', 'subtype', 'ApoE_gt', "OCD",
            'Sex', 'Head_Injury', 'Vascular', 'Age', 'Epilepsy', 'Seizures', 'Tumor', 'PD', 'ALS', "FTD", "ADHD",
            'CDRScore', 'PMI', 'Cognitive_Resilience', 'Cognitive_and_Tau_Resilience', 'SubID',
            'snRNAseq_ID', 'SCZ', 'MDD', 'Brain_bank', "n_counts", "n_genes",
        ]
        self.var_keys = [
            'gene_id', 'gene_name', 'gene_type', 'gene_chrom', 'robust', 'highly_variable_features', 'ribosomal',
            'mitochondrial', 'protein_coding', 'mitocarta_genes', 'robust_protein_coding', 'percent_cells',
        ]

        if len(source_paths) == 1:
            self.anndata = sc.read_h5ad(source_paths[0], 'r')
        else:
            temp = [sc.read_h5ad(fn, 'r') for fn in source_paths]
            self._quality_check(temp)
            self.anndata = AnnCollection(temp, join_vars='inner', join_obs='inner')
            self.anndata.var = temp[0].var

        self._get_cell_index()
        self._get_gene_index()

        print(f"Size of anndata {self.anndata.shape[0]}")

    def _quality_check(self, data: List[Any]):
        """Ensure that the first two Anndata objects have matching gene names and percent cells"""
        vars = ["gene_name", "percent_cells"]
        for v in vars:
            match = [g0 == g1 for g0, g1 in zip(data[0].var[v], data[1].var[v])]
            assert np.mean(np.array(match)) == 1, f"{v} DID NOT MATCH match between the first two datasets"

            print(f"{v} matched between the first two datasets")


    def _create_metadata(self, train: bool = True):


        meta = {
            "obs": {k: self.anndata.obs[k][self.cell_idx].values for k in self.obs_keys},
            "var": {k: self.anndata.var[k][self.gene_idx].values for k in self.var_keys},
        }
        meta = self._add_graded_dementia(meta)
        meta["obs"]["barcode"] = list(np.array(self.anndata.obs.index)[self.cell_idx])

        meatadata_fn = os.path.join(self.target_path, "metadata.pkl")
        pickle.dump(meta, open(meatadata_fn, "wb"))

    @staticmethod
    def _add_graded_dementia(meta):

        meta["obs"]["Dementia_graded"] = np.array(meta["obs"]["Dementia"]).astype(np.float32)
        idx = np.where((meta["obs"]["Dementia"] == 0) * (meta["obs"]["MCI"] == 0))[0]
        meta["obs"]["Dementia_graded"][idx] = 0.0
        idx = np.where((meta["obs"]["Dementia"] == 0) * (meta["obs"]["MCI"] == 1))[0]
        meta["obs"]["Dementia_graded"][idx] = 0.5
        idx = np.where((meta["obs"]["Dementia"] == 1) * (meta["obs"]["MCI"] == 0))[0]
        meta["obs"]["Dementia_graded"][idx] = 1.0

        return meta

    def _get_gene_index(self):

        if "percent_cells" in self.anndata.var.keys():
            self.gene_idx = self.anndata.var["percent_cells"] > 100 * self.min_percent_cells_per_gene
            if self.protein_coding_only:
                self.gene_idx *= self.anndata.var["protein_coding"]
            self.gene_idx = np.where(self.gene_idx)[0]

        else:
            chunk_size = 10_000
            n_segments = 10
            n = self.anndata.shape[0]
            start_idx = np.linspace(0, n - chunk_size - 1, n_segments)
            gene_expression = []

            for i in start_idx:
                x = self.anndata[int(i): int(i + chunk_size)]
                x = x.X.toarray()
                gene_expression.append(np.mean(x > 0, axis=0))

            gene_expression = np.mean(np.stack(gene_expression), axis=0)
            self.gene_idx = np.where(gene_expression >= self.min_percent_cells_per_gene)[0]

        print(f"Number of genes selected: {len(self.gene_idx)}")

    def _get_cell_index(self):

        n_genes = self.anndata.obs["n_genes"].values
        self.cell_idx = np.where(n_genes > self.min_genes_per_cell)[0]

    def _create_dataset(self):

        data_fn = os.path.join(self.target_path, "data.dat")
        n_genes = len(self.gene_idx)
        n_cells = len(self.cell_idx)
        print(f"Creating data. Number of cell: {n_cells}, number of genes: {n_genes}")

        chunk_size = 10_000  # chunk size for loading data into memory
        fp = np.memmap(data_fn, dtype='uint8', mode='w+', shape=(n_cells, n_genes))

        for n in range(int(n_cells / chunk_size)):
            m = np.minimum(n_cells, (n + 1) * chunk_size)
            current_idx = self.cell_idx[n * chunk_size: m]
            print(f"Creating dataset, cell number = {current_idx[0]}")
            y = self.anndata[current_idx]
            y = y.X.toarray()
            y = y[:, self.gene_idx]
            y[y >= 255] = 255
            y = y.astype(np.uint8)
            fp[n * chunk_size: m, :] = y

            print(f"Chunk number {n} out of {int(np.ceil(len(self.cell_idx) / chunk_size))} created")

        # flush to memory
        fp.flush()

        return fp

    def _create_datasets(self) -> None:

        print("Saving the metadata...")
        self._create_metadata()

        print("Saving the expression data in the memmap array...")
        fp = self._create_dataset()

def create_datasets(source_paths: List[str], target_path: str) -> None:

	c = CreateData(source_paths, target_path)
    	c.create_datasets()




