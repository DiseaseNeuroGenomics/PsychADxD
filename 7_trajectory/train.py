from typing import List
import torch
import pickle
import pytorch_lightning as pl
from data import DataModule
from config import dataset_cfg, model_cfg, task_cfg, trainer_cfg
from task import LitVAE
from networks import VAE, load_model

torch.set_float32_matmul_precision('medium')

def main(train_idx: List[int], test_idx: List[int]):

    # Set seed
    pl.seed_everything(42)

    # Add train and test indices
    dataset_cfg["train_idx"] = train_idx
    dataset_cfg["test_idx"] = test_idx

    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup("train")

    # Add parameters to model properties
    task_cfg["cell_properties"] = model_cfg["cell_properties"] = dm.cell_properties
    model_cfg["n_input"] = dm.train_dataset.n_genes
    model_cfg["batch_properties"] = dm.batch_properties
    model_cfg["gene_chrom_dict"] = dm.train_dataset.gene_chrom_dict

    # initialize model and task
    network = VAE(**model_cfg)
    task = LitVAE(network=network, **task_cfg)

    # initialize trainer
    trainer = pl.Trainer(
        enable_checkpointing=False,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=trainer_cfg["max_epochs"],
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
    )

    # train
    trainer.fit(task, dm)


if __name__ == "__main__":

    splits = pickle.load(open(trainer_cfg["splits_path"], "rb"))

    for split_num in range(0, 10):

        print(f"Split number: {split_num}")
        train_idx = splits[split_num]["train_idx"]
        test_idx = splits[split_num]["test_idx"]
        main(train_idx, test_idx)