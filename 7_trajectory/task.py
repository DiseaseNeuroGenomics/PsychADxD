import os
from typing import Any, Dict, List, Optional

import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions import Normal, kl_divergence as kl
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchmetrics import MetricCollection, ExplainedVariance
from torchmetrics.classification import Accuracy

from distributions import (
    ZeroInflatedNegativeBinomial,
    NegativeBinomial,
    Poisson,
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CellPrediction(pl.LightningModule):
    def __init__(
        self,
        network,
        cell_properties: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.0005,
        warmup_steps: float = 2000.0,
        weight_decay: float = 0.0,
        balance_classes: bool = False,
        focal_loss: bool = False,
        momentum: float = 0.0,
        label_smoothing: float = 0.0,
        gene_names: Optional[List[str]] = None,
        perturb_gene_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        self.network = network
        self.cell_properties = cell_properties
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.balance_classes = balance_classes
        self.focal_loss = focal_loss
        self.momentum = momentum
        self.label_smoothing = label_smoothing
        self.gene_names = gene_names

        # not used for Capstone paper, can be used to infer the affect of gene perturbations
        self._get_perturb_gene_index(perturb_gene_names)

        self._cell_properties_metrics()
        self._reset_results_dict()
        self.train_step = 0
        self.source_code_copied = False

    def copy_source_code(self, version_num):
        """To document each training run"""
        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        src_files = [
            "/home/masse/work/vae/src/config.py",
            "/home/masse/work/vae/src/data.py",
            "/home/masse/work/vae/src/modules.py",
            "/home/masse/work/vae/src/networks.py",
            "/home/masse/work/vae/src/task.py",
            "/home/masse/work/vae/src/losses.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")
        self.source_code_copied = True

    def _get_perturb_gene_index(self, perturb_gene_names):

        """Used to infer how gene perturbations affect model predictions"""
        self.perturb_gene_names = []
        self.perturb_gene_idx = []
        for g in perturb_gene_names:
            if g in self.gene_names:
                j = np.where(np.array(self.gene_names) == g)[0][0]
                self.perturb_gene_names.append(g)
                self.perturb_gene_idx.append(j)


    def _reset_results_dict(self):

        """Model predictions will be saved in this dict"""
        self.results = {}
        self.results["epoch"] = self.current_epoch
        if self.gene_names is not None:
            self.results["gene_names"] = self.gene_names
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results[f"pred_{k}"] = []
        if len(self.perturb_gene_names) > 0:
            self.results["perturb_gene_names"] = self.perturb_gene_names
            for k in self.cell_properties.keys():
                self.results[f"pred_perturb_up_{k}"] = []
                self.results[f"pred_perturb_down_{k}"] = []
        self.results["cell_idx"] = []
        self.results["cell_mask"] = []


    def _cell_properties_metrics(self):

        """Used to calculate validation metrics, and to save model predictions"""
        self.cell_cross_ent = nn.ModuleDict()
        self.cell_mse = nn.ModuleDict()
        self.cell_accuracy = nn.ModuleDict()
        self.cell_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module
                weight = torch.from_numpy(
                    np.float32(np.clip(1 / cell_prop["freq"], 0.1, 10.0))
                ) if self.balance_classes else None
                if self.focal_loss:
                    self.cell_cross_ent[k] = FocalLoss(
                        len(cell_prop["values"]),
                        gamma=self.focal_loss_gamma,
                        alpha=2.0,
                    )
                else:
                    self.cell_cross_ent[k] = nn.CrossEntropyLoss(
                        weight=weight,
                        reduction="none",
                        ignore_index=-100,
                        label_smoothing=self.label_smoothing,
                    )

                self.cell_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_mse[k] = nn.MSELoss(reduction="none")
                self.cell_explained_var[k] = ExplainedVariance()


    def validation_step(self, batch, batch_idx):

        gene_vals, cell_targets, cell_mask, batch_labels, batch_mask, cell_idx, _, _ = batch
        cell_pred, _ = self.network(gene_vals, batch_labels, batch_mask)

        self._cell_scores(cell_pred, cell_targets, cell_mask)
        self._save_predictions(cell_pred, cell_targets, cell_mask, cell_idx)
        loss = self._cell_loss(cell_pred, cell_targets, cell_mask)

        if len(self.perturb_gene_names) > 0:
            self._generate_perturbations(batch)

        return loss

    def _generate_perturbations(self, batch):

        gene_vals, cell_targets, cell_mask, batch_labels, batch_mask, cell_idx, _, _ = batch

        gene_vals_perturb_up = gene_vals.detach().clone()
        gene_vals_perturb_down = gene_vals.detach().clone()
        for i in self.perturb_gene_idx:
            gene_vals_perturb_up[:, i] = gene_vals_perturb_up[:, i] + 1
            gene_vals_perturb_down[:, i] = 0.0

        cell_perturb_up, _ = self.network(gene_vals_perturb_up, batch_labels, batch_mask)
        cell_perturb_down, _ = self.network(gene_vals_perturb_down, batch_labels, batch_mask)

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                pred_prob_up = F.softmax(cell_perturb_up[k], dim=-1).to(torch.float32).detach().cpu().numpy()
                pred_prob_down = F.softmax(cell_perturb_down[k], dim=-1).to(torch.float32).detach().cpu().numpy()
                if k != "SubID":
                    self.results["pred_perturb_up_" + k].append(pred_prob_up)
                    self.results["pred_perturb_down_" + k].append(pred_prob_down)
            else:
                self.results[f"pred_perturb_up_{k}"].append(
                    cell_perturb_up[k][:, 0].detach().cpu().to(torch.float32).numpy()
                )
                self.results[f"pred_perturb_down_{k}"].append(
                    cell_perturb_up[k][:, 0].detach().cpu().to(torch.float32).numpy()
                )


    def on_validation_epoch_end(self):

        for k in self.cell_properties.keys():
            if len(self.results[k]) > 0:
                self.results[k] = np.concatenate(self.results[k], axis=0)
                if k != "SubID":
                    self.results[f"pred_{k}"] = np.concatenate(self.results[f"pred_{k}"], axis=0)

        self.results["cell_mask"] = np.concatenate(self.results["cell_mask"], axis=0)
        self.results["cell_idx"] = np.concatenate(self.results["cell_idx"], axis=0)

        v = self.trainer.logger.version
        if not self.source_code_copied:
            self.copy_source_code(v)

        fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"
        if self.current_epoch > -1:
            pickle.dump(self.results, open(fn, "wb"))

        self._reset_results_dict()

    def _save_predictions(self, cell_pred, cell_targets, cell_mask, cell_idx):

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                targets = cell_targets[:, n].to(torch.int64)
                pred_prob = F.softmax(cell_pred[k], dim=-1).to(torch.float32).detach().cpu().numpy()
                if k != "SubID":
                    self.results[k].append(targets.detach().cpu().numpy())
                    self.results["pred_" + k].append(pred_prob)
            else:
                pred = cell_pred[k][:, 0]
                self.results[k].append(cell_targets[:, n].detach().cpu().numpy())
                self.results[f"pred_{k}"].append(pred.detach().cpu().to(torch.float32).numpy())

        self.results["cell_mask"].append(cell_mask.detach().cpu().to(torch.float32).numpy())
        self.results["cell_idx"].append(cell_idx.detach().cpu().numpy())

    def _cell_scores(self, cell_pred, cell_targets, cell_mask):

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            idx = torch.nonzero(cell_mask[:, n])

            if cell_prop["discrete"]:
                pred_idx = torch.argmax(cell_pred[k], dim=-1).to(torch.int64)
                pred_prob = F.softmax(cell_pred[k], dim=-1).to(torch.float32).detach().cpu().numpy()

                targets = cell_targets[:, n].to(torch.int64)
                self.cell_accuracy[k].update(pred_idx[idx][None, :], targets[idx][None, :])

            else:
                pred = cell_pred[k][:, 0]
                try:  # rare error
                    self.cell_explained_var[k].update(pred[idx], cell_targets[idx, n])
                except:
                    self.cell_explained_var[k].update(pred, cell_targets[idx, n])

        for k, v in self.cell_accuracy.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_explained_var.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):

        self.train_step += 1

        gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, _, group_idx, _ = batch
        cell_pred, _ = self.network(gene_vals, batch_labels, batch_mask)
        loss = self._cell_loss(cell_pred, cell_prop_vals, cell_mask)

        return loss

    def _cell_loss(
        self,
        cell_pred: Dict[str, torch.Tensor],
        cell_prop_vals: torch.Tensor,
        cell_mask: torch.Tensor,
    ):

        cell_loss = 0.0

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                loss = self.cell_cross_ent[k](cell_pred[k], cell_prop_vals[:, n].to(torch.int64))
                masked_loss = (loss * cell_mask[:, n]).mean()
                cell_loss += masked_loss


            else:
                alpha = 1 / self.cell_properties[k]["std"] ** 2
                loss = alpha * self.cell_mse[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n])
                masked_loss = (loss * cell_mask[:, n]).mean()
                cell_loss += masked_loss


            if self.training:
                self.log(f"loss_{k}", masked_loss, on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True)

        loss_name = "cell_loss_train" if self.training else "cell_loss_val"
        self.log(loss_name, cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return cell_loss

    def configure_optimizers(self):

        opt = torch.optim.SGD(
            self.network.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = {
            'scheduler': WarmupSchedule(opt, warmup_steps=self.warmup_steps),
            'interval': 'step',
        }

        return {
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,  # Changed scheduler to lr_scheduler
            'interval': 'step',
        }


# define the LightningModule
class LitVAE(pl.LightningModule):
    def __init__(
        self,
        network,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        balance_classes: bool = False,
        dispersion: str = "gene",
        reconstruction_loss: str = "zinb",
        latent_distribution: str = "normal",
        n_epochs_kl_warmup: Optional[int] = None,
        learning_rate: float = 0.0005,
        warmup_steps: float = 2000.0,
        weight_decay: float = 0.1,
        l1_lambda: float = 0.0,
        gene_loss_coeff: float = 5e-4,
    ):
        super().__init__()

        self.network = network
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.balance_classes = balance_classes
        self.dispersion = dispersion
        self.reconstruction_loss = reconstruction_loss
        self.latent_distribution = latent_distribution
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.gene_loss_coeff = gene_loss_coeff

        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")

        self._cell_properties_metrics()
        self._create_results_dict()
        self.train_step = 0
        self.source_code_copied = False


    def _create_results_dict(self):

        self.results = {"epoch": 0}
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []
        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = []
        self.results["cell_idx"] = []

    def _cell_properties_metrics(self):

        self.cell_cross_ent = nn.ModuleDict()
        self.cell_mse = nn.ModuleDict()
        self.cell_accuracy = nn.ModuleDict()
        self.cell_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module
                weight = torch.from_numpy(
                    np.float32(np.clip(1 / cell_prop["freq"], 0.1, 10.0))
                ) if self.balance_classes else None
                self.cell_cross_ent[k] = nn.CrossEntropyLoss(weight=weight, reduction="none", ignore_index=-100)
                # self.cell_cross_ent[k] = FocalLoss(len(cell_prop["values"]), gamma=2.0, alpha=5.0)
                # self.cell_cross_ent[k] = Poly1FocalLoss(len(cell_prop["values"]), gamma=2.0)
                self.cell_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_mse[k] = nn.MSELoss(reduction="none")
                self.cell_explained_var[k] = ExplainedVariance()

    def gene_loss(
        self,
        x: torch.Tensor,
        qz_m: torch.Tensor,
        qz_v: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        training: bool = True,
    ):

        # KL Divergence

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(mean, scale)
        ).sum(dim=1)

        recon_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        loss = recon_loss.mean() + self.kl_weight * kl_z.mean()

        if training:
            self.log("recon_loss", recon_loss.mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("kl_z", kl_z.mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("recon_loss_val", recon_loss.mean(), on_step = False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("kl_z_val", kl_z.mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss, kl_z

    @property
    def kl_weight(self):
        if self.n_epochs_kl_warmup is not None:
            kl_weight = np.clip((1 + self.current_epoch) / self.n_epochs_kl_warmup, 0.0, 1.0)
        else:
            kl_weight = 1.0

        return kl_weight

    def validation_step(self, batch, batch_idx):

        gene_vals, cell_targets, cell_mask, batch_labels, batch_mask, cell_idx, _, subject = batch
        qz_m, qz_v, z = self.network.z_encoder(gene_vals, batch_labels, batch_mask)

        library_size_per_cell = torch.log(gene_vals.sum(dim=1, keepdim=True))
        px_scale, _, px_rate, px_dropout = self.network.decoder(
            self.dispersion,
            z,
            library_size_per_cell,
            batch_labels,
            batch_mask,
        )

        px_r = self.network.px_r
        px_r = torch.exp(px_r)

        gene_loss, kl_z = self.gene_loss(
            gene_vals,
            qz_m,
            qz_v,
            px_rate,
            px_r,
            px_dropout,
            training=False,
        )

        self.results["cell_idx"].append(cell_idx.detach().cpu().numpy())
        if self.cell_properties is not None:
            cell_pred = self.network.cell_decoder(qz_m, batch_labels, batch_mask)
            self.cell_scores(cell_pred, cell_targets, cell_mask, qz_m, subject)
            cell_loss = self._cell_loss(cell_pred, cell_targets, cell_mask, None)

        if self.batch_properties is not None:
            for n, k in enumerate(self.batch_properties.keys()):
                self.results[k].append(batch_labels[:, n].detach().cpu().numpy())

    def copy_source_code(self, version_num):

        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        src_files = [
            "config.py",
            "data.py",
            "modules.py",
            "networks.py",
            "task.py",
            "train.py",
            "distributions.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")
        self.source_code_copied = True

    def on_validation_epoch_end(self):

        for k in self.cell_properties.keys():
            if len(self.results[k]) > 0:
                self.results[k] = np.concatenate(self.results[k], axis=0)
                if k != "SubID":
                    self.results["pred_" + k] = np.concatenate(self.results["pred_" + k], axis=0)

        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = np.concatenate(self.results[k], axis=0)
        self.results["cell_idx"] = np.concatenate(self.results["cell_idx"], axis=0)


        v = self.trainer.logger.version
        if not self.source_code_copied:
            self.copy_source_code(v)

        fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"
        pickle.dump(self.results, open(fn, "wb"))

        self.results["epoch"] = self.current_epoch + 1
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []
        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = []

        self.results["cell_idx"] = []

    def cell_scores(self, cell_pred, cell_targets, cell_mask, latent_mean, subject):

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            idx = torch.nonzero(cell_mask[:, n])

            if cell_prop["discrete"]:
                pred_idx = torch.argmax(cell_pred[k], dim=-1).to(torch.int64)
                pred_prob = F.softmax(cell_pred[k], dim=-1).to(torch.float32).detach().cpu().numpy()
                targets = cell_targets[:, n].to(torch.int64)

                self.cell_accuracy[k].update(pred_idx[idx][None, :], targets[idx][None, :])
                if k != "SubID":
                    self.results[k].append(targets.detach().cpu().numpy())
                    self.results["pred_" + k].append(pred_prob)
            else:

                pred = cell_pred[k][:, 0]
                try: # rare error
                    self.cell_explained_var[k].update(pred[idx], cell_targets[idx, n])
                except:
                    self.cell_explained_var[k].update(pred, cell_targets[idx, n])

                self.results[k].append(cell_targets[:, n].detach().cpu().numpy())
                self.results["pred_" + k].append(pred.detach().cpu().to(torch.float32).numpy())

        for k, v in self.cell_accuracy.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_explained_var.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def _cell_loss(
        self,
        cell_pred: Dict[str, torch.Tensor],
        cell_prop_vals: torch.Tensor,
        cell_mask: torch.Tensor,
        group_idx: torch.Tensor,
    ):

        cell_loss = 0.0
        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                loss = self.cell_cross_ent[k](cell_pred[k], cell_prop_vals[:, n].to(torch.int64))
                cell_loss += (loss * cell_mask[:, n]).mean()
            else:
                loss = self.cell_mse[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n])
                cell_loss += (loss * cell_mask[:, n]).mean()

        if self.training:
            self.log("cell_loss_train", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("cell_loss_val", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return cell_loss

    def training_step(self, batch, batch_idx):

        self.train_step += 1

        gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, _, group_idx, _ = batch

        # qz_m is the mean of the latent, qz_v is the variance, and z is the sampled latent
        qz_m, qz_v, z = self.network.z_encoder(gene_vals, batch_labels, batch_mask)

        # same as above, but for library size
        library_size_per_cell = torch.log(gene_vals.sum(dim=1, keepdim=True))

        px_scale, px_r, px_rate, px_dropout = self.network.decoder(
            self.dispersion, z, library_size_per_cell, batch_labels, batch_mask,
        )

        if self.cell_properties is not None:
            cell_pred = self.network.cell_decoder(z, batch_labels, batch_mask)
            cell_loss = self._cell_loss(cell_pred, cell_prop_vals, cell_mask, group_idx)
        else:
            cell_loss = 0.0

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.network.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.network.px_r)
        elif self.dispersion == "gene":
            px_r = self.network.px_r
        px_r = torch.exp(px_r)

        # Loss
        gene_loss, kl_z = self.gene_loss(gene_vals, qz_m, qz_v, px_rate, px_r, px_dropout)
        loss = self.gene_loss_coeff * gene_loss + cell_loss

        if self.l1_lambda > 0:
            l1_regularization = 0.0
            for name, param in self.network.named_parameters():
                if 'z_encoder' in name and "bias" not in name:
                    l1_regularization += torch.norm(param, p=1)
            loss += self.l1_lambda * l1_regularization

        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("cell_loss", cell_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def get_reconstruction_loss(
            self,
            x,
            px_rate,
            px_r,
            px_dropout,
            **kwargs,
    ) -> torch.Tensor:
        """Return the reconstruction loss (for a minibatch)"""

        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.reconstruction_loss == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.reconstruction_loss == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss


    def configure_optimizers(self):

        encoder_params = []
        other_params = []
        for n, p in self.network.named_parameters():
            if "z_encoder" in n:
                encoder_params.append(p)
            else:
                other_params.append(p)

        opt = torch.optim.AdamW(
            [
                {'params': encoder_params, "weight_decay": self.weight_decay},
                {'params': other_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.998),
            eps=1e-7,
        )

        lr_scheduler = {
            'scheduler': WarmupConstantAndDecaySchedule(opt, warmup_steps=self.warmup_steps),
            'interval': 'step',
        }

        return {
           'optimizer': opt,
           'lr_scheduler': lr_scheduler, # Changed scheduler to lr_scheduler
           'interval': 'step',
       }

class WarmupConstantAndDecaySchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.alpha = np.sqrt(warmup_steps)
        super(WarmupConstantAndDecaySchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        arg1 = (step + 1) ** (-0.5)
        arg2 = (step + 1) * (self.warmup_steps ** -1.5)
        return self.alpha * np.minimum(arg1, arg2)


class WarmupSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return np.clip(step / self.warmup_steps, 0.0, 1.0)


class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.alpha = alpha
        print(f"FOCAL LOSS, num classes {num_classes}")

    def _convert_target(self, target):

        target = torch.clip(target, 0, self.num_classes - 1)

        if self.num_classes == 2:
            new_target = (
                0.95 * F.one_hot(target, num_classes=self.num_classes) +
                0.05 * F.one_hot(1 - target, num_classes=self.num_classes)
            )
        else:
            new_target = 0.9 * F.one_hot(target, num_classes=self.num_classes)
            t0 = torch.clip(target - 1, 0, self.num_classes - 1)
            t1 = torch.clip(target + 1, 0, self.num_classes - 1)
            new_target += 0.05 * F.one_hot(t0, num_classes=self.num_classes)
            new_target += 0.05 * F.one_hot(t1, num_classes=self.num_classes)

        return new_target


    def forward(self, input, target):

        new_target = self._convert_target(target)
        # new_target = F.one_hot(target, num_classes=self.num_classes)
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        loss = - new_target * log_prob * (1 - prob) ** self.gamma
        return self.alpha * loss.sum(dim=-1)
