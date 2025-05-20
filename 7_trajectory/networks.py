# modified from https://github.com/tabdelaal/scVI/blob/master/scvi/models/vae.py
from typing import Any, Dict, List, Optional, Tuple

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, CellDecoder, DecoderSCVI, FCLayers

# VAE model
class VAE(nn.Module):


    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_hidden_decoder: int = 128,
        n_hidden_library: int = 128,
        n_latent: int = 10,
        n_latent_cell_decoder: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        input_dropout_rate: float = 0.0,
        gene_chrom_dict: Optional[Dict] = None,
        chromosome_dropout: bool = False,
        standard_dropout: bool = False,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        latent_distribution: str = "normal",
        grad_reverse_dict: Optional[Dict] = None,
        cell_decoder_hidden_layer: bool = False,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            batch_properties=batch_properties,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            input_dropout_rate=input_dropout_rate,
            gene_chrom_dict=gene_chrom_dict,
            chromosome_dropout=chromosome_dropout,
            standard_dropout=standard_dropout,
            distribution=latent_distribution,
        )

        if cell_properties is not None:
            self.cell_decoder = CellDecoder(
                n_latent_cell_decoder,
                cell_properties,
                batch_properties=batch_properties,
                grad_reverse_dict=grad_reverse_dict,
                use_hidden_layer=cell_decoder_hidden_layer,
                dropout_rate=dropout_rate,
            )

        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden_decoder,
            batch_properties=None,
            dropout_rate=dropout_rate,
        )

class FeedforwardNetwork(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_layers: int = 1,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        input_dropout_rate: float = 0.0,
        log_normalize: bool = True,
        cell_decoder_hidden_layer: bool = False,
        layer_norm: bool = True,
        normalize_total: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.log_normalize = log_normalize
        self.normalize_total = normalize_total

        if n_layers > 0:
            self.encoder = FCLayers(
                n_in=n_input,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
                batch_properties=batch_properties,
            )


        self.cell_decoder = CellDecoder(
            latent_dim=n_hidden if n_layers > 0 else n_input,
            cell_properties=cell_properties,
            use_hidden_layer=cell_decoder_hidden_layer,
        )

        self.drop = nn.Dropout(p=float(input_dropout_rate)) if input_dropout_rate > 0  else nn.Identity()


    def forward(self, x: torch.Tensor, batch_labels: torch.Tensor, batch_mask: torch.Tensor):

        if self.normalize_total:
            x = 10_000 * x / x.sum(dim=1, keepdim=True)

        if self.log_normalize:
            x = torch.log(1.0 + x)

        x = self.drop(x)

        if self.n_layers > 0:
            x = self.encoder(x)

        return self.cell_decoder(x), x


def load_model(model_save_path, model):

    params_loaded = []
    non_network_params = []
    state_dict = {}
    ckpt = torch.load(model_save_path)
    key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    for k, v in ckpt[key].items():

        if "cell_property" in k:
            non_network_params.append(k)
        elif "network" in k:
            k = k.split(".")
            k = ".".join(k[1:])

        for n, p in model.named_parameters():
            if n == k:
                pass
                # print(k, p.size(), v.size(), p.size() == v.size())
            if n == k and p.size() == v.size():
                state_dict[k] = v
                params_loaded.append(n)

    model.load_state_dict(state_dict, strict=True)
    print(f"Number of params loaded: {len(params_loaded)}")
    print(f"Non-network parameters not loaded: {non_network_params}")
    return model