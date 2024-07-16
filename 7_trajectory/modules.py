import collections
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from functional import GradReverseLayer
import numpy as np

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    return x

def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)



class LinearWithMask(nn.Module):

    def __init__(self, n_input: int, n_output: int, mask:torch.Tensor):
        super().__init__()
        # self.ff = nn.Linear(n_input, n_output)
        self.W = nn.parameter.Parameter(
            data=torch.randn(n_input, n_output),
            requires_grad=True,
        )
        self.b = nn.parameter.Parameter(
            data=torch.zeros((1, n_output), dtype=torch.float32),
            requires_grad=True,
        )
        self.mask = mask

    def forward(self, x):
        W = self.W * self.mask.to(x.device)
        return x @ W + self.b


class ResidualLayer(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        dropout_rate: float = 0.1,
    ):

        super().__init__()
        self.ff0 = nn.Linear(n_in, n_hidden)
        self.ff1 = nn.Linear(n_hidden, n_in)
        self.ln = nn.LayerNorm(n_in, elementwise_affine=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=float(dropout_rate))
        nn.init.xavier_uniform_(self.ff0.weight, gain=0.25)
        nn.init.xavier_uniform_(self.ff1.weight, gain=0.25)

    def forward(self, input: torch.Tensor):

        x = self.ff0(input)
        x = self.act(x)
        x = self.drop(x)
        x = self.ff1(x)


        return self.ln(input + x)

class FCLayers(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        batch_properties: Optional[Dict[str, Any]] = None,
        layer_norm: bool = True,
    ):
        super().__init__()
        # we will inject the batch variables into the first hidden layer
        self.batch_dims = 0 if batch_properties is None else [
            len(batch_properties[k]["values"]) for k in batch_properties.keys()
        ]

        layers_dim = [n_in + np.sum(self.batch_dims)] + n_layers * [n_hidden]
        self.batch_drop = nn.Dropout(p=float(dropout_rate))

        layers = []
        for n in range(n_layers):
            layers.append(nn.Linear(layers_dim[n], layers_dim[n + 1]))
            layers.append(nn.ReLU())
            if layer_norm:
                layers.append(nn.LayerNorm(n_hidden, elementwise_affine=True))
            layers.append(nn.Dropout(p=float(dropout_rate)))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, batch_vals: torch.Tensor, batch_mask: torch.Tensor):

        if np.sum(self.batch_dims) > 0:
            batch_vals[batch_vals < 0] = 0
            batch_vars = []
            for n in range(len(self.batch_dims)):
                b = F.one_hot(batch_vals[:, n], num_classes=self.batch_dims[n])
                b = b * batch_mask[:, n: n+1]
                batch_vars.append(b)

            x = torch.cat((x, *batch_vars), dim=-1)

        return self.fc_layers(x)


class ChromDropout(nn.Module):
    # currently not used

    def __init__(self, p: float, gene_chrom_dict: Dict, n_input: int):
        super().__init__()
        print("Initializing chromosome dropout")
        self.gene_chrom_dict = {}
        for n, (k, v) in enumerate(gene_chrom_dict.items()):
            y = torch.ones(n_input, dtype=torch.float32)
            y[v] = 0.0
            self.gene_chrom_dict[torch.tensor(n)] = y.to("cuda")

        self.num_chroms = len(self.gene_chrom_dict.keys())
        self.N = int(p * self.num_chroms)
        self.ratio = self.N / self.num_chroms

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.N < 1e-6:
            return input
        else:
            for n in range(input.size(0)):
                idx = torch.randperm(self.num_chroms)[:self.N]
                for i in idx:
                    k = list(self.gene_chrom_dict.keys())[i]
                    input[n, :] = input[n, :] * self.gene_chrom_dict[k]

            return input / self.ratio


# Encoder
class Encoder(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        batch_properties: Optional[Dict[str, Any]] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.0,
        input_dropout_rate: float = 0.0,
        gene_chrom_dict: Optional[Dict] = None,
        chromosome_dropout: bool = False, # currently not used
        standard_dropout: bool = True,
        distribution: str = "normal",
        shared_variance: bool = True,
    ):
        super().__init__()

        self.distribution = distribution
        self.encoder = FCLayers(
            n_in=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            batch_properties=batch_properties,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, 1) if shared_variance else nn.Linear(n_hidden, n_output)
        self.gene_chrom_dict = gene_chrom_dict
        self.softplus = nn.Softplus()

        if chromosome_dropout:
            self.drop_chrom = ChromDropout(float(input_dropout_rate), gene_chrom_dict, n_input)
        else:
            self.drop_chrom = nn.Identity()
        if standard_dropout:
            self.drop_standard = nn.Dropout(p=float(input_dropout_rate))
        else:
            self.drop_standard = nn.Identity()
    def forward(self, input: torch.Tensor, batch_labels: torch.Tensor, batch_mask: torch.Tensor):

        x = torch.log(1.0 + input)
        x = self.drop_chrom(x)
        x = self.drop_standard(x)

        # Parameters for latent distribution
        q = self.encoder(x, batch_labels, batch_mask)
        q_m = self.mean_encoder(q)
        q_v = self.softplus(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


class CellDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        cell_properties: Dict[str, Any],
        batch_properties: Optional[Dict[str, Any]] = None,
        grad_reverse_dict: Optional[Dict] = None,
        use_hidden_layer: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.batch_dims = 0 if batch_properties is None else [
            len(batch_properties[k]["values"]) for k in batch_properties.keys()
        ]
        self.latent_dim = latent_dim + np.sum(self.batch_dims)
        print("Batch and latent dims", np.sum(self.batch_dims) , self.latent_dim)

        self.cell_properties = cell_properties
        self.cell_mlp = nn.ModuleDict()
        self.batch_drop = nn.Dropout(p=float(dropout_rate))

        self.grad_reverse_dict = grad_reverse_dict
        if grad_reverse_dict is not None:
            self.grad_reverse = nn.ModuleDict()
            for k, v in grad_reverse_dict.items():
                print(f"Grad reverse {k}: {v}")
                self.grad_reverse[k] = GradReverseLayer(v)

        for k, cell_prop in cell_properties.items():
            # the output size of the cell property prediction MLP will be 1 if the property is continuous;
            # if it is discrete, then it will be the length of the possible values
            n_targets = 1 if not cell_prop["discrete"] else len(cell_prop["values"])
            if use_hidden_layer:
                print("Cell decoder hidden layer set to TRUE")
                self.cell_mlp[k] = nn.Sequential(
                    nn.Linear(self.latent_dim, 2 * self.latent_dim),
                    nn.ReLU(),
                    #nn.Dropout(p=float(dropout_rate)),
                    nn.Linear(2 * self.latent_dim, n_targets),
                )
            else:
                print("Cell decoder hidden layer set to FALSE")
                self.cell_mlp[k] = nn.Linear(self.latent_dim, n_targets, bias=True)


    def forward(self, latent: torch.Tensor, batch_vals: torch.Tensor, batch_mask: torch.Tensor):

        # Predict cell properties
        if np.sum(self.batch_dims) > 0:
            batch_vals[batch_vals < 0] = 0
            batch_vars = []
            for n in range(len(self.batch_dims)):
                b = F.one_hot(batch_vals[:, n], num_classes=self.batch_dims[n])
                b = b * batch_mask[:, n: n+1]
                batch_vars.append(b)

            latent = torch.cat((latent, *batch_vars), dim=-1)

        output = {}
        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
            if k in self.grad_reverse_dict.keys():
                x = self.grad_reverse[k](latent)
            else:
                x = latent.detach() if cell_prop["stop_grad"] else latent

            output[k] = self.cell_mlp[k](x)

        return output


# Decoder
class DecoderSCVI(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        batch_properties: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            batch_properties=batch_properties,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_labels: torch.Tensor,
        batch_mask: torch.Tensor,
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, batch_labels, batch_mask)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

