
cell_properties = {
    "Dementia": {"discrete": True, "values": [0, 1], "stop_grad": False},
    "BRAAK_AD": {"discrete": False, "values": [-1], "stop_grad": False},
    "SubID": {"discrete": True, "values": None,  "stop_grad": True},
}

batch_properties = None

dataset_cfg = {
    "data_path": "/home/masse/work/data/mssm_rush_v2/data1.dat",
    "metadata_path": "/home/masse/work/data/mssm_rush_v2/metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 512,
    "num_workers": 14,
    "batch_properties": batch_properties,
    "remove_sex_chrom": True,
    "protein_coding_only": True,
    "top_k_genes": 25_000,
    "cell_restrictions": {"class": "Astro"},
    "mixup": False,
    "group_balancing": "bd",
}

model_cfg = {
    "n_layers": 2,
    "n_hidden": 512,
    "n_hidden_decoder": 512,
    "n_latent": 32,
    "n_latent_cell_decoder": 32,
    "dropout_rate": 0.5,
    "input_dropout_rate": 0.25,
    "chromosome_dropout": False,
    "standard_dropout": True,
    "grad_reverse_dict": {
        "SubID": 0., "Age": 0., "Sex": 0., "Brain_bank": 0., "PMI": 0.,
    },
    "cell_decoder_hidden_layer": False,
}

task_cfg = {
    "learning_rate": 2e-4,
    "warmup_steps": 2000.0,
    "weight_decay": 0.05,
    "l1_lambda": 0.0,
    "gene_loss_coeff": 1e-3,
    "balance_classes": False,
    "n_epochs_kl_warmup": None,
    "batch_properties": batch_properties,
}

trainer_cfg = {
    "n_devices": 1,
    "grad_clip_value": 0.25,
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
}