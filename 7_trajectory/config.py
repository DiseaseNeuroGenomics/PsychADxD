
cell_properties = {
    "Dementia": {"discrete": True, "values": [0, 1], "stop_grad": False},
    "BRAAK_AD": {"discrete": False, "values": [-1], "stop_grad": False},
    "Sex": {"discrete": True, "values": ["Male", "Female"], "stop_grad": True},
    "Brain_bank": {"discrete": True, "values": ["MSSM", "RUSH"], "stop_grad": True},
    "Age": {"discrete": False, "values": [-1], "stop_grad": True},
    "PMI": {"discrete": False, "values": [-1], "stop_grad": True},
    "SubID": {"discrete": True, "values": None,  "stop_grad": True},
}

batch_properties = None

dataset_cfg = {
    "data_path": None, # this needs to be changed to location of processed dataset (.dat file)
    "metadata_path": None, # this needs to be changed to location of metadata (.pkl file)
    "cell_properties": cell_properties,
    "batch_size": 512,
    "num_workers": 8,
    "batch_properties": batch_properties,
    "remove_sex_chrom": True,
    "cell_restrictions": {"class": "Astro"},
    "mixup": False,
    "group_balancing": "bd",
}

model_cfg = {
    "n_layers": 2,
    "n_hidden": 512,
    "n_hidden_decoder": 512,
    "n_hidden_library": 64,
    "n_latent": 32,
    "n_latent_cell_decoder": 32,
    "dropout_rate": 0.5,
    "input_dropout_rate": 0.5,
    "grad_reverse_dict": {
        "SubID": 0.2, "Brain_bank": 0.2, "Age": 0.2, "Sex": 0.2, "PMI": 0.2,
    },
    "cell_decoder_hidden_layer": False,
}

task_cfg = {
    "learning_rate": 5e-4,
    "warmup_steps": 2000.0,
    "weight_decay": 0.0,
    "l1_lambda": 0.0,
    "gene_loss_coeff": 2e-3,
    "balance_classes": False,
    "n_epochs_kl_warmup": None,
    "batch_properties": batch_properties,
    "save_gene_vals": False,
    "use_gdro": False,
}

trainer_cfg = {
    "n_devices": 1,
    "grad_clip_value": 0.25,
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
}