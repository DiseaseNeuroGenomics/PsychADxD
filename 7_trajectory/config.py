
cell_properties = {
    "Dementia": {"discrete": True, "values": [0, 1], "stop_grad": False},
    "BRAAK_AD": {"discrete": False, "values": [-1], "stop_grad": False},
    "SubID": {"discrete": True, "values": None,  "stop_grad": True},
}

batch_properties = None

dataset_cfg = {
    "data_path": None, # ADD PATH HERE
    "metadata_path": None, # ADD PATH HERE
    "cell_properties": cell_properties,
    "batch_size": 256,
    "num_workers": 10,
    "batch_properties": batch_properties,
    "remove_sex_chrom": True,
    "protein_coding_only": True,
    "top_k_genes": 10_000,
    "cell_restrictions": {"class": "Immune"},
    "mixup": False,
    "group_balancing": "bd",
}

model_cfg = {
    "n_layers": 2,
    "n_hidden": 512,
    "dropout_rate": 0.5,
    "input_dropout_rate": 0.0,
    "grad_reverse_dict": {
        "SubID": 0., "Age": 0., "Sex": 0., "Brain_bank": 0., "PMI": 0.,
    },
    "cell_decoder_hidden_layer": False,
}

task_cfg = {
    "learning_rate": 2e-2,
    "momentum": 0.0,
    "warmup_steps": 2000.0,
    "weight_decay": 0.0,
    "balance_classes": False,
    "label_smoothing": 0.0,
    "batch_properties": batch_properties,
    "perturb_gene_names": None,
}

trainer_cfg = {
    "splits_path": None, # ADD PATH HERE
    "n_devices": 1,
    "grad_clip_value": 0.5,
    "accumulate_grad_batches": 1,
    "precision": "32-true",
    "max_epochs": 20,
}