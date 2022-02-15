import unittest

import torch
from omegaconf import OmegaConf
from src.models.spike_drgn_module import SpikeDRGN

REPO_PATH = "/Users/johnzhou/research/spike-sorting"
PROCESS_DATA_DIR = f"{REPO_PATH}/data/processed"

featurize_train_experiment_name = "featurization_train_no_noise"
featurize_val_experiment_name = "featurization_val_no_noise"

train_template_path = f"{PROCESS_DATA_DIR}/{featurize_train_experiment_name}/templates.npy"
val_template_path = f"{PROCESS_DATA_DIR}/{featurize_val_experiment_name}/templates.npy"

base_config = OmegaConf.create({
            "random_seed": 4995,
            "model": {
                "l_in": 121,
                "in_channels": 20,
                "kernel": 5,
                "stride": 2,
                "out_channels_1": 32,
                "out_channels_2": 16,
                "encoder_output_dims": [16, 28],
            },
            "learning_rate": 1e-4,
            "data": {
                "train_data_path": train_template_path,
                "val_data_path": val_template_path,
                "train_batch_size": 100,
                "val_batch_size": 100
            },
            "trainer": {
                "gpus": 1,
                "max_epochs": 100
            }

        })

drgn_configs = [OmegaConf.merge(base_config, c) for c in [
    {
        "name": "drgn_latents_10",
        "model": {
            "latent_dim": 10
        }
    },
    {
        "name": "drgn_latents_8",
        "model": {
            "latent_dim": 8
        }
    },
    {
        "name": "drgn_latents_6",
        "model": {
            "latent_dim": 6
        }
    },
]]

n_channels = 20
timesteps = 121
latents = 10
test = torch.ones((1, n_channels, timesteps))

for config in drgn_configs:
    SpikeNet = SpikeDRGN(OmegaConf.to_container(config))
    SpikeNet.forward(test)
    print("done")

