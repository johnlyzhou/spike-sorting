from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.models.spike_vae_module import SpikeSortingVAE
from src.models.spike_psvae_module import SpikeSortingPSVAE


def get_latents(system_cls, model_fname, templates_fname):
    model = system_cls.load_from_checkpoint(model_fname)
    outputs = model(torch.tensor(np.load(templates_fname)).float())

    if system_cls == SpikeSortingPSVAE:
        supervised_latents = outputs[0].detach().numpy()
        unsupervised_latents = outputs[1].detach().numpy()
        latent_rep_np = np.hstack((supervised_latents, unsupervised_latents))
    elif system_cls == SpikeSortingVAE:
        latent_rep_np = outputs[0].detach().numpy()
    else:
        raise NotImplementedError("Only implemented for PS-VAE and VAE!")

    return latent_rep_np


def plot_latent_features_vs_position(latent_reps, positions, data_dir, vary_feature="x"):
    """
    Plot and save a scatter plot of values for each latent space dimension as a function of a specific positional
    feature value (x, y, z, or alpha).
    """
    num_dims = latent_reps.shape[1]
    plt.figure()
    [xs, zs, ys, alphas] = positions
    if vary_feature == "x":
        x = xs
    elif vary_feature == "z":
        x = zs
    elif vary_feature == "y":
        x = ys
    elif vary_feature == "alpha":
        x = alphas
    else:
        raise ValueError("Invalid feature specified! Choose one of \'x\', \'y\', \'z\', or \'alpha\'.")
    for dim in range(num_dims):
        y = latent_reps[:, dim]
        plt.scatter(x, y, s=5, cmap="viridis", label=dim)
    plt.title("{} Position vs. Latent Dimensions".format(vary_feature))
    plt.xlabel(vary_feature)
    plt.ylabel("Latent Dimension Values")
    plt.legend(bbox_to_anchor=(1.1, 1.05))

    Path("{}/position_invariance".format(data_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig("{}/position_invariance/{}_vs_{}_latents.png".format(data_dir, vary_feature, num_dims))
