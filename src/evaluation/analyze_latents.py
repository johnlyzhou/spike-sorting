from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.models.spike_vaes import SpikeSortingVAE


def get_latents(model_fname, templates_fname):
    ss_model = SpikeSortingVAE.load_from_checkpoint(model_fname)
    x = np.load(templates_fname)
    x = torch.tensor(x).float()
    latent_rep, _ = ss_model.model.encode(x)
    latent_rep_np = latent_rep.detach().numpy()

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
