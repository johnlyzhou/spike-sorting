import numpy as np
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt

from src.models.spike_vae_module import SpikeSortingVAE
from src.models.spike_psvae_module import SpikeSortingPSVAE


def get_reconstructions(system_cls, model_fname, templates_fname):
    if system_cls == SpikeSortingPSVAE:
        ss_model = SpikeSortingPSVAE.load_from_checkpoint(model_fname)
    elif system_cls == SpikeSortingVAE:
        ss_model = SpikeSortingVAE.load_from_checkpoint(model_fname)
    else:
        raise NotImplementedError("Only implemented for PS-VAE and VAE!")

    x = np.load(templates_fname)
    x = torch.tensor(x).float()
    _, _, recon_latents = ss_model.model(x)
    recon_latents_np = recon_latents.detach().numpy()

    return recon_latents_np


def plot_template_reconstructions(og_templates, reconstructed_templates, n_channels=20, n_samples=20, out_fname=None):
    """
    Plot and save a specified number of sample reconstructed waveforms overlaid against the original waveforms.
    """
    pdf = mpl.backends.backend_pdf.PdfPages(out_fname)

    for i in range(n_samples):
        fig = plt.figure(figsize=(n_channels, 2.5))
        plt.plot(reconstructed_templates[i, :80, :].T.flatten(), color='red')
        plt.plot(og_templates[i, :80, :].T.flatten(), color='blue')
        for j in range(n_channels - 1):
            plt.axvline(80 + 80 * j, color='black')
        plt.title("Reconstructed: {}".format(i))
        pdf.savefig(fig)

    pdf.close()
