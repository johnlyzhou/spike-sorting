import numpy as np
import torch

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
