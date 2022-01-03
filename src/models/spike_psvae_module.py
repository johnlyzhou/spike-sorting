import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.models.spike_vae_module import SpikeSortingVAE
from src.models.utils.loss_metrics import gaussian_nll, decomposed_kl_divergence, kl_divergence
from src.models.components import PSVAE


class SpikeSortingPSVAE(SpikeSortingVAE):
    def __init__(self, config: dict):
        super().__init__(config)
        self.anneal_epochs = self.config.get("anneal_epochs", 0)
        self.alpha = self.config.get("alpha", 1.0)
        self.beta = self.config.get("beta", 1.0)
        if self.anneal_epochs > 0:
            self.beta_annealing = np.linspace(0, self.beta, self.anneal_epochs)
            self.gamma_annealing = np.linspace(0, 1, self.anneal_epochs)

        self.get_beta = lambda e: self.beta if e >= self.anneal_epochs else self.beta_annealing[e]
        self.get_gamma = lambda e: 1.0 if e >= self.anneal_epochs else self.gamma_annealing[e]

    def init_model(self, encoder, decoder, model_config):
        self.model = PSVAE(
            encoder,
            decoder,
            model_config["encoder_output_dim"],
            model_config["latent_dim"],
            model_config["label_dim"]
        )

    def prepare_data(self):
        data_config = self.config["data"]
        train_templates = np.load(data_config["train_data_path"])
        train_labels = np.load(data_config["train_label_path"])
        val_templates = np.load(data_config["val_data_path"])
        val_labels = np.load(data_config["val_label_path"])

        x_train = torch.from_numpy(train_templates).float()
        y_train = torch.from_numpy(train_labels).float()
        x_val = torch.from_numpy(val_templates).float()
        y_val = torch.from_numpy(val_labels).float()
        self.train_dataset = TensorDataset(x_train, y_train)
        self.val_dataset = TensorDataset(x_val, y_val)

    def alpha(self):
        return self.config.get("alpha", 1.0)

    def beta(self):
        if self.anneal_epochs > 0:
            beta = self.config["beta"]
            if self.current_epoch >= self.anneal_epochs:
                return 1.0
            else:
                return np.linspace(0, beta, self.anneal_epochs)[self.current_epoch]

    def loss(self, batch, model_outputs):
        x, y = batch
        supervised_latents, unsupervised_latents, z, log_var, y_hat, x_hat = \
            model_outputs

        beta = self.get_beta(self.current_epoch)
        gamma = self.get_gamma(self.current_epoch)

        data_reconstruction_loss = gaussian_nll(x, x_hat)
        label_reconstruction_loss = self.alpha * gaussian_nll(y, y_hat)

        n_labels = supervised_latents.shape[1]
        supervised_latents_kld = kl_divergence(supervised_latents, log_var[:, :n_labels])

        # Unsupervised latents KLD
        index_code_mutual_information, total_correlation, dimension_wise_kld = \
            decomposed_kl_divergence(z[:, n_labels:], unsupervised_latents, log_var[:, n_labels:])

        unsupervised_latents_kld = (
            gamma * index_code_mutual_information +
            beta * total_correlation +
            gamma * dimension_wise_kld
        )

        loss = (
            data_reconstruction_loss +
            label_reconstruction_loss +
            supervised_latents_kld +
            unsupervised_latents_kld
        ).mean()

        return (
            loss,
            data_reconstruction_loss.mean(),
            label_reconstruction_loss.mean(),
            supervised_latents_kld.mean(),
            unsupervised_latents_kld.mean()
        )

    def training_step(self, batch, _):
        (
            loss,
            data_reconstruction_loss,
            label_reconstruction_loss,
            supervised_latents_kld,
            unsupervised_latents_kld
        ) = self.loss(batch, self.model(batch[0]))

        self.log_dict({
            'train_loss': loss,
            'train_data_recon_loss': data_reconstruction_loss,
            'train_label_recon_loss': label_reconstruction_loss,
            'train_supervised_latents_kld': supervised_latents_kld,
            'train_unsupervised_latents_kld': unsupervised_latents_kld,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        (
            loss,
            data_reconstruction_loss,
            label_reconstruction_loss,
            supervised_latents_kld,
            unsupervised_latents_kld
        ) = self.loss(batch, self.model(batch[0]))

        self.log_dict({
            'val_loss': loss,
            'val_data_recon_loss': data_reconstruction_loss,
            'val_label_recon_loss': label_reconstruction_loss,
            'val_supervised_latents_kld': supervised_latents_kld,
            'val_unsupervised_latents_kld': unsupervised_latents_kld,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss
