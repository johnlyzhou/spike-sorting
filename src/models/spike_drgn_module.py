from functools import reduce
from operator import mul
from typing import Tuple

import numpy as np
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from src.models.utils.loss_metrics import gaussian_nll, kl_divergence

NP2_X_BOUNDS = (-150, 182)
NP2_Y_BOUNDS = (0, 150)


# Works for Conv1d and MaxPool1d
def conv_1d_shape(l_in, kernel_size, stride=1, padding=0, dilation=1):
    l_out = int(np.floor((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    return l_out


# Works for Conv2d and MaxPool2d
def conv_2d_shape(h_in, w_in, kernel_size, stride=1, padding=0, dilation=1):
    h = int(np.floor((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    w = int(np.floor((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    return h, w


class DRGNEncoder(nn.Module):
    def __init__(self, in_channels, kernel=5, stride=2, out_channels_1=32, out_channels_2=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_1, kernel, stride=stride),
            nn.BatchNorm1d(out_channels_1),
            nn.LeakyReLU(0.05),
            nn.Conv1d(out_channels_1, out_channels_2, kernel, stride=stride),
            nn.BatchNorm1d(out_channels_2),
            nn.LeakyReLU(0.05),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        return self.encoder(x)


class DRGNDecoder(nn.Module):
    def __init__(self, in_channels, n_latents, encoder_output_dims, kernel=5, stride=2, out_channels_2=16):
        super().__init__()

        flattened_dims = reduce(mul, encoder_output_dims)

        self.decoder = nn.Sequential(
            nn.Linear(n_latents - 4, flattened_dims),
            nn.Unflatten(1, encoder_output_dims),
            nn.ConvTranspose1d(out_channels_2, out_channels_2, kernel, stride=stride),
            nn.BatchNorm1d(out_channels_2),
            nn.LeakyReLU(0.05),
            nn.ConvTranspose1d(out_channels_2, in_channels, kernel, stride=stride),
            nn.BatchNorm1d(in_channels),
            nn.Tanh()
        )

    def forward(self, y):
        return self.decoder(y)


class DRGN(nn.Module):
    def __init__(self, encoder, decoder, l_in, n_latents, kernel=5, stride=2, out_channels_2=16):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.squash = nn.Tanh()

        l_out = conv_1d_shape(l_in, kernel, stride=stride)
        l_out = conv_1d_shape(l_out, kernel, stride=stride)
        encoder_output_dims = (out_channels_2, l_out)

        flattened_dims = reduce(mul, encoder_output_dims)
        self.mean = nn.Linear(flattened_dims, n_latents)
        self.log_var = nn.Linear(flattened_dims, n_latents)

    def limit_bounds(self, x, bounds: Tuple):
        x = self.squash(x)
        l, u = bounds
        x = x / (u - l) + l
        return x

    @staticmethod
    def sample(mean, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps * std + mean

    def encode(self, w):
        w = self.encoder(w)
        return self.mean(w), self.log_var(w)

    def decode(self, j):
        x, y, z, alpha = j[0, :4]
        x = self.limit_bounds(x, NP2_X_BOUNDS)
        y = self.limit_bounds(y, NP2_Y_BOUNDS)
        shape = j[:, 4:]
        ptp = alpha / (((torch.FloatTensor([x, z])) ** 2).sum() + y ** 2) ** 0.5
        j = self.decoder(shape)
        return j * ptp

    def forward(self, w):
        mean, log_var = self.encode(w)
        j = self.sample(mean, log_var)
        return mean, log_var, self.decode(j)


class SpikeDRGN(LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        model_config = config["model"]
        encoder = DRGNEncoder(model_config["in_channels"],
                              kernel=model_config["kernel"],
                              stride=model_config["stride"],
                              out_channels_1=model_config["out_channels_1"],
                              out_channels_2=model_config["out_channels_2"]
                              )
        decoder = DRGNDecoder(model_config["in_channels"],
                              model_config["latent_dim"],
                              model_config["encoder_output_dims"],
                              kernel=model_config["kernel"],
                              stride=model_config["stride"],
                              out_channels_2=model_config["out_channels_2"]
                              )
        self.model = self.init_model(encoder, decoder, model_config)
        self.train_dataset, self.val_dataset = self.prepare_data()

    @staticmethod
    def init_model(encoder, decoder, model_config: dict):
        return DRGN(
            encoder,
            decoder,
            model_config["l_in"],
            model_config["latent_dim"],
            kernel=model_config["kernel"],
            stride=model_config["stride"],
            out_channels_2=model_config["out_channels_2"]
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=(self.config["learning_rate"] or 1e-4))

    def prepare_data(self) -> Tuple[TensorDataset, TensorDataset]:
        data_config = self.config["data"]
        train_templates = np.load(data_config["train_data_path"])
        val_templates = np.load(data_config["val_data_path"])

        x_train = torch.from_numpy(train_templates).float()
        x_val = torch.from_numpy(val_templates).float()
        train_dataset = TensorDataset(x_train, x_train)
        val_dataset = TensorDataset(x_val, x_val)
        return train_dataset, val_dataset

    def train_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.config["data"]["train_batch_size"])

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, batch_size=self.config["data"]["val_batch_size"])

    @staticmethod
    def loss(batch, outputs):
        x, _ = batch
        mean, log_var, x_hat = outputs

        reconstruction_loss = gaussian_nll(x, x_hat)
        kld = kl_divergence(mean, log_var)
        elbo = (reconstruction_loss + kld).mean()

        return (
            elbo,
            reconstruction_loss.mean(),
            kld.mean()
        )

    def training_step(self, batch, batch_idx):
        output = self.model(batch[0])
        elbo, reconstruction_loss, kld = self.loss(batch, output)

        self.log_dict({
            'train_loss': elbo,
            'train_recon_loss': reconstruction_loss,
            'train_kld': kld
        }, on_step=False, on_epoch=True, prog_bar=True)

        return elbo

    def validation_step(self, batch, batch_idx):
        output = self.model(batch[0])
        elbo, reconstruction_loss, kld = self.loss(batch, output)

        self.log_dict({
            'val_loss': elbo,
            'val_recon_loss': reconstruction_loss,
            'val_kld': kld
        }, prog_bar=True)

        return elbo
