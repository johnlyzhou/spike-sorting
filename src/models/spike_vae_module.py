import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset

from src.models.components import ConvEncoder, ConvDecoder, VAE
from src.models.utils.loss_metrics import gaussian_nll, kl_divergence


class SpikeSortingVAE(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        model_config = config["model"]
        encoder = ConvEncoder(model_config["in_channels"],
                              model_config["conv_encoder_layers"],
                              use_batch_norm=model_config.get("use_batch_norm", True))
        decoder = ConvDecoder(model_config["latent_dim"], 
                              model_config["encoder_output_dim"],
                              model_config["conv_decoder_layers"],
                              use_batch_norm=model_config.get("use_batch_norm", True))
        self.init_model(encoder, decoder, model_config)

    def init_model(self, encoder, decoder, model_config: dict):
        self.model = VAE(
            encoder,
            decoder,
            model_config["encoder_output_dim"],
            model_config["latent_dim"]
        )

    def prepare_data(self):
        data_config = self.config["data"]
        train_templates = np.load(data_config["train_data_path"])
        val_templates = np.load(data_config["val_data_path"])
        
        x_train = torch.from_numpy(train_templates).float()
        x_val = torch.from_numpy(val_templates).float()
        self.train_dataset = TensorDataset(x_train, x_train)
        self.val_dataset = TensorDataset(x_val, x_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config["data"]["train_batch_size"])
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.config["data"]["val_batch_size"])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=(self.config["learning_rate"] or 1e-4))

    def loss(self, batch, outputs):
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
