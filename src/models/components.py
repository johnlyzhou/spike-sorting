from operator import mul
from functools import reduce
from math import sqrt

import torch
from torch import nn
from scipy.stats import ortho_group


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, conv_encoder_layers, use_batch_norm=False):
        super().__init__()

        layers = []
        for (out_channels, kernel, stride) in conv_encoder_layers:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel, stride=stride))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(0.05))

            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_features,
                 encoder_output_dim,
                 conv_decoder_layers,
                 use_batch_norm=False):
        """1-D Convolutional Decoder

        Args:
            in_features (int): Number of input features
            encoder_output_dim (list): Shape of the unflattened output from the encoder
                (num_channels x ...)
            conv_decoder_layers (list):List of tuples specifying ConvTranspose1d layers
            use_batch_norm (bool, optional): Whether to add BatchNorm1d layers in 
                between ConvTranspose1d layers. Defaults to False.
        """
        super().__init__()

        layers = [
            nn.Linear(in_features, reduce(mul, encoder_output_dim)),
            nn.Unflatten(1, encoder_output_dim)
        ]

        in_channels = encoder_output_dim[0]
        for i, (out_channels, kernel, stride, output_padding) in enumerate(conv_decoder_layers):
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    output_padding=output_padding))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            
            # Don't add activation for last layer
            if i != len(conv_decoder_layers) - 1:
                layers.append(nn.LeakyReLU(0.05))

            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AE(nn.Module):
    def __init__(self, encoder, decoder, encoder_output_dim, encoding_dim):
        super(AE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        flattened_dim = reduce(mul, encoder_output_dim)
        self.encoding = nn.Linear(flattened_dim, encoding_dim)

    def encode(self, x):
        output = torch.flatten(self.encoder(x), start_dim=1)
        return self.encoding(output)
    
    def decode(self, encoding):
        return self.decoder(encoding)
    
    def forward(self, x):
        return self.decode(self.encode(x))


class VAE(nn.Module):
    def __init__(self, encoder, decoder, encoder_output_dim, latent_dim):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        flattened_dim = reduce(mul, encoder_output_dim)
        self.latent_mean = nn.Linear(flattened_dim, latent_dim)
        self.latent_log_var = nn.Linear(flattened_dim, latent_dim)

    def encode(self, x):
        output = torch.flatten(self.encoder(x), start_dim=1)
        return self.latent_mean(output), self.latent_log_var(output)

    def sample(self, mean, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps * std + mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sample(mean, log_var)
        return mean, log_var, self.decode(z)


class DiagLinear(nn.Module):
    """Borrowed from https://github.com/themattinthehatt/behavenet"""

    __constants__ = ['n_features']

    def __init__(self, n_features: int, bias=True):
        super(DiagLinear, self).__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.tensor(n_features).float())
        if bias:
            self.bias = nn.Parameter(torch.tensor(n_features).float())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / sqrt(self.n_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            bound = 1 / sqrt(self.n_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input.mul(self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return 'n_features={}, bias={}'.format(self.n_features, self.bias is not None)


class PSVAE(VAE):
    def __init__(self, encoder, decoder, encoder_output_dim, latent_dim, label_dim):
        super().__init__(encoder, decoder, encoder_output_dim, latent_dim)

        self.A = nn.Linear(latent_dim, label_dim, bias=False)
        self.B = nn.Linear(latent_dim, latent_dim - label_dim, bias=False)
        self.D = DiagLinear(label_dim, bias=True)

        m = ortho_group.rvs(dim=latent_dim).astype('float32')
        self.A.weight = nn.Parameter(
            torch.from_numpy(m[:label_dim, :]), requires_grad=False
        )
        self.B.weight = nn.Parameter(
            torch.from_numpy(m[label_dim:, :]), requires_grad=False
        )

    def encode(self, x):
        encoder_output = torch.flatten(self.encoder(x), start_dim=1)
        mean, log_var = (
            self.latent_mean(encoder_output),
            self.latent_log_var(encoder_output)
        )
        supervised_latents = self.A(mean)
        unsupervised_latents = self.B(mean)
        return supervised_latents, unsupervised_latents, log_var

    def sample(self, mean, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps * std + mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        supervised_latents, unsupervised_latents, log_var = self.encode(x)
        mean = torch.cat([supervised_latents, unsupervised_latents], dim=1)
        z = self.sample(mean, log_var)
        x_hat = self.decode(z)
        y_hat = self.D(supervised_latents)
        return supervised_latents, unsupervised_latents, z, log_var, y_hat, x_hat
