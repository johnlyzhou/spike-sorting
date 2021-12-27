from operator import mul
from functools import reduce
from math import sqrt

import torch
from torch import nn
from scipy.stats import ortho_group


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