from operator import mul
from functools import reduce

import torch
from torch import nn


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