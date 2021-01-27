from typing import Optional

import torch
import torch.nn as nn
from asteroid.losses.mse import SingleSrcMSE
from asteroid.losses.sdr import SingleSrcNegSDR
from asteroid.losses.stoi import NegSTOILoss as SingleSrcNegSTOI
from asteroid.models.conv_tasnet import ConvTasNet


def loss_contrastive(estimates_1: torch.Tensor, sources_1: torch.Tensor,
                     estimates_2: torch.Tensor, sources_2: torch.Tensor,
                     labels: torch.BoolTensor,
                     distance_fn: torch.nn.modules.loss._Loss) -> torch.Tensor:
    r"""Custom contrastive loss function on positive and negative pairs.

    Parameters
    ----------
    estimates_1 : torch.Tensor
    estimates_2 : torch.Tensor
        Denoised waveforms; shape = (batch_idx, signal_idx, length)
    sources_1 : torch.Tensor
    sources_2 : torch.Tensor
        Ground truth waveforms; shape = (batch_idx, signal_idx, length)
    labels : torch.BoolTensor
        Booleans indicating positive or negative pairing; shape = (batch_idx, 1)
    distance_fn : torch.nn.modules.loss._Loss
        PyTorch loss function
    """
    if not isinstance(distance_fn, torch.nn.modules.loss._Loss):
        raise ValueError('Distance metric must be a valid PyTorch loss.')

    lambda_positive = 5e-1
    lambda_negative = 2e-4

    # remove singleton dimensions
    estimates_1 = estimates_1.squeeze()
    estimates_2 = estimates_2.squeeze()
    sources_1 = sources_1.squeeze()
    sources_2 = sources_2.squeeze()

    # calculate standard source separation loss terms
    l_source_separation = (
        distance_fn(estimates_1, sources_1) +
        distance_fn(estimates_2, sources_2)
    )

    # calculate positive pair agreement loss terms
    l_positive = lambda_positive * (
        distance_fn(estimates_1[labels], estimates_2[labels])
    )

    # calculate negative pair disagreement loss terms
    l_negative = lambda_negative * (
        - distance_fn(estimates_1[~labels], estimates_2[~labels])
        + distance_fn(sources_1[~labels], sources_2[~labels])
    )**2

    return l_source_separation + l_positive + l_negative


class NetworkRNN(nn.Module):
    # STFT parameters
    fft_size: int = 1024
    hop_length: int = 256

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create a neural network which predicts a TF binary ratio mask
        self.encoder = nn.GRU(
            input_size=int(self.fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_size,
                out_features=int(self.fft_size // 2 + 1)
            ),
            nn.Sigmoid()
        )
        self.window = nn.Parameter(torch.hann_window(self.fft_size), False)
        self.name = (self.__class__.__name__ +
                     f'_{hidden_size:04d}x{num_layers:02d}')

    def stft(self, waveform: torch.Tensor):
        """Calculates the Short-time Fourier transform (STFT)."""

        # perform the short-time Fourier transform
        spectrogram = torch.stft(
            waveform, self.fft_size, self.hop_length, window=self.window
        )

        # swap seq_len & feature_dim of the spectrogram (for RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # calculate the magnitude spectrogram
        magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                           spectrogram[..., 1] ** 2)

        return spectrogram, magnitude_spectrogram

    def istft(self, spectrogram: torch.Tensor,
              mask: Optional[torch.Tensor] = None):
        """Calculates the inverse Short-time Fourier transform (ISTFT)."""

        # apply a time-frequency mask if provided
        if mask is not None:
            spectrogram[..., 0] *= mask
            spectrogram[..., 1] *= mask

        # swap seq_len & feature_dim of the spectrogram (undo RNN processing)
        spectrogram = spectrogram.permute(0, 2, 1, 3)

        # perform the inverse short-time Fourier transform
        waveform = torch.istft(
            spectrogram, self.fft_size, self.hop_length, window=self.window
        )

        return waveform

    def forward(self, waveform):
        # convert waveform to spectrogram
        (X, X_magnitude) = self.stft(waveform)

        # generate a time-frequency mask
        H = self.encoder(X_magnitude)[0]
        Y = self.decoder(H)
        Y = Y.reshape_as(X_magnitude)

        # convert masked spectrogram back to waveform
        denoised = self.istft(X, mask=Y)
        residual = self.istft(X.clone(), mask=(1 - Y.clone()))

        return denoised, residual


class NetworkCTN(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = ConvTasNet(n_src=2,
            n_filters=128,       # N
                                 # L?
            bn_chan=128,         # B
            hid_chan=256,        # H
            skip_chan=128,       # Sc
            conv_kernel_size=3,  # P
            n_blocks=7,          # X
            n_repeats=2,         # R
            sample_rate=16000,
        )
        self.name = self.__class__.__name__

    def forward(self, waveform):
        output = self.network(waveform)
        denoised = output[..., 0, :]
        residual = output[..., 1, :]
        return denoised, residual
