from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch not available in some environments
    torch = None
    nn = None  # type: ignore


class AuroraEncoder(nn.Module):  # type: ignore[misc]
    """Simple 1D conv + transformer encoder for multi-modal signals.

    Inputs are EEG and biometric channels, both as (B, C, T) tensors.
    Output is a latent vector z_t in R^D for each batch element.
    """

    def __init__(
        self,
        in_channels_eeg: int,
        in_channels_bio: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        latent_dim: int = 64,
    ) -> None:
        if nn is None:
            raise ImportError(
                "PyTorch is required for AuroraEncoder but could not be imported."
            )
        super().__init__()
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(in_channels_eeg, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.bio_conv = nn.Sequential(
            nn.Conv1d(in_channels_bio, 16, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.proj = nn.Linear(64 + 16, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_head = nn.Linear(d_model, latent_dim)

    def forward(self, eeg, bio):  # type: ignore[override]
        # eeg, bio: (B, C, T)
        eeg_f = self.eeg_conv(eeg)  # (B, 64, T)
        bio_f = self.bio_conv(bio)  # (B, 16, T)
        x = torch.cat([eeg_f, bio_f], dim=1)  # (B, 80, T)
        x = x.transpose(1, 2)  # (B, T, 80)
        x = self.proj(x)  # (B, T, d_model)
        x = self.transformer(x)  # (B, T, d_model)
        z = x.mean(dim=1)  # (B, d_model)
        z = self.latent_head(z)  # (B, latent_dim)
        return z


def build_encoder(
    eeg_channels: int,
    bio_channels: int,
    latent_dim: int = 64,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    device: str = "cpu",
) -> AuroraEncoder:
    """Construct an AuroraEncoder and move it to the given device."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for AuroraEncoder but could not be imported."
        )
    model = AuroraEncoder(
        in_channels_eeg=eeg_channels,
        in_channels_bio=bio_channels,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        latent_dim=latent_dim,
    )
    return model.to(device)


def encode_batch(
    model: AuroraEncoder,
    eeg: np.ndarray,
    bio: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Encode a batch of numpy arrays into latent vectors.

    Parameters
    ----------
    model:
        AuroraEncoder instance.
    eeg:
        Array of shape (B, C_eeg, T) or (C_eeg, T) for a single example.
    bio:
        Array of shape (B, C_bio, T) or (C_bio, T).

    Returns
    -------
    z :
        Array of shape (B, latent_dim).
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required for AuroraEncoder but could not be imported."
        )

    model.eval()
    with torch.no_grad():
        eeg_t = torch.from_numpy(eeg).float().to(device)
        bio_t = torch.from_numpy(bio).float().to(device)

        if eeg_t.dim() == 2:
            eeg_t = eeg_t.unsqueeze(0)
        if bio_t.dim() == 2:
            bio_t = bio_t.unsqueeze(0)

        z = model(eeg_t, bio_t)
        return z.cpu().numpy()
