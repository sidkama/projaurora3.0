from dataclasses import dataclass

@dataclass
class AuroraConfig:
    """Configuration for the Aurora Synaptic Engine backend."""

    # Signal settings
    sample_rate_hz: int = 128
    window_seconds: float = 2.0
    eeg_channels: int = 8
    bio_channels: int = 2  # HR + EDA

    # Latent model
    latent_dim: int = 64
    d_model: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4

    # Manifold / hierarchy
    hierarchy_depth: int = 3
    k_per_level: int = 5
    posterior_alpha: float = 0.95  # EMA for posterior mass
    temperature: float = 1.0
    pruning_threshold: float = 0.01

    # Temporal smoothing
    smoothing_window: int = 5

    # Dataset construction
    init_dataset_samples_per_task: int = 50

DEFAULT_CONFIG = AuroraConfig()
