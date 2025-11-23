from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import AuroraConfig, DEFAULT_CONFIG
from .data.synthetic_generator import SyntheticEEGGenerator
from .models.encoder import build_encoder, encode_batch
from .manifold.hierarchy import HierarchicalManifold
from .decoding.temporal import TemporalSmoother, TemporalStabilityTracker
from .decoding.decoder import CognitiveDecoder


@dataclass
class AuroraRuntime:
    """Runtime container for the Aurora Synaptic Engine backend."""

    config: AuroraConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    device: str = "cpu"

    generator: SyntheticEEGGenerator | None = None
    encoder: object | None = None  # AuroraEncoder, but keep loose typing for import-safety
    manifold: HierarchicalManifold | None = None
    smoother: TemporalSmoother | None = None
    stability_tracker: TemporalStabilityTracker | None = None
    decoder: CognitiveDecoder | None = None

    def initialize(self) -> None:
        # Create synthetic data generator
        self.generator = SyntheticEEGGenerator(
            sample_rate_hz=self.config.sample_rate_hz,
            window_seconds=self.config.window_seconds,
            eeg_channels=self.config.eeg_channels,
            bio_channels=self.config.bio_channels,
            random_state=42,
        )

        # Build encoder
        self.encoder = build_encoder(
            eeg_channels=self.config.eeg_channels,
            bio_channels=self.config.bio_channels,
            latent_dim=self.config.latent_dim,
            d_model=self.config.d_model,
            num_heads=self.config.transformer_heads,
            num_layers=self.config.transformer_layers,
            device=self.device,
        )

        # Construct an initial dataset and build the manifold offline
        eeg_data, bio_data, labels = self.generator.generate_dataset(
            n_per_task=self.config.init_dataset_samples_per_task
        )
        Z_list: List[np.ndarray] = []
        for i in range(eeg_data.shape[0]):
            z = encode_batch(self.encoder, eeg_data[i], bio_data[i], device=self.device)
            Z_list.append(z[0])
        Z = np.stack(Z_list, axis=0)

        self.manifold = HierarchicalManifold(
            latent_dim=self.config.latent_dim,
            alpha=self.config.posterior_alpha,
        )
        self.manifold.build_from_dataset(
            Z,
            depth=self.config.hierarchy_depth,
            k_per_level=self.config.k_per_level,
            random_state=0,
        )

        # Temporal components
        self.smoother = TemporalSmoother(window=self.config.smoothing_window)
        self.stability_tracker = TemporalStabilityTracker()

        # Decoder
        self.decoder = CognitiveDecoder()

        # Optionally auto-label leaf clusters by majority label of their training points
        self._auto_label_clusters(Z, labels)

    # ------------------------------------------------------------------
    def _auto_label_clusters(self, Z: np.ndarray, labels: List[str]) -> None:
        if self.manifold is None or self.decoder is None:
            return

        # Map from node_id to list of labels assigned to that node
        node_label_map: Dict[str, List[str]] = {}
        for idx, z in enumerate(Z):
            path, _ = self.manifold.infer_posterior(z)
            leaf_id = path[-1] if path else None
            if leaf_id is None:
                continue
            node_label_map.setdefault(leaf_id, []).append(labels[idx])

        for node_id, node_labels in node_label_map.items():
            if not node_labels:
                continue
            # Majority vote
            counts: Dict[str, int] = {}
            for lab in node_labels:
                counts[lab] = counts.get(lab, 0) + 1
            majority_label = max(counts.items(), key=lambda kv: kv[1])[0]
            self.decoder.set_label(
                node_id,
                label=majority_label,
                notes="Auto-labeled by majority of synthetic training labels.",
            )

    # ------------------------------------------------------------------
    def acquire_window(
        self, task_label: str | None = None
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        if self.generator is None:
            raise RuntimeError("AuroraRuntime not initialized.")
        eeg, bio, label = self.generator.sample_window(task_label)
        return eeg, bio, label

    def encode_window(self, eeg: np.ndarray, bio: np.ndarray) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("AuroraRuntime not initialized.")
        z = encode_batch(self.encoder, eeg, bio, device=self.device)
        return z[0]

    def update_manifold(
        self, z: np.ndarray
    ) -> Tuple[List[str], Dict[str, float], List[str]]:
        if self.manifold is None:
            raise RuntimeError("AuroraRuntime not initialized.")
        path, posterior = self.manifold.infer_posterior(
            z, temp=self.config.temperature
        )
        # Return list of pruned nodes as well
        pruned = self.manifold.prune(self.config.pruning_threshold)
        return path, posterior, pruned

    def decode_state(
        self, cluster_path: List[str], posterior: Dict[str, float]
    ) -> Dict[str, object]:
        if self.decoder is None:
            raise RuntimeError("AuroraRuntime not initialized.")

        raw_leaf = cluster_path[-1] if cluster_path else "none"
        if self.smoother is not None and self.stability_tracker is not None:
            smooth_leaf = self.smoother.update(raw_leaf)
            self.stability_tracker.update(raw_leaf, smooth_leaf)
        else:
            smooth_leaf = raw_leaf

        decoded = self.decoder.decode(cluster_path, posterior)
        decoded["smoothed_leaf_id"] = smooth_leaf
        decoded["temporal_consistency"] = (
            self.stability_tracker.consistency()
            if self.stability_tracker is not None
            else 0.0
        )
        return decoded

    def manifold_stats(self) -> Dict[str, object]:
        if self.manifold is None:
            raise RuntimeError("AuroraRuntime not initialized.")
        stats_by_level = self.manifold.stats_by_level()
        return {
            "levels": stats_by_level,
            "total_nodes": int(sum(stats_by_level.values())),
        }


_RUNTIME: AuroraRuntime | None = None


def get_runtime() -> AuroraRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = AuroraRuntime()
        _RUNTIME.initialize()
    return _RUNTIME
