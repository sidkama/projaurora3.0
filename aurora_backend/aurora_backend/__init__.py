"""Aurora Synaptic Engine v2 backend.

This package implements the computational backend for the Aurora Synaptic Engine:
- Synthetic EEG/biometric data generation
- Multi-modal encoder
- Hierarchical manifold construction & pruning
- Temporal smoothing & cognitive decoding
- SpoonOS (spoon-core) integration via StateGraph and custom agents/tools
"""

from .config import AuroraConfig, DEFAULT_CONFIG
