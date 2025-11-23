# Aurora Synaptic Engine v2 – Backend

This directory contains the backend implementation for **Aurora Synaptic Engine v2**,
a SpoonOS-native AI4Science system for exploring hierarchical cognitive manifolds.

## Features

- Synthetic EEG + biometric data generator (EEG-like, HR, EDA-style channels)
- Multi-modal encoder (1D conv + Transformer) producing latent vectors \(z_t \in \mathbb{R}^D\)
- Hierarchical manifold engine:
  - Offline construction from a synthetic latent dataset
  - Multi-level KMeans clustering with per-node centroids
  - Posterior mass tracking and dynamic pruning
- Temporal smoothing and stability tracking over cluster assignments
- Cognitive decoder that maps cluster paths to human-readable states
- SpoonOS (spoon-core) integration:
  - `StateGraph` workflow in `aurora_backend/graph/pipeline.py`
  - Custom tools in `aurora_backend/tools/aurora_tools.py`
  - Four conceptual agents in `aurora_backend/agents/`:
    - `AcquisitionAgent`
    - `ManifoldAgent`
    - `DecoderAgent`
    - `NeuroscientistAgent`
- FastAPI backend (`app/api.py`) exposing:
  - `/encode`
  - `/manifold/update`
  - `/decode`
  - `/manifold/stats`
  - `/pipeline/step` for a full acquisition→encoding→manifold→decoding step

## Quickstart

```bash
# 1. Create venv and install deps
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt

# 2. Copy and edit environment
cp .env.example .env
# Fill in any LLM provider keys needed by spoon-core (optional)

# 3. Run the backend
python main.py
```

The API will be available at `http://localhost:8000`.

### Example: Full Pipeline Step

```bash
curl -X POST http://localhost:8000/pipeline/step
```

Response (schema):

```json
{
  "state": {
    "step_index": 1,
    "task_label": "motor_imagery",
    "raw_eeg": [[...]],
    "raw_bio": [[...]],
    "embedding": [...],
    "cluster_path": ["L1_0", "L2_L1_0_3", "..."],
    "posterior": {"L1_0": 0.9, "...": 0.1},
    "pruned_nodes": [],
    "decoded_state": {
      "state": "motor_imagery",
      "cluster_id": "L3_L2_L1_0_3_1",
      "confidence": 0.82,
      "explanation": "...",
      "smoothed_leaf_id": "L3_L2_L1_0_3_1",
      "temporal_consistency": 0.41
    }
  }
}
```

## SpoonOS Integration

This backend is designed to be **SpoonOS-native** using the `spoon-core` framework:

- **Graph System** – we use `spoon_ai.graph.StateGraph` to build an
  acquisition→encoding→manifold→decoding workflow in
  `aurora_backend/graph/pipeline.py`.

- **Tools & Agents** – we define custom tools in
  `aurora_backend/tools/aurora_tools.py` and wire them into
  `ToolCallAgent`-style agents in `aurora_backend/agents/`, matching the
  SpoonOS patterns (`BaseTool`, `ToolManager`, `ToolCallAgent`, `ChatBot`).

If `spoon_ai` is *not* installed, the code falls back to lightweight
compatibility stubs so that the backend can still be imported and run,
but full agent/graph behavior will only be available when `spoon-core`
is installed.

## Scientific Workflow

End-to-end, the system supports the following AI4Science loop:

1. **Acquisition** – generate synthetic EEG+biometric windows with known task labels.
2. **Encoding** – map each window to a latent vector via the multi-modal encoder.
3. **Manifold Construction** – build a hierarchical KMeans tree over the latent space.
4. **Streaming Inference** – for each new latent vector, infer a path through the tree
   and update per-node posterior mass.
5. **Temporal Coherence** – smooth cluster assignments over time and compute stability.
6. **Decoding & Analysis** – map cluster paths to cognitive labels, prune low-mass
   branches, and expose manifold statistics to the NeuroscientistAgent / UI.

This backend is meant to plug directly into a SpoonOS-driven front-end/UI that
renders manifold visualizations, pruning curves, and temporal trajectories in real time.
