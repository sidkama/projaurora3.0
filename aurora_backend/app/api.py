from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from aurora_backend.runtime import get_runtime
from aurora_backend.graph.pipeline import run_pipeline_step, AuroraState

app = FastAPI(
    title="Aurora Synaptic Engine v2 Backend",
    description=(
        "Backend API for the Aurora Synaptic Engine v2. "
        "Provides endpoints for synthetic data streaming, latent encoding, "
        "hierarchical manifold exploration, and cognitive decoding. "
        "Integrates with SpoonOS via spoon_ai.graph.StateGraph and custom agents/tools."
    ),
)


# ----------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    message: str


class EncodeRequest(BaseModel):
    eeg: List[List[float]]
    bio: List[List[float]]


class EncodeResponse(BaseModel):
    embedding: List[float]


class ManifoldUpdateRequest(BaseModel):
    embedding: List[float]


class ManifoldUpdateResponse(BaseModel):
    cluster_path: List[str]
    posterior: Dict[str, float]
    pruned_nodes: List[str]


class DecodeRequest(BaseModel):
    cluster_path: List[str]
    posterior: Dict[str, float]


class DecodeResponse(BaseModel):
    state: str
    cluster_id: Optional[str]
    confidence: float
    explanation: str
    smoothed_leaf_id: str
    temporal_consistency: float


class ManifoldStatsResponse(BaseModel):
    levels: Dict[int, int]
    total_nodes: int


class PipelineStepResponse(BaseModel):
    state: AuroraState  # type: ignore[valid-type]


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", message="Aurora backend is running.")


@app.post("/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest) -> EncodeResponse:
    import numpy as np

    runtime = get_runtime()
    eeg = np.asarray(req.eeg, dtype="float32")
    bio = np.asarray(req.bio, dtype="float32")
    z = runtime.encode_window(eeg, bio)
    return EncodeResponse(embedding=z.tolist())


@app.post("/manifold/update", response_model=ManifoldUpdateResponse)
async def manifold_update(req: ManifoldUpdateRequest) -> ManifoldUpdateResponse:
    import numpy as np

    runtime = get_runtime()
    z = np.asarray(req.embedding, dtype="float32")
    path, posterior, pruned = runtime.update_manifold(z)
    return ManifoldUpdateResponse(
        cluster_path=path,
        posterior=posterior,
        pruned_nodes=pruned,
    )


@app.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest) -> DecodeResponse:
    runtime = get_runtime()
    decoded = runtime.decode_state(req.cluster_path, req.posterior)
    return DecodeResponse(**decoded)  # type: ignore[arg-type]


@app.get("/manifold/stats", response_model=ManifoldStatsResponse)
async def manifold_stats() -> ManifoldStatsResponse:
    runtime = get_runtime()
    stats = runtime.manifold_stats()
    return ManifoldStatsResponse(**stats)  # type: ignore[arg-type]


@app.post("/pipeline/step", response_model=PipelineStepResponse)
async def pipeline_step(state: Optional[AuroraState] = None) -> PipelineStepResponse:
    """Run one full acquisition→encoding→manifold→decoding step.

    This endpoint is ideal for driving a live UI demo: each call advances the
    simulated experiment by one time step and returns the full state.
    """
    new_state = await run_pipeline_step(state)
    return PipelineStepResponse(state=new_state)
