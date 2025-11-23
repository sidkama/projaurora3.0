from __future__ import annotations

from typing import Dict, List, Optional, TypedDict, Any

try:
    from spoon_ai.graph import StateGraph
except ImportError:  # pragma: no cover
    StateGraph = None  # type: ignore

from aurora_backend.runtime import get_runtime


class AuroraState(TypedDict, total=False):
    """Shared state flowing through the Aurora SpoonGraph pipeline."""

    step_index: int
    task_label: str
    raw_eeg: List[List[float]]
    raw_bio: List[List[float]]
    embedding: List[float]
    cluster_path: List[str]
    posterior: Dict[str, float]
    pruned_nodes: List[str]
    decoded_state: Dict[str, Any]


# ----------------------------------------------------------------------
# Node functions
# ----------------------------------------------------------------------
def node_acquire(state: AuroraState) -> Dict[str, Any]:
    runtime = get_runtime()
    eeg, bio, label = runtime.acquire_window()
    return {
        "step_index": int(state.get("step_index", 0)) + 1,
        "task_label": label,
        "raw_eeg": eeg.tolist(),
        "raw_bio": bio.tolist(),
    }


def node_encode(state: AuroraState) -> Dict[str, Any]:
    import numpy as np

    runtime = get_runtime()
    eeg = np.asarray(state["raw_eeg"], dtype="float32")
    bio = np.asarray(state["raw_bio"], dtype="float32")
    z = runtime.encode_window(eeg, bio)
    return {"embedding": z.tolist()}


def node_update_hierarchy(state: AuroraState) -> Dict[str, Any]:
    import numpy as np

    runtime = get_runtime()
    z = np.asarray(state["embedding"], dtype="float32")
    path, posterior, pruned = runtime.update_manifold(z)
    return {
        "cluster_path": path,
        "posterior": posterior,
        "pruned_nodes": pruned,
    }


def node_decode(state: AuroraState) -> Dict[str, Any]:
    runtime = get_runtime()
    decoded = runtime.decode_state(state["cluster_path"], state["posterior"])
    return {"decoded_state": decoded}


def build_graph():
    """Build the StateGraph pipeline if spoon_ai.graph is available.

    Returns the compiled graph, or None if spoon-core is not installed.
    """
    if StateGraph is None:  # pragma: no cover
        return None

    graph = StateGraph(AuroraState)
    graph.add_node("acquire", node_acquire)
    graph.add_node("encode", node_encode)
    graph.add_node("update_hierarchy", node_update_hierarchy)
    graph.add_node("decode", node_decode)

    graph.add_edge("acquire", "encode")
    graph.add_edge("encode", "update_hierarchy")
    graph.add_edge("update_hierarchy", "decode")
    graph.set_entry_point("acquire")

    compiled = graph.compile()
    return compiled


_COMPILED_GRAPH = build_graph()


async def run_pipeline_step(initial_state: Optional[AuroraState] = None) -> AuroraState:
    """Run a single acquisition→encoding→hierarchy→decoding step.

    This uses the SpoonGraph StateGraph if available; otherwise it falls back
    to directly calling the node functions in sequence.
    """
    state: AuroraState = initial_state or AuroraState()
    if _COMPILED_GRAPH is not None:
        # spoon_ai.graph.StateGraph returns a new state dict
        result = await _COMPILED_GRAPH.invoke(state)
        return AuroraState(**result)  # type: ignore[arg-type]

    # Fallback path (no spoon-core installed)
    s = state.copy()
    s.update(node_acquire(s))
    s.update(node_encode(s))
    s.update(node_update_hierarchy(s))
    s.update(node_decode(s))
    return s  # type: ignore[return-value]
