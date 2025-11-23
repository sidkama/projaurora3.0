from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from spoon_ai.tools.base import BaseTool
except ImportError:  # pragma: no cover - spoon-core might not be installed in some envs
    class BaseTool:  # type: ignore[no-redef]
        """Fallback BaseTool so the module still imports without spoon-core.

        In a real SpoonOS environment, spoon_ai.tools.base.BaseTool will be used.
        """

        name: str = "dummy"
        description: str = "Dummy tool (spoon_ai not installed)."
        parameters: Dict[str, Any] = {}

        async def execute(self, **kwargs: Any) -> Any:  # pragma: no cover
            raise RuntimeError("spoon_ai is not installed; this is a dummy BaseTool.")


from aurora_backend.runtime import get_runtime


class AuroraAcquireWindowTool(BaseTool):
    name: str = "aurora_acquire_window"
    description: str = (
        "Acquire a synthetic EEG+biometric window from the Aurora generator."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_label": {
                "type": "string",
                "description": "Optional task label to condition the synthetic data "
                '(e.g., "rest", "motor_imagery", "visuospatial", "working_memory").',
            }
        },
    }

    async def execute(self, task_label: Optional[str] = None) -> Dict[str, Any]:
        runtime = get_runtime()
        eeg, bio, label = runtime.acquire_window(task_label)
        return {
            "task_label": label,
            "eeg": eeg.tolist(),
            "bio": bio.tolist(),
        }


class AuroraEncodeTool(BaseTool):
    name: str = "aurora_encode"
    description: str = "Encode raw EEG+biometric signals into a latent vector z_t."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "eeg": {
                "type": "array",
                "description": "EEG array shaped (C_eeg, T).",
            },
            "bio": {
                "type": "array",
                "description": "Biometric array shaped (C_bio, T).",
            },
        },
        "required": ["eeg", "bio"],
    }

    async def execute(
        self,
        eeg: List[List[float]],
        bio: List[List[float]],
    ) -> Dict[str, Any]:
        import numpy as np

        runtime = get_runtime()
        eeg_arr = np.asarray(eeg, dtype="float32")
        bio_arr = np.asarray(bio, dtype="float32")
        z = runtime.encode_window(eeg_arr, bio_arr)
        return {"embedding": z.tolist()}


class AuroraUpdateHierarchyTool(BaseTool):
    name: str = "aurora_update_hierarchy"
    description: str = (
        "Update the hierarchical manifold with a new latent vector and "
        "return the inferred cluster path, posterior, and any pruned nodes."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "embedding": {
                "type": "array",
                "description": "Latent vector z_t in R^D.",
            }
        },
        "required": ["embedding"],
    }

    async def execute(self, embedding: List[float]) -> Dict[str, Any]:
        import numpy as np

        runtime = get_runtime()
        z = np.asarray(embedding, dtype="float32")
        path, posterior, pruned = runtime.update_manifold(z)
        return {
            "cluster_path": path,
            "posterior": posterior,
            "pruned_nodes": pruned,
        }


class AuroraDecodeStateTool(BaseTool):
    name: str = "aurora_decode_state"
    description: str = (
        "Decode a cognitive state from a cluster path and posterior over nodes."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "cluster_path": {
                "type": "array",
                "description": "List of node IDs from root to leaf.",
            },
            "posterior": {
                "type": "object",
                "description": "Mapping node_id -> posterior mass.",
            },
        },
        "required": ["cluster_path", "posterior"],
    }

    async def execute(
        self,
        cluster_path: List[str],
        posterior: Dict[str, float],
    ) -> Dict[str, Any]:
        runtime = get_runtime()
        decoded = runtime.decode_state(cluster_path, posterior)
        return decoded


class AuroraManifoldStatsTool(BaseTool):
    name: str = "aurora_manifold_stats"
    description: str = "Return global statistics about the hierarchical manifold."
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self) -> Dict[str, Any]:
        runtime = get_runtime()
        return runtime.manifold_stats()
