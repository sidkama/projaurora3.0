from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from spoon_ai.agents import ToolCallAgent
    from spoon_ai.tools import ToolManager
    from spoon_ai.chat import ChatBot
except ImportError:  # pragma: no cover
    class ToolCallAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "spoon_ai is not installed; ToolCallAgent is unavailable."
            )

    class ToolManager(list):  # type: ignore[no-redef]
        pass

    class ChatBot:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("spoon_ai is not installed; ChatBot is unavailable.")

from aurora_backend.runtime import get_runtime


@dataclass
class ClusterSummary:
    node_id: str
    level: int
    label: Optional[str]
    notes: Optional[str]
    posterior_mass: float


def summarize_manifold() -> List[ClusterSummary]:
    """Return simple summaries of all manifold nodes."""
    runtime = get_runtime()
    if runtime.manifold is None:
        raise RuntimeError("Runtime manifold not initialized.")
    summaries: List[ClusterSummary] = []
    for node_id, node in runtime.manifold.nodes.items():
        summaries.append(
            ClusterSummary(
                node_id=node_id,
                level=node.level,
                label=node.label,
                notes=node.notes,
                posterior_mass=node.posterior_mass,
            )
        )
    summaries.sort(key=lambda s: (s.level, -s.posterior_mass))
    return summaries


class NeuroscientistAgent(ToolCallAgent):  # type: ignore[misc]
    """High-level agent for scientific analysis and reporting.

    In a full SpoonOS deployment, this agent would use the unified LLM
    infrastructure (ChatBot/LLMManager) to write reports, generate hypotheses,
    and compare manifold structure across experiments. Here we primarily define
    its prompt and available context so it can be plugged into graphs/workflows.
    """

    name: str = "aurora_neuroscientist_agent"
    description: str = (
        "NeuroscientistAgent: coordinates experiments, analyzes manifolds, and "
        "writes structured reports for human researchers."
    )
    system_prompt: str = (
        "You are the NeuroscientistAgent operating Aurora Synaptic Engine. "
        "You have access to summaries of hierarchical latent manifolds, temporal "
        "stability metrics, and decoded cognitive states. You help scientists "
        "design experiments, compare conditions, and articulate hypotheses."
    )
    max_steps: int = 6

    # No direct tools here yet; in practice you could expose analysis/report tools.
    available_tools: ToolManager = Field(default_factory=ToolManager)  # type: ignore[call-arg]
