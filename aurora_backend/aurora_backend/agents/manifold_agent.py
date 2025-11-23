from __future__ import annotations

from typing import Any

from pydantic import Field

try:
    from spoon_ai.agents import ToolCallAgent
    from spoon_ai.tools import ToolManager
except ImportError:  # pragma: no cover
    class ToolCallAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "spoon_ai is not installed; ToolCallAgent is unavailable."
            )

    class ToolManager(list):  # type: ignore[no-redef]
        pass

from aurora_backend.tools.aurora_tools import (
    AuroraEncodeTool,
    AuroraUpdateHierarchyTool,
    AuroraManifoldStatsTool,
)


class ManifoldAgent(ToolCallAgent):  # type: ignore[misc]
    """Agent that maintains and analyzes the cognitive manifold."""

    name: str = "aurora_manifold_agent"
    description: str = (
        "ManifoldAgent: updates the hierarchical manifold, performs pruning, "
        "and reports structure and statistics over cluster levels."
    )
    system_prompt: str = (
        "You are the ManifoldAgent inside Aurora. "
        "Given raw signals or embeddings, you call tools to update and inspect "
        "the hierarchical cognitive manifold."
    )
    max_steps: int = 4

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager(
            [
                AuroraEncodeTool(),
                AuroraUpdateHierarchyTool(),
                AuroraManifoldStatsTool(),
            ]
        )
    )
