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

from aurora_backend.tools.aurora_tools import AuroraDecodeStateTool


class DecoderAgent(ToolCallAgent):  # type: ignore[misc]
    """Agent that maps manifold states to cognitive interpretations."""

    name: str = "aurora_decoder_agent"
    description: str = (
        "DecoderAgent: given a cluster path/posterior, it decodes a cognitive "
        "state and explanation using Aurora's cognitive decoder."
    )
    system_prompt: str = (
        "You are the DecoderAgent inside Aurora. "
        "When you receive a cluster path and posterior, call tools to decode "
        "a cognitive state and produce human-readable explanations."
    )
    max_steps: int = 3

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager([AuroraDecodeStateTool()])
    )
