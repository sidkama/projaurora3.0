from __future__ import annotations

from typing import Any

from pydantic import Field

try:
    from spoon_ai.agents import ToolCallAgent
    from spoon_ai.tools import ToolManager
except ImportError:  # pragma: no cover - spoon-core not installed in some envs
    class ToolCallAgent:  # type: ignore[no-redef]
        """Fallback agent base to keep imports working without spoon-core."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "spoon_ai is not installed; ToolCallAgent is unavailable."
            )

    class ToolManager(list):  # type: ignore[no-redef]
        pass

from aurora_backend.tools.aurora_tools import AuroraAcquireWindowTool


class AcquisitionAgent(ToolCallAgent):  # type: ignore[misc]
    """SpoonOS agent responsible for acquiring neural/biometric windows."""

    name: str = "aurora_acquisition_agent"
    description: str = (
        "AcquisitionAgent: pulls synthetic neural and biometric data windows "
        "for the Aurora Synaptic Engine."
    )
    system_prompt: str = (
        "You are the AcquisitionAgent in the Aurora Synaptic Engine. "
        "When asked for data, call tools to acquire the next EEG+biometric window."
    )
    max_steps: int = 2

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager([AuroraAcquireWindowTool()])
    )
