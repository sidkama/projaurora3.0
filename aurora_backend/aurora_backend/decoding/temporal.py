from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TemporalSmoother:
    """Simple majority-vote temporal smoother over cluster IDs."""

    window: int = 5
    history: List[str] = field(default_factory=list)

    def update(self, cluster_id: str) -> str:
        self.history.append(cluster_id)
        if len(self.history) > self.window:
            self.history.pop(0)

        counts = {}
        for cid in self.history:
            counts[cid] = counts.get(cid, 0) + 1
        # Return the cluster ID with maximum count
        best_id = max(counts.items(), key=lambda kv: kv[1])[0]
        return best_id

    def reset(self) -> None:
        self.history.clear()


@dataclass
class TemporalStabilityTracker:
    """Track stability between raw and smoothed states over time."""

    raw_history: List[str] = field(default_factory=list)
    smooth_history: List[str] = field(default_factory=list)

    def update(self, raw_id: str, smooth_id: str) -> None:
        self.raw_history.append(raw_id)
        self.smooth_history.append(smooth_id)

    def reset(self) -> None:
        self.raw_history.clear()
        self.smooth_history.clear()

    def consistency(self) -> float:
        if not self.raw_history or not self.smooth_history:
            return 0.0
        n = min(len(self.raw_history), len(self.smooth_history))
        if n == 0:
            return 0.0
        matches = 0
        for i in range(n):
            if self.raw_history[i] == self.smooth_history[i]:
                matches += 1
        return matches / float(n)
