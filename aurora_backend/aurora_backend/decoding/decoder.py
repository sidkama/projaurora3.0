from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CognitiveDecoder:
    """Map hierarchical cluster paths to human-interpretable cognitive states."""

    cluster_labels: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def set_label(self, cluster_id: str, label: str, notes: str | None = None) -> None:
        self.cluster_labels[cluster_id] = {
            "label": label,
            "notes": notes or "",
        }

    def decode(
        self,
        cluster_path: List[str],
        posterior: Dict[str, float],
    ) -> Dict[str, object]:
        if not cluster_path:
            return {
                "state": "unknown",
                "cluster_id": None,
                "confidence": 0.0,
                "explanation": "No cluster path was available for decoding.",
            }

        leaf_id = cluster_path[-1]
        info = self.cluster_labels.get(leaf_id)

        if info is None:
            state = f"unlabeled_state_{leaf_id}"
            explanation = (
                "This cluster has not been labeled yet. "
                "Use the NeuroscientistAgent to assign a cognitive interpretation."
            )
        else:
            state = info["label"]
            explanation = info.get("notes") or (
                "Cognitive state inferred from majority labels assigned to this cluster."
            )

        # Use posterior mass at the leaf as a proxy for confidence
        confidence = float(posterior.get(leaf_id, 0.0))
        return {
            "state": state,
            "cluster_id": leaf_id,
            "confidence": confidence,
            "explanation": explanation,
        }
