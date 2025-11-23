from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover - sklearn might not be installed in some envs
    KMeans = None  # type: ignore


@dataclass
class ManifoldNode:
    """Node in the hierarchical manifold tree."""

    node_id: str
    level: int
    parent_id: Optional[str]
    centroid: np.ndarray
    children: List[str] = field(default_factory=list)
    posterior_mass: float = 0.0
    label: Optional[str] = None
    notes: Optional[str] = None
    point_indices: List[int] = field(default_factory=list)


class HierarchicalManifold:
    """Hierarchical clustering and pruning over latent space.

    This implementation is intentionally simple but captures the key behaviors:
    - Offline construction from a latent dataset
    - Hierarchical clustering up to a fixed depth
    - Online inference of a path through the tree
    - Exponential moving average of per-node posterior mass
    - Pruning of low-mass leaf nodes
    """

    def __init__(self, latent_dim: int, alpha: float = 0.95) -> None:
        if KMeans is None:
            raise ImportError(
                "scikit-learn is required for HierarchicalManifold but could not be imported."
            )
        self.latent_dim = latent_dim
        self.alpha = float(alpha)
        self.nodes: Dict[str, ManifoldNode] = {}
        self.root_ids: List[str] = []
        self.max_level: int = 0
        self._is_built: bool = False

    # ------------------------------------------------------------------
    @property
    def is_built(self) -> bool:
        return self._is_built

    # ------------------------------------------------------------------
    def build_from_dataset(
        self,
        Z: np.ndarray,
        depth: int = 2,
        k_per_level: int = 5,
        random_state: int = 0,
    ) -> None:
        """Build the tree from a full latent dataset.

        Parameters
        ----------
        Z:
            Latent vectors, shape (N, D).
        depth:
            Maximum depth of the hierarchy (>= 1).
        k_per_level:
            Desired number of clusters per level (per parent).
        """
        if Z.ndim != 2:
            raise ValueError("Z must have shape (N, D).")
        if Z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {Z.shape[1]}."
            )

        self.nodes.clear()
        self.root_ids.clear()
        self.max_level = int(depth)
        n_samples = Z.shape[0]
        all_indices = np.arange(n_samples)

        # Level 1 clustering over all points
        k1 = int(min(k_per_level, max(1, n_samples)))
        root_kmeans = KMeans(n_clusters=k1, random_state=random_state, n_init=10)
        root_assign = root_kmeans.fit_predict(Z)

        for i in range(k1):
            idxs = all_indices[root_assign == i]
            if idxs.size == 0:
                continue
            node_id = f"L1_{i}"
            centroid = Z[idxs].mean(axis=0)
            node = ManifoldNode(
                node_id=node_id,
                level=1,
                parent_id=None,
                centroid=centroid,
                point_indices=list(map(int, idxs)),
            )
            self.nodes[node_id] = node
            self.root_ids.append(node_id)

        # Deeper levels
        if depth > 1:
            for level in range(2, depth + 1):
                parent_candidates = [
                    nid for nid, n in self.nodes.items() if n.level == level - 1
                ]
                for parent_id in parent_candidates:
                    parent_node = self.nodes[parent_id]
                    parent_idxs = np.array(parent_node.point_indices, dtype=int)
                    if parent_idxs.size < k_per_level * 2:
                        # Too few points to split meaningfully
                        continue

                    k = int(min(k_per_level, max(1, parent_idxs.size)))
                    kmeans = KMeans(
                        n_clusters=k,
                        random_state=random_state + level,
                        n_init=10,
                    )
                    assignments = kmeans.fit_predict(Z[parent_idxs])
                    for j in range(k):
                        mask = assignments == j
                        if not np.any(mask):
                            continue
                        child_idxs = parent_idxs[mask]
                        child_id = f"L{level}_{parent_id}_{j}"
                        centroid = Z[child_idxs].mean(axis=0)
                        child_node = ManifoldNode(
                            node_id=child_id,
                            level=level,
                            parent_id=parent_id,
                            centroid=centroid,
                            point_indices=list(map(int, child_idxs)),
                        )
                        self.nodes[child_id] = child_node
                        parent_node.children.append(child_id)

        self._is_built = True

    # ------------------------------------------------------------------
    def infer_posterior(
        self,
        z: np.ndarray,
        temp: float = 1.0,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Infer a hierarchical path and soft posterior for a single latent vector.

        Returns
        -------
        path:
            List of node IDs chosen greedily from root to leaf.
        posterior:
            Dictionary of node_id -> accumulated probability mass at this step.
        """
        if not self._is_built:
            raise RuntimeError("Manifold not built yet. Call build_from_dataset first.")

        if z.ndim == 2:
            if z.shape[0] != 1:
                raise ValueError("z must be a single vector (D,) or (1, D).")
            z = z[0]
        if z.ndim != 1:
            raise ValueError("z must be a single latent vector of shape (D,)")

        z = z.astype(np.float64)

        path: List[str] = []
        posterior: Dict[str, float] = {}

        current_ids = list(self.root_ids)
        while current_ids:
            centroids = np.stack(
                [self.nodes[nid].centroid for nid in current_ids], axis=0
            ).astype(np.float64)
            dists = np.sum((centroids - z[None, :]) ** 2, axis=1)

            dists = dists - float(dists.min())
            temp_val = max(float(temp), 1e-6)
            logits = -dists / temp_val
            logits = logits - float(logits.max())
            exp_logits = np.exp(logits)
            probs = exp_logits / float(exp_logits.sum())

            best_idx = int(np.argmax(probs))
            best_node_id = current_ids[best_idx]
            path.append(best_node_id)

            for nid, p in zip(current_ids, probs):
                p_float = float(p)
                node = self.nodes[nid]
                node.posterior_mass = self.alpha * node.posterior_mass + (
                    1.0 - self.alpha
                ) * p_float
                posterior[nid] = posterior.get(nid, 0.0) + p_float

            child_ids = self.nodes[best_node_id].children
            current_ids = list(child_ids)

        return path, posterior

    # ------------------------------------------------------------------
    def prune(self, min_mass: float) -> List[str]:
        """Prune leaf nodes with posterior mass below min_mass.

        Returns the list of pruned node IDs.
        """
        pruned: List[str] = []
        for node_id, node in list(self.nodes.items()):
            if node.posterior_mass < min_mass and not node.children:
                pruned.append(node_id)
                parent_id = node.parent_id
                if parent_id and parent_id in self.nodes:
                    parent = self.nodes[parent_id]
                    parent.children = [c for c in parent.children if c != node_id]
                if node_id in self.root_ids:
                    self.root_ids = [r for r in self.root_ids if r != node_id]
                del self.nodes[node_id]
        return pruned

    # ------------------------------------------------------------------
    def stats_by_level(self) -> Dict[int, int]:
        """Return a mapping level -> number of active nodes."""
        stats: Dict[int, int] = {}
        for node in self.nodes.values():
            stats[node.level] = stats.get(node.level, 0) + 1
        return stats

    # ------------------------------------------------------------------
    def all_leaf_ids(self) -> List[str]:
        """Return IDs of all current leaves in the tree."""
        return [nid for nid, n in self.nodes.items() if not n.children]
