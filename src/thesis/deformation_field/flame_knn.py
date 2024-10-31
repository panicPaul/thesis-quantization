""" Get the nearest flame indices."""

import torch
import torch.nn as nn
import torch_geometric
from jaxtyping import Float, Int

from thesis.constants import CANONICAL_FLAME_PARAMS
from thesis.data_management.data_classes import UnbatchedFlameParams
from thesis.flame import FlameHead


class FlameKNN(nn.Module):
    """ Get the nearest flame indices."""

    def __init__(
        self,
        k: int,
        canonical_params: UnbatchedFlameParams | tuple = CANONICAL_FLAME_PARAMS,
    ) -> None:
        """
        Args:
            k (int): The number of nearest neighbors.
        """
        super().__init__()
        self.k = k
        self.register_buffer('position_cache', None)  # (n_gaussians, 3)
        self.register_buffer('recompute_distance_cache', None)  # (n_gaussians, )
        self.register_buffer('index_cache', None)  # (n_gaussians, k)
        flame_head = FlameHead()
        canonical_params = UnbatchedFlameParams(*canonical_params)
        canonical_vertices = flame_head.forward(canonical_params)
        self.canonical_vertices = canonical_vertices.squeeze(0).cuda()  # (n_vertices, 3)
        self.index = torch_geometric.nn.pool.L2KNNIndex(emb=self.canonical_vertices)

    def _gather_per_gaussian(
        self,
        indices: Int[torch.Tensor, "k"],
        feature: Float[torch.Tensor, 'n_vertices dim'],
    ) -> Float[torch.Tensor, "k dim"]:
        """
        Args:
            indices (Int[torch.Tensor, 'k']): The indices of the nearest neighbors.
            feature (Float[torch.Tensor, 'n_vertices dim']): The features of the vertices.

        Returns:
            Float[torch.Tensor, 'k dim']: The features of the nearest neighbors.
        """
        return feature[indices]

    def gather(
        self,
        indices: Int[torch.Tensor, "n_gaussians k"],
        feature: Float[torch.Tensor, "n_vertices dim"],
    ) -> Float[torch.Tensor, "n_gaussians k dim"]:
        """
        Args:
            indices (Int[torch.Tensor, 'n_gaussians k']): The indices of the nearest neighbors.
            feature (Float[torch.Tensor, 'n_vertices dim']): The features of the vertices.

        Returns:
            Float[torch.Tensor, 'n_gaussians k dim']: The features of the nearest neighbors.
        """
        return torch.vmap(self._gather_per_gaussian, in_dims=(0, None))(indices, feature)

    # only gather needs gradients
    @torch.no_grad()
    def forward(
        self,
        means: Float[torch.Tensor, "n_gaussians 3"],
        refresh_cache: bool = False,
    ) -> tuple[Int[torch.Tensor, "n_gaussians k"], float]:
        """
        Args:
            means (Float[torch.Tensor, 'n_gaussians 3']): The means of the gaussians.
            refresh_cache (bool): Whether to refresh the cache.

        Returns:
            tuple: The indices of the nearest neighbors as well as the cache hit rate.
        """

        if refresh_cache or self.position_cache is None or means.shape[
                0] != self.position_cache.shape[0]:
            self.position_cache = None
            self.recompute_distance_cache = None
            self.index_cache = None
            distances, indices = self.index.search(means, k=self.k + 1)
            distances, sort_indices = torch.sort(distances, dim=-1)
            indices = torch.gather(indices, -1, sort_indices)
            self.position_cache = means
            self.index_cache = indices[:, :-1]
            # we need to recompute if we moved more than the delta of the distance of the k-th
            # nearest neighbor and the k+1-th nearest neighbor
            self.recompute_distance_cache = distances[:, -1] - distances[:, -2]
            hit_rate = 0.0

        else:
            # Compute the distance between the mean and the cached positions.
            moved_distance = torch.norm(means - self.position_cache, dim=-1)  # (n_gaussians, )
            query_indices = torch.where(moved_distance >= self.recompute_distance_cache)[0]
            cache_indices = torch.where(moved_distance < self.recompute_distance_cache)[0]
            query_dists, query_neighbors = self.index.search(means[query_indices], k=self.k + 1)
            query_dists, sort_indices = torch.sort(query_dists, dim=-1)
            query_neighbors = torch.gather(query_neighbors, -1, sort_indices)
            self.position_cache[query_indices] = means[query_indices]
            self.index_cache[query_indices] = query_neighbors[:, :-1]
            self.recompute_distance_cache[query_indices] = query_dists[:, -1] - query_dists[:, -2]
            hit_rate = cache_indices.shape[0] / means.shape[0]

        return self.index_cache, hit_rate
