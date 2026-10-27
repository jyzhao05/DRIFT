import numpy as np
from typing import Tuple, Optional
import warnings

import faiss
from scipy.spatial import cKDTree

try:
    import torch
    import torch_cluster
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud by averaging points within each voxel."""
    if points.shape[0] == 0:
        return points
    if points.shape[1] < 3:
        raise ValueError("Points must have at least 3 dimensions (x, y, z)")

    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int64)

    hashed_indices = (voxel_indices[:, 0] * 73856093) ^ \
                     (voxel_indices[:, 1] * 19349663) ^ \
                     (voxel_indices[:, 2] * 83492791)

    sorted_order = np.argsort(hashed_indices)
    sorted_points = points[sorted_order]
    sorted_hashed_indices = hashed_indices[sorted_order]

    unique_indices, counts = np.unique(sorted_hashed_indices, return_counts=True)

    summed_points = np.add.reduceat(sorted_points, np.cumsum(np.concatenate(([0], counts[:-1]))))

    downsampled_points = summed_points / counts[:, np.newaxis]

    return downsampled_points

def build_knn_graph(pos: np.ndarray, k: int) -> np.ndarray:
    """Build a k-nearest-neighbor edge index with FAISS on CPU."""
    if pos.shape[0] <= k:
        k = pos.shape[0] - 1
        if k <= 0:
            return np.empty((2, 0), dtype=np.int64)

    n, dim = pos.shape
    pos_float32 = pos.astype('float32')

    index = faiss.IndexFlatL2(dim)

    index.add(pos_float32)
    distances, indices = index.search(pos_float32, k + 1)

    source_nodes = np.arange(n).repeat(k)
    target_nodes = indices[:, 1:].flatten()

    return np.stack((source_nodes, target_nodes))

def compute_local_density(voxel_centers: np.ndarray, original_points: np.ndarray,
                         voxel_size: float) -> np.ndarray:
    """Estimate local point density around each voxel center."""
    if voxel_centers.shape[0] == 0:
        return np.array([])

    tree = cKDTree(original_points[:, :3])

    radius = voxel_size * np.sqrt(3) / 2
    points_in_voxel = tree.query_ball_point(voxel_centers, r=radius, return_length=True)

    voxel_volume = voxel_size ** 3
    densities = points_in_voxel / (voxel_volume + 1e-8)

    return densities

def compute_node_features(pos: np.ndarray, original_points: Optional[np.ndarray] = None,
                         voxel_size: Optional[float] = None) -> np.ndarray:
    """Compute node features from voxel positions and local density."""
    if pos.shape[0] == 0:
        return np.empty((0, 4))

    features = [pos]

    if original_points is not None and voxel_size is not None:
        density = compute_local_density(pos, original_points, voxel_size)
        features.append(density.reshape(-1, 1))
    else:
        features.append(np.ones((pos.shape[0], 1)))

    return np.concatenate(features, axis=1)

def compute_edge_features(pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Compute relative geometric features for graph edges."""
    if edge_index.shape[1] == 0:
        return np.empty((0, 6))

    source_nodes, target_nodes = edge_index[0], edge_index[1]
    source_pos, target_pos = pos[source_nodes], pos[target_nodes]
    delta_pos = target_pos - source_pos

    distances = np.linalg.norm(delta_pos, axis=1, keepdims=True)
    distances_safe = np.where(distances > 1e-8, distances, 1e-8)

    azimuth = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])
    elevation = np.arcsin(np.clip(delta_pos[:, 2] / distances_safe.flatten(), -1, 1))

    return np.concatenate([
        delta_pos, distances, azimuth.reshape(-1, 1), elevation.reshape(-1, 1)
    ], axis=1)

def build_graph_gpu_batched(pos_list: list, k: int) -> tuple:
    """Build k-NN graphs for a batch of point clouds with torch-cluster."""
    if not HAS_TORCH_CLUSTER:
        raise ImportError("torch_cluster is required for GPU-accelerated batched k-NN. Please install it.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Graph Builder] Using device: {device}")

    pos_tensors = [torch.from_numpy(pos).to(device, dtype=torch.float) for pos in pos_list]

    batch_index_list = [torch.full((pos.shape[0],), i, dtype=torch.long)
                        for i, pos in enumerate(pos_tensors)]
    batch_index_tensor = torch.cat(batch_index_list).to(device)

    batched_pos = torch.cat(pos_tensors, dim=0)

    batched_edge_index = torch_cluster.knn_graph(batched_pos, k=k, batch=batch_index_tensor, loop=False)

    return batched_edge_index.cpu().numpy(), batch_index_tensor.cpu().numpy()
