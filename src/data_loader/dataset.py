import copy
import os
import pickle
from os.path import join
import random
from random import shuffle
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import traceback

from src.utils.transform import apply_transformation
from src.utils.voxel_graph import (
    voxel_downsample,
    build_knn_graph,
    compute_edge_features,
    compute_node_features
)
from src.utils.config import DataDict
from src.utils.integrated_processor import IntegratedDataProcessor
class TrainData(Dataset):
    """Dataset wrapper for trajectory, pose, velocity, target, and local graph samples."""
    def __init__(self, cfg, split: str = 'train'):
        self.full_config = cfg
        self.split = split
        dataset_cfg = cfg.dataset
        graph_cfg = cfg.dataset.graph_cfg

        self.data_root = join(dataset_cfg.root, "data_folder")
        self.traj_len = dataset_cfg.traj_len
        self.vel_num = dataset_cfg.vel_num
        self.voxel_size = graph_cfg.voxel_size
        self.k_nn = graph_cfg.k_nn
        self.max_nodes = graph_cfg.max_nodes

        with open(join(dataset_cfg.root, "data.pkl"), "rb") as input_file:
            data = pickle.load(input_file)

        self.original_files = data[DataDict.file_names]
        self.all_files = []
        self._process_files(dataset_cfg)
        self.processor = IntegratedDataProcessor(dataset_cfg)
        print(f"Dataset initialized: {len(self.all_files)} samples in {split} split")

    def _process_files(self, dataset_cfg):
        all_files = []
        for fn, tnum in self.original_files:
            for i in range(tnum):
                all_files.append((fn, i))

        random.seed(0)
        random.shuffle(all_files)

        data_percentage = getattr(dataset_cfg, 'training_data_percentage', 0.8)
        total_samples = len(all_files)
        train_split = int(total_samples * data_percentage)
        val_split = int(total_samples * (data_percentage + 0.1))

        if self.split == 'train':
            self.all_files = all_files[:train_split]
        elif self.split == 'val':
            self.all_files = all_files[train_split:val_split]
        else: # test or all
            self.all_files = all_files[val_split:]

    def load_raw_data(self, idx: int) -> Optional[dict]:
        """Load one serialized sample and validate the required fields."""
        try:
            fn, sample_idx = self.all_files[idx]
            full_path = join(self.data_root, fn)
            with open(full_path, "rb") as input_file:
                pkl_data = pickle.load(input_file)

            required_keys = [DataDict.pose, DataDict.trajectories, DataDict.targets]
            if not all(k in pkl_data for k in required_keys):
                return None

            if len(pkl_data[DataDict.trajectories]) <= sample_idx or len(pkl_data[DataDict.targets]) <= sample_idx:
                 return None

            return {
                'pose': pkl_data.get(DataDict.pose),
                'lidar': pkl_data.get(DataDict.lidar, []),
                'velocity': pkl_data.get(DataDict.vel, []),
                'trajectories': pkl_data.get(DataDict.trajectories),
                'targets': pkl_data.get(DataDict.targets),
                'sample_idx': sample_idx
            }
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Build one training sample, including coordinate normalization and a local point-cloud graph."""
        raw_data = self.load_raw_data(idx)
        if raw_data is None:
            return None

        try:
            sample_idx = raw_data['sample_idx']
            original_pose_4x4 = self.processor.pose_to_transform_matrix(raw_data['pose']).astype(np.float64)
            if not self.processor.check_transform_validity(original_pose_4x4):
                return None

            world_from_local_transform = np.linalg.inv(original_pose_4x4)
            traj_gt_world = self.processor.process_trajectory_data(raw_data['trajectories'][sample_idx], self.traj_len)

            if traj_gt_world.shape[0] == 0: return None

            traj_gt_local = apply_transformation(traj_gt_world, original_pose_4x4)
            if traj_gt_local.shape[0] == 0: return None

            lidar_frames = self.processor.process_lidar_frames(raw_data['lidar'])
            if len(lidar_frames) > 0:
                merged_pointcloud = self.processor.align_and_merge_pointclouds(lidar_frames, raw_data['pose'])

                if merged_pointcloud.shape[0] < self.k_nn + 1:
                    warnings.warn(f"Sample {idx} has too few point-cloud nodes ({merged_pointcloud.shape[0]}) for k-NN; skipping.")
                    return None

                pos, edge_index, node_features = self._preprocess_pointcloud(merged_pointcloud)
                local_graph = self.build_local_graph(pos, edge_index, node_features)
            else:
                local_graph = self._get_empty_graph()

            vel = self.processor.process_velocity_data(raw_data['velocity'], self.vel_num)
            target_pos_world, _ = self.processor.process_target_data(raw_data['targets'][sample_idx])
            target_relative = apply_transformation(target_pos_world.reshape(1, -1), original_pose_4x4).flatten()

            shift = torch.from_numpy(traj_gt_local[0, :]).clone()
            centered_traj = torch.from_numpy(traj_gt_local) - shift.unsqueeze(0)
            scale = torch.norm(centered_traj, p=2, dim=-1).max() + 1e-8
            normalized_traj = centered_traj / scale

            return {
                'local_graph': local_graph,
                'vel': torch.from_numpy(vel).float(),
                'pose': torch.from_numpy(world_from_local_transform).float(),
                'traj_gt_world': torch.from_numpy(traj_gt_world).float(),
                'traj_gt_local': torch.from_numpy(traj_gt_local).float(),
                'traj_gt': normalized_traj.float(),
                'traj_scale': scale.expand(2).float(),
                'traj_shift': shift.float(),
                'target': normalized_traj[-1, :].float(),
                'meta': {'file_idx': idx}
            }
        except Exception:
            return None

    def _preprocess_pointcloud(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if points.shape[0] == 0: return np.empty((0, 3)), np.empty((2, 0), dtype=np.int64), np.empty((0, 4))
        downsampled_points = voxel_downsample(points, self.voxel_size)
        if downsampled_points.shape[0] > self.max_nodes:
            indices = np.random.choice(downsampled_points.shape[0], self.max_nodes, replace=False)
            downsampled_points = downsampled_points[indices]
        pos = downsampled_points[:, :3]
        edge_index = build_knn_graph(pos, k=self.k_nn)
        node_features = compute_node_features(pos=pos, original_points=points, voxel_size=self.voxel_size)
        return pos, edge_index, node_features

    def build_local_graph(self, pos: np.ndarray, edge_index: np.ndarray, node_features: np.ndarray) -> Data:
        if pos.shape[0] == 0: return self._get_empty_graph()
        edge_attr = compute_edge_features(pos, edge_index) if edge_index.shape[1] > 0 else np.empty((0, 6))
        return Data(x=torch.from_numpy(node_features).float(), pos=torch.from_numpy(pos).float(), edge_index=torch.from_numpy(edge_index).long(), edge_attr=torch.from_numpy(edge_attr).float())

    def _get_empty_graph(self) -> Data:
        return Data(x=torch.zeros((1, 4)), pos=torch.zeros((1, 3)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 6)))

    def __len__(self) -> int:
        return len(self.all_files)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from a small sample of the dataset."""
        if len(self.all_files) == 0:
            return {}

        sample_size = min(50, len(self.all_files))
        sample_indices = np.random.choice(len(self.all_files), sample_size, replace=False)

        stats = {
            'num_samples': len(self.all_files),
            'node_counts': [],
            'edge_counts': [],
            'traj_lengths': [],
            'lidar_frame_counts': [],
            'merged_point_counts': []
        }

        for idx in sample_indices:
            try:
                sample = self[idx]
                stats['node_counts'].append(sample['meta']['num_nodes'])
                stats['edge_counts'].append(sample['meta']['num_edges'])
                stats['traj_lengths'].append(sample['meta']['original_traj_len'])
                stats['lidar_frame_counts'].append(sample['meta']['num_lidar_frames'])
                stats['merged_point_counts'].append(sample['meta']['merged_points'])
            except:
                continue

        for key in ['node_counts', 'edge_counts', 'traj_lengths', 'lidar_frame_counts', 'merged_point_counts']:
            if len(stats[key]) > 0:
                values = np.array(stats[key])
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))
                stats[f'{key}_min'] = int(np.min(values))
                stats[f'{key}_max'] = int(np.max(values))

        return stats

    def validate_sample(self, idx: int) -> Dict[str, bool]:
        """Validate the required fields and derived graph inputs for one sample."""
        validation_results = {
            'data_loadable': False,
            'has_lidar': False,
            'has_trajectory': False,
            'has_target': False,
            'pose_valid': False,
            'graph_valid': False
        }

        try:
            raw_data = self.load_raw_data(idx)
            validation_results['data_loadable'] = True

            lidar_frames = self.processor.process_lidar_frames(raw_data['lidar'])
            validation_results['has_lidar'] = len(lidar_frames) > 0

            sample_idx = raw_data['sample_idx']
            trajectories = raw_data['trajectories']
            if len(trajectories) > sample_idx and len(trajectories[sample_idx]) > 0:
                validation_results['has_trajectory'] = True

            targets = raw_data['targets']
            if len(targets) > sample_idx:
                validation_results['has_target'] = True

            pose_4x4 = self.processor.pose_to_transform_matrix(raw_data['pose'])
            validation_results['pose_valid'] = self.processor.check_transform_validity(pose_4x4)

            if validation_results['has_lidar']:
                merged_pc = self.processor.align_and_merge_pointclouds(lidar_frames, raw_data['pose'])
                pos, feats = self._preprocess_pointcloud(merged_pc)
                if pos.shape[0] > 0:
                    validation_results['graph_valid'] = True

        except Exception as e:
            warnings.warn(f"Validation failed for sample {idx}: {e}")

        return validation_results

import glob

class PreprocessedTrainData(Dataset):
    """Dataset wrapper for preprocessed graph tensors and trajectory targets."""
    def __init__(self, cfg, split: str = 'train'):

        processed_dir_name = getattr(cfg, 'processed_dir_name', 'processed_data')
        self.data_root = os.path.join(cfg.root, processed_dir_name)

        self.split = split

        if not os.path.exists(self.data_root):
            raise FileNotFoundError(
                f"Preprocessed data directory does not exist: {self.data_root}\n"
                "Run preprocess_data.py before loading this dataset."
            )

        import glob
        self.all_files = sorted(glob.glob(os.path.join(self.data_root, "*.pt")))

        if len(self.all_files) == 0:
            raise FileNotFoundError(
                f"No preprocessed .pt files found in {self.data_root}."
            )

        random.seed(0)
        shuffle(self.all_files)

        total_samples = len(self.all_files)
        data_percentage = getattr(cfg, 'training_data_percentage', 0.8)
        train_split = int(total_samples * data_percentage)
        val_split = int(total_samples * (data_percentage + 0.1))

        if self.split == 'train':
            self.files = self.all_files[:train_split]
        elif self.split == 'val':
            self.files = self.all_files[train_split:val_split]
        else: # test
            self.files = self.all_files[val_split:]

        print(f"PreprocessedDataset initialized: {len(self.files)} samples in {split} split")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])
