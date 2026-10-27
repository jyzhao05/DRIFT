import numpy as np
import warnings
from typing import List, Dict, Any, Optional
import torch

class IntegratedDataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        lidar_cfg = getattr(cfg, 'lidar_cfg', cfg)
        self.use_motion_compensation = getattr(lidar_cfg, 'use_motion_compensation', True)
        self.max_frames = getattr(lidar_cfg, 'max_frames', 3)

    @staticmethod
    def pose_to_transform_matrix(pose: np.ndarray) -> np.ndarray:
        """Normalize supported pose layouts to a float64 homogeneous transform."""
        pose_64 = np.asarray(pose, dtype=np.float64)
        if pose_64.shape == (3, 4):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :4] = pose_64
            return transform
        elif pose_64.shape == (4, 4):
            return pose_64
        elif pose_64.shape == (4, 3):
            return IntegratedDataProcessor.pose_to_transform_matrix(pose_64.T)
        else:
            warnings.warn(f"Unexpected pose shape {pose_64.shape}, using identity")
            return np.eye(4, dtype=np.float64)

    @staticmethod
    def check_transform_validity(transform: np.ndarray) -> bool:
        """Validate shape, nonsingularity, and approximate rotation orthogonality."""
        if transform.shape != (4, 4): return False
        try:
            det = np.linalg.det(transform)
            if abs(det) < 1e-10: return False
            rotation = transform[:3, :3]
            should_be_identity = rotation @ rotation.T
            identity_error = np.abs(should_be_identity - np.eye(3)).max()
            if identity_error > 0.1: return False
        except:
            return False
        return True

    def process_lidar_frames(self, lidar_data: List[np.ndarray]) -> List[np.ndarray]:
        """Convert raw LiDAR frames to finite XYZ arrays and cap the frame history."""
        if not isinstance(lidar_data, list):
            try: lidar_data = [lidar_data]
            except: return []
        processed_frames = []
        for frame in lidar_data:
            try:
                frame_array = np.asarray(frame)
                if frame_array.size == 0: continue
                if frame_array.ndim == 1:
                    if frame_array.size % 3 == 0: frame_array = frame_array.reshape(-1, 3)
                    else: continue
                elif frame_array.ndim == 2:
                    if frame_array.shape[1] >= 3: frame_array = frame_array[:, :3]
                    else: continue
                else: continue
                valid_mask = np.isfinite(frame_array).all(axis=1)
                if valid_mask.sum() > 0:
                    processed_frames.append(frame_array[valid_mask])
            except Exception as e:
                warnings.warn(f"Failed to process lidar frame: {e}")
        if len(processed_frames) > self.max_frames:
            processed_frames = processed_frames[-self.max_frames:]
        return processed_frames

    def create_dummy_poses_for_frames(self, current_pose: np.ndarray, num_frames: int) -> List[np.ndarray]:
        """Create approximate historical poses when a sequence lacks pose metadata."""
        current_transform = self.pose_to_transform_matrix(current_pose)
        if num_frames <= 1: return [current_transform]
        poses = []
        current_position = current_transform[:3, 3]
        current_rotation = current_transform[:3, :3]
        forward_direction = current_rotation[:, 0]
        estimated_speed = 0.5
        time_interval = 0.1
        for i in range(num_frames):
            if i == num_frames - 1:
                poses.append(current_transform)
            else:
                time_back = (num_frames - 1 - i) * time_interval
                position_offset = forward_direction * estimated_speed * time_back
                historical_transform = current_transform.copy()
                historical_transform[:3, 3] = current_position - position_offset
                poses.append(historical_transform)
        return poses

    def align_and_merge_pointclouds(self, pointcloud_frames: List[np.ndarray], current_pose: np.ndarray) -> np.ndarray:
        """Return the latest point-cloud frame in local coordinates.
        """
        if not pointcloud_frames:
            return np.empty((0, 3))
        return pointcloud_frames[-1]

    def process_velocity_data(self, vel_data: List, target_length: int) -> np.ndarray:
        """Flatten velocity records and pad or truncate them to a fixed length."""
        if not isinstance(vel_data, list) or len(vel_data) == 0: return np.zeros(target_length)
        try:
            vel_flat = []
            for vel_item in vel_data:
                if isinstance(vel_item, list): vel_flat.extend(vel_item)
                else: vel_flat.append(float(vel_item))
            vel_array = np.array(vel_flat)
            if len(vel_array) >= target_length: return vel_array[:target_length]
            else:
                result = np.zeros(target_length)
                result[:len(vel_array)] = vel_array
                return result
        except Exception as e:
            warnings.warn(f"Failed to process velocity data: {e}")
            return np.zeros(target_length)

    def process_trajectory_data(self, traj_data: List, target_length: int) -> np.ndarray:
        """Convert a raw trajectory to a finite two-dimensional array with the requested length."""
        try:
            if not isinstance(traj_data, list) or len(traj_data) == 0:
                return np.zeros((target_length, 2), dtype=np.float32)

            traj_points = []
            for point in traj_data:
                point_array = np.asarray(point)
                if point_array.size >= 2 and np.all(np.isfinite(point_array[:2])):
                    traj_points.append(point_array[:2])

            if len(traj_points) == 0:
                return np.zeros((target_length, 2), dtype=np.float32)

            traj_array = np.stack(traj_points).astype(np.float32)

            current_len = traj_array.shape[0]
            if current_len == target_length:
                return traj_array
            elif current_len > target_length:
                indices = np.linspace(0, current_len - 1, target_length, dtype=int)
                return traj_array[indices]
            else: # current_len < target_length
                num_padding = target_length - current_len
                last_point = traj_array[-1, :].reshape(1, 2)
                padding = np.repeat(last_point, num_padding, axis=0)
                return np.vstack([traj_array, padding])

        except Exception as e:
            warnings.warn(f"Trajectory processing failed: {e}; returning a zero trajectory")
            return np.zeros((target_length, 2), dtype=np.float32)

    def process_target_data(self, target_data: Any) -> tuple:
        """Normalize target position to three coordinates and return its scalar value."""
        try:
            if isinstance(target_data, (list, tuple)) and len(target_data) >= 2:
                position = np.asarray(target_data[0])
                scalar = float(target_data[1])
                if position.size >= 3: return position[:3], scalar
                elif position.size == 2: return np.array([position[0], position[1], 0.0]), scalar
                else: return np.array([0.0, 0.0, 0.0]), scalar
            else:
                pos = np.asarray(target_data)
                if pos.size >= 3: return pos[:3], 0.0
                elif pos.size == 2: return np.array([pos[0], pos[1], 0.0]), 0.0
                else: return np.array([0.0, 0.0, 0.0]), 0.0
        except Exception as e:
            warnings.warn(f"Target processing failed: {e}")
            return np.array([0.0, 0.0, 0.0]), 0.0
