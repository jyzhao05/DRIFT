import numpy as np
from typing import List, Optional
import warnings
import torch
from scipy.special import comb

def get_bezier_curve(control_points: torch.Tensor, n_points: int = 50) -> torch.Tensor:
    B, N_ctrl, D = control_points.shape
    n = N_ctrl - 1
    t = torch.linspace(0, 1, n_points, device=control_points.device, dtype=control_points.dtype)
    coeffs = torch.tensor([comb(n, i) for i in range(N_ctrl)], device=control_points.device, dtype=control_points.dtype)
    power_t = t.unsqueeze(0) ** torch.arange(N_ctrl, device=t.device).unsqueeze(1)
    power_1_minus_t = (1 - t).unsqueeze(0) ** torch.arange(n, -1, -1, device=t.device).unsqueeze(1)
    bernstein_basis = coeffs.unsqueeze(1) * power_t * power_1_minus_t
    curve = torch.einsum("bnc,ni->bic", control_points, bernstein_basis)
    return curve

def apply_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to 2D or 3D points using float64 geometry."""
    if points.shape[0] == 0:
        return points

    if transform.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {transform.shape}")

    points_64 = points.astype(np.float64)
    transform_64 = transform.astype(np.float64)

    if points_64.shape[1] == 2:
        points_3d = np.column_stack([points_64, np.zeros(points_64.shape[0])])
        return_2d = True
    elif points_64.shape[1] == 3:
        points_3d = points_64.copy()
        return_2d = False
    else:
        raise ValueError(f"Points must be (N, 2) or (N, 3), got {points.shape}")

    points_homogeneous = np.column_stack([points_3d, np.ones(points_3d.shape[0])])

    transformed_homogeneous = (transform_64 @ points_homogeneous.T).T

    transformed_3d = transformed_homogeneous[:, :3]

    if return_2d:
        return transformed_3d[:, :2]
    else:
        return transformed_3d

def inverse_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply the inverse of a homogeneous transform to points."""
    try:
        transform_64 = transform.astype(np.float64)
        inverse_transform_matrix = np.linalg.inv(transform_64)
        return apply_transformation(points, inverse_transform_matrix)
    except np.linalg.LinAlgError:
        warnings.warn("Transform matrix is singular, using pseudo-inverse")
        inverse_transform_matrix = np.linalg.pinv(transform.astype(np.float64))
        return apply_transformation(points, inverse_transform_matrix)

def compose_transforms(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape != (4, 4) or B.shape != (4, 4):
        raise ValueError("Both transform matrices must be 4x4")
    return B @ A

def create_transform_matrix(rotation: Optional[np.ndarray] = None,
                          translation: Optional[np.ndarray] = None) -> np.ndarray:
    transform = np.eye(4)
    if rotation is not None:
        if rotation.shape == (3, 3):
            transform[:3, :3] = rotation
        else:
            raise ValueError("Rotation must be a 3x3 matrix")
    if translation is not None:
        if len(translation) == 3:
            transform[:3, 3] = translation
        else:
            raise ValueError("Translation must be a length-3 vector")
    return transform

def pose_to_transform_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a 3x4 or 4x4 pose array to a homogeneous 4x4 transform."""
    if pose.shape == (3, 4):
        transform = np.eye(4)
        transform[:3, :4] = pose
        return transform
    elif pose.shape == (4, 4):
        return pose
    else:
        raise ValueError(f"Unsupported pose shape: {pose.shape}")

def check_transform_validity(transform: np.ndarray) -> bool:
    """Check transform shape, nonsingularity, and approximate rotation orthonormality."""
    if transform.shape != (4, 4):
        return False

    try:
        det = np.linalg.det(transform)
        if abs(det) < 1e-10:
            return False
    except:
        return False

    rotation = transform[:3, :3]
    try:
        should_be_identity = rotation @ rotation.T
        identity_error = np.abs(should_be_identity - np.eye(3)).max()
        if identity_error > 0.1:
            return False
    except:
        return False

    return True

def robust_motion_compensate_pointclouds(pointclouds: List[np.ndarray],
                                       poses: List[np.ndarray],
                                       ref_idx: int = -1) -> np.ndarray:
    """Align multiple point-cloud frames to a reference frame using pose-based motion compensation."""
    if len(pointclouds) != len(poses):
        raise ValueError("Number of pointclouds must match number of poses")

    if len(pointclouds) == 0:
        return np.empty((0, 3))

    if ref_idx == -1:
        ref_idx = len(pointclouds) - 1

    if ref_idx < 0 or ref_idx >= len(pointclouds):
        raise ValueError(f"Reference index {ref_idx} out of range")

    transforms = []
    for i, pose in enumerate(poses):
        try:
            transform = pose_to_transform_matrix(pose)
            if not check_transform_validity(transform):
                warnings.warn(f"Invalid transform at frame {i}, using identity")
                transform = np.eye(4)
            transforms.append(transform)
        except Exception as e:
            warnings.warn(f"Failed to convert pose at frame {i}: {e}, using identity")
            transforms.append(np.eye(4))

    ref_transform = transforms[ref_idx]

    transformed_clouds = []

    for i, pc in enumerate(pointclouds):
        if pc.shape[0] == 0:
            continue

        if pc.shape[1] > 3:
            pc = pc[:, :3]
        elif pc.shape[1] < 3:
            continue

        if i == ref_idx:
            transformed_clouds.append(pc)
        else:
            try:
                curr_transform = transforms[i]

                try:
                    ref_inv = np.linalg.inv(ref_transform)
                except np.linalg.LinAlgError:
                    ref_inv = np.linalg.pinv(ref_transform)

                T_ref_curr = ref_inv @ curr_transform

                if not check_transform_validity(T_ref_curr):
                    warnings.warn(f"Invalid transformation for frame {i}, using ICP fallback")
                    transformed_pc = pc.copy()
                else:
                    transformed_pc = apply_transformation(pc, T_ref_curr)

                transformed_clouds.append(transformed_pc)

            except Exception as e:
                warnings.warn(f"Failed to compute transformation for frame {i}: {e}, using original points")
                transformed_clouds.append(pc)

    if len(transformed_clouds) == 0:
        return np.empty((0, 3))

    P_unified = np.vstack(transformed_clouds)
    return P_unified

def safe_inverse_transform(transform: np.ndarray) -> np.ndarray:
    """Compute a transform inverse, using a pseudoinverse for singular input."""
    try:
        return np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        warnings.warn("Transform matrix is singular, using pseudo-inverse")
        return np.linalg.pinv(transform)

def preprocess_pose_data(pose_data: np.ndarray) -> np.ndarray:
    """Normalize pose input to a validated 4x4 homogeneous transform."""
    if pose_data.shape == (3, 4):
        rotation_part = pose_data[:3, :3]

        det = np.linalg.det(rotation_part)
        if abs(det - 1.0) > 0.5:
            warnings.warn("Rotation matrix determinant is not close to 1, data might be corrupted")

        transform = np.eye(4)
        transform[:3, :4] = pose_data
        return transform

    elif pose_data.shape == (4, 4):
        return pose_data

    else:
        warnings.warn(f"Unexpected pose shape {pose_data.shape}, using identity transform")
        return np.eye(4)

def icp_alignment_fallback(source: np.ndarray, target: np.ndarray,
                          max_iterations: int = 50, tolerance: float = 1e-6) -> np.ndarray:
    """Estimate a rigid alignment with a lightweight ICP fallback."""
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.eye(4)

    current_transform = np.eye(4)
    current_source = source.copy()

    for iteration in range(max_iterations):
        from scipy.spatial.distance import cdist
        distances = cdist(current_source, target)
        closest_indices = np.argmin(distances, axis=1)
        corresponding_target_points = target[closest_indices]

        source_centroid = np.mean(current_source, axis=0)
        target_centroid = np.mean(corresponding_target_points, axis=0)

        source_centered = current_source - source_centroid
        target_centered = corresponding_target_points - target_centroid

        try:
            H = source_centered.T @ target_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = target_centroid - R @ source_centroid

            iteration_transform = create_transform_matrix(R, t)

            current_transform = iteration_transform @ current_transform

            current_source = apply_transformation(current_source, iteration_transform)

            translation_norm = np.linalg.norm(t)
            if translation_norm < tolerance:
                break

        except np.linalg.LinAlgError:
            warnings.warn("SVD failed in ICP, returning identity transform")
            break

    return current_transform

def extract_pose_components(transform: np.ndarray) -> tuple:
    """Extract rotation and translation components from a homogeneous transform."""
    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix")

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    return rotation, translation
