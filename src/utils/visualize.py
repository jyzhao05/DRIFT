"""
Visualization utilities for DRIFT - Cross-Modal Attention Graph Neural Network
Handles trajectory visualization, attention heatmaps, and debugging plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple
import warnings
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available, some visualization features may be limited")

try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False

def write_png(local_map: np.ndarray, trajs: List[np.ndarray], save_path: str,
              resolution: float = 0.1, origin: Optional[Tuple[float, float]] = None,
              colors: Optional[List[str]] = None, trajectory_width: int = 3):
    """Render a map with one or more trajectories and save it as a PNG image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(local_map.shape) == 3:
        if local_map.shape[2] == 3:
            map_gray = np.mean(local_map, axis=2)
        else:
            map_gray = local_map[:, :, 0]
    else:
        map_gray = local_map

    if map_gray.max() > 1.0:
        map_normalized = np.clip(map_gray, 0, 255).astype(np.uint8)
    else:
        map_normalized = (map_gray * 255).astype(np.uint8)

    map_rgb = np.stack([map_normalized] * 3, axis=2)

    if origin is None:
        origin = (0.0, 0.0)

    if colors is None:
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    color_map = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'purple': (255, 0, 255),
        'orange': (255, 165, 0),
        'cyan': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }

    for i, traj in enumerate(trajs):
        if traj.shape[0] < 2:
            continue

        color_name = colors[i % len(colors)]
        color = color_map.get(color_name, (255, 0, 0))

        pixel_coords = world_to_pixel(traj, resolution, origin, map_rgb.shape[:2])

        if HAS_CV2:
            for j in range(len(pixel_coords) - 1):
                pt1 = tuple(pixel_coords[j].astype(int))
                pt2 = tuple(pixel_coords[j + 1].astype(int))
                cv2.line(map_rgb, pt1, pt2, color, trajectory_width)

            start_point = tuple(pixel_coords[0].astype(int))
            end_point = tuple(pixel_coords[-1].astype(int))
            cv2.circle(map_rgb, start_point, trajectory_width + 2, (0, 255, 0), -1)
            cv2.circle(map_rgb, end_point, trajectory_width + 2, (255, 0, 0), -1)
        else:
            for j in range(len(pixel_coords) - 1):
                y1, x1 = pixel_coords[j].astype(int)
                y2, x2 = pixel_coords[j + 1].astype(int)
                draw_line_on_image(map_rgb, (x1, y1), (x2, y2), color, trajectory_width)

    try:
        if HAS_CV2:
            cv2.imwrite(save_path, cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR))
        else:
            plt.figure(figsize=(10, 10))
            plt.imshow(map_rgb)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f"Map with trajectories saved to: {save_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save visualization to {save_path}: {str(e)}")

def visualize_attention(pointcloud: np.ndarray, node_pos: np.ndarray,
                       attention_weights: np.ndarray, waypoint: np.ndarray,
                       save_path: str, title: str = "Attention Visualization"):
    """Project node attention weights onto a point cloud and save a diagnostic plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if pointcloud.shape[0] > 0:
        ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                  c='lightgray', alpha=0.3, s=1, label='Original Points')

    if node_pos.shape[0] > 0 and attention_weights.shape[0] > 0:
        weights_normalized = (attention_weights - attention_weights.min()) / \
                           (attention_weights.max() - attention_weights.min() + 1e-8)

        scatter = ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
                           c=weights_normalized, cmap='hot', s=50, alpha=0.8,
                           label='Graph Nodes')

        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Attention Weight')

    if len(waypoint) >= 2:
        waypoint_3d = np.append(waypoint[:2], 0 if len(waypoint) == 2 else waypoint[2])
        ax.scatter(waypoint_3d[0], waypoint_3d[1], waypoint_3d[2],
                  c='red', s=200, marker='*', label='Current Waypoint')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

    ax.set_box_aspect([1,1,0.5])

    try:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Attention visualization saved to: {save_path}")
    except Exception as e:
        plt.close()
        raise RuntimeError(f"Failed to save attention visualization: {str(e)}")

def plot_trajectory_on_pointcloud(pc: np.ndarray, traj: np.ndarray,
                                 save_path: Optional[str] = None,
                                 title: str = "Trajectory on Point Cloud"):
    """Plot a trajectory over a point cloud for coordinate-frame inspection."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if pc.shape[0] > 0:
        if pc.shape[0] > 10000:
            indices = np.random.choice(pc.shape[0], 10000, replace=False)
            pc_sampled = pc[indices]
        else:
            pc_sampled = pc

        ax.scatter(pc_sampled[:, 0], pc_sampled[:, 1], pc_sampled[:, 2],
                  c='lightblue', alpha=0.5, s=1, label='Point Cloud')

    if traj.shape[0] > 0:
        if traj.shape[1] == 2:
            traj_3d = np.column_stack([traj, np.zeros(traj.shape[0])])
        else:
            traj_3d = traj

        ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
               'r-', linewidth=3, label='Trajectory')

        ax.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2],
                  c='red', s=100, marker='s', label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Trajectory visualization saved to: {save_path}")
    else:
        plt.show()

def world_to_pixel(points: np.ndarray, resolution: float,
                  origin: Tuple[float, float], map_shape: Tuple[int, int]) -> np.ndarray:
    """Convert world coordinates to image pixel coordinates."""
    pixel_x = (points[:, 0] - origin[0]) / resolution
    pixel_y = map_shape[0] - (points[:, 1] - origin[1]) / resolution

    pixel_x = np.clip(pixel_x, 0, map_shape[1] - 1)
    pixel_y = np.clip(pixel_y, 0, map_shape[0] - 1)

    return np.column_stack([pixel_y, pixel_x])

def draw_line_on_image(image: np.ndarray, pt1: Tuple[int, int],
                      pt2: Tuple[int, int], color: Tuple[int, int, int],
                      thickness: int = 1):
    """Draw a line segment with a small raster-only fallback implementation."""
    x1, y1 = pt1
    x2, y2 = pt2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    x, y = x1, y1

    while True:
        for i in range(-thickness//2, thickness//2 + 1):
            for j in range(-thickness//2, thickness//2 + 1):
                px, py = x + i, y + j
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    image[py, px] = color

        if x == x2 and y == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

def plot_training_curves(losses: List[float], metrics: dict, save_path: str):
    """Plot training losses and metric histories."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)

    axes[0, 0].plot(losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    metric_names = list(metrics.keys())
    for i, (metric_name, values) in enumerate(metrics.items()):
        if i < 3:
            row = (i + 1) // 2
            col = (i + 1) % 2
            axes[row, col].plot(values, 'g-', linewidth=2)
            axes[row, col].set_title(f'{metric_name.title()}')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric_name.title())
            axes[row, col].grid(True)

    for i in range(len(metric_names) + 1, 4):
        row = i // 2
        col = i % 2
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

def plot_attention_matrix(attention_weights: np.ndarray, save_path: str,
                         title: str = "Attention Matrix"):
    """Render an attention matrix as a heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title(title)
    plt.xlabel('Environment Nodes')
    plt.ylabel('Trajectory Points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention matrix saved to: {save_path}")

def plot_trajectory_comparison(gt_traj: np.ndarray, pred_trajs: List[np.ndarray],
                              labels: List[str], save_path: str):
    """Compare a ground-truth trajectory with one or more predictions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 8))

    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'k-', linewidth=3,
            label='Ground Truth', alpha=0.8)
    plt.scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=100,
               marker='o', label='Start', zorder=5)
    plt.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=100,
               marker='s', label='Goal', zorder=5)

    colors = ['blue', 'orange', 'purple', 'brown', 'pink']
    all_x_coords = [gt_traj[:, 0]]
    all_y_coords = [gt_traj[:, 1]]
    for i, (traj, label) in enumerate(zip(pred_trajs, labels)):
        color = colors[i % len(colors)]
        plt.plot(traj[:, 0], traj[:, 1], '--', color=color,
                linewidth=2, label=label, alpha=0.7)
        all_x_coords.append(traj[:, 0])
        all_y_coords.append(traj[:, 1])

    all_x = np.concatenate(all_x_coords)
    all_y = np.concatenate(all_y_coords)

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    plt.xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory comparison saved to: {save_path}")

def create_animation_from_images(image_paths: List[str], output_path: str,
                                fps: int = 10):
    """Create an animation from a sequence of image files when imageio is available."""
    try:
        import imageio

        images = []
        for path in image_paths:
            if os.path.exists(path):
                images.append(imageio.imread(path))

        if images:
            imageio.mimsave(output_path, images, fps=fps)
            print(f"Animation saved to: {output_path}")
        else:
            print("No valid images found for animation")

    except ImportError:
        print("imageio not available, animation creation skipped")
    except Exception as e:
        print(f"Failed to create animation: {str(e)}")
