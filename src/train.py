import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import traceback
from src.models.model import HNav
from src.utils.device import to_device, get_device, release_cuda, clear_cuda_cache
from src.utils.config import DataDict, LossNames, GeneratorType
from src.data_loader.data_loader import train_data_loader, evaluation_data_loader

from src.utils.visualize import plot_trajectory_comparison, write_png, plot_trajectory_on_pointcloud
from src.utils.device import release_cuda
from src.utils.transform import apply_transformation 
try:
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    Data = None
    Batch = None

def calculate_jerk(trajectory: np.ndarray, dt: float = 0.1) -> float:
    """Compute the mean jerk magnitude of a sampled trajectory."""
    if trajectory.shape[0] < 4:
        return 0.0
    # v = diff(p) / dt
    velocity = np.diff(trajectory, axis=0) / dt
    # a = diff(v) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    # jerk = diff(a) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    # magnitude of jerk vectors
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    return np.mean(jerk_magnitude)
def check_collisions(trajectory: torch.Tensor, point_cloud: torch.Tensor, collision_radius: float = 0.5) -> bool:
    """Check whether any trajectory point lies within the collision radius of the point cloud."""
    if point_cloud.shape[0] == 0 or trajectory.shape[0] == 0:
        return False
    
    traj_points = trajectory[:, :2].float()
    pc_points = point_cloud[:, :2].float()
    
    # (M, N)
    distances = torch.cdist(traj_points, pc_points)
    
    min_distances_to_pc, _ = torch.min(distances, dim=1)
    
    if torch.any(min_distances_to_pc < collision_radius):
        return True
        
    return False


class Trainer:
    """Manage model optimization, checkpointing, and trajectory validation."""

    def __init__(self, config=None, cfg=None):
        """Initialize trainer state, model, optimizer, data loaders, and checkpoints."""
        if config is not None:
            self.config = config
            self.cfg = config.training
            self.model_cfg = config.model
            self.data_cfg = config.dataset
            self.loss_cfg = config.loss
            self.inference_cfg = getattr(config, 'inference', None)
        elif cfg is not None:
            self.cfg = cfg
            self.config = cfg
            self.model_cfg = getattr(cfg, 'model', cfg)
            self.data_cfg = getattr(cfg, 'dataset', getattr(cfg, 'data', cfg))
            self.loss_cfg = getattr(cfg, 'loss', cfg)
            self.inference_cfg = getattr(cfg, 'inference', None)
        else:
            raise ValueError("Either config or cfg must be provided")

        self.name = getattr(self.cfg, 'name', 'DRIFT_Training')

        if hasattr(self.cfg, 'max_epoch'):
            self.max_epoch = self.cfg.max_epoch
        elif hasattr(self.cfg, 'max_epochs'):
            self.max_epoch = self.cfg.max_epochs
        else:
            self.max_epoch = 150

        self.evaluation_freq = getattr(self.cfg, 'evaluation_freq', 5)
        self.save_freq = getattr(self.cfg, 'save_freq', 10)
        self.log_freq = getattr(self.cfg, 'log_freq', 100)
        self.iteration = 0
        self.epoch = 0
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.training = False
        self.early_stopping = getattr(self.cfg, 'early_stopping', False)
        self.patience = getattr(self.cfg, 'patience', 20)
        self.patience_counter = 0
        self.min_delta = getattr(self.cfg, 'min_delta', 1e-4)

        self._setup_device_and_distributed()

        output_dir_attr = getattr(self.cfg, 'output_dir', './results')
        self.output_dir = Path(output_dir_attr)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.log_path = self.output_dir / "train_log.txt"
        self._init_log_file()

        try:
            self.model = HNav(config=self.config, device=self.device)
            self._log_info(" Model (HNav) initialized successfully.")
        except Exception as e:
            self._log_info(f" Model initialization failed: {e}")
            raise

        self._log_info(f"Moving model to device: {self.device}")
        self.model = self.model.to(self.device)

        # self.model = torch.compile(self.model)

        if torch.cuda.is_available() and 'cuda' in str(self.device):
            model_device = next(self.model.parameters()).device
            self._log_info(f"Current model device: {model_device}")
            if model_device != self.device:
                self._log_info(f"Warning: model device {model_device} does not match expected device {self.device}")

        gpus_cfg = getattr(self.cfg, 'gpus', self.cfg)
        self.use_amp = getattr(gpus_cfg, 'mixed_precision', False)
        if self.use_amp:
            self._log_info("Enabling mixed-precision training")
            if 'cuda' not in str(next(self.model.parameters()).device):
                raise RuntimeError("Mixed-precision training requires a GPU model")

        self._setup_model_parallel()
        self._setup_optimizer_and_scheduler()

        self._setup_data_loaders()
        self.scaler = GradScaler() if self.use_amp else None

        snapshot = getattr(self.cfg, 'snapshot', None)
        only_model = getattr(self.cfg, 'only_model', False)
        if snapshot:
            self.load_checkpoint(snapshot, strict=not only_model)

        self._log_info(f"Trainer initialized. Device: {self.device}, distributed: {self.distributed}")

    def _setup_device_and_distributed(self):
        try:
            gpus_cfg = getattr(self.cfg, 'gpus', self.cfg)
            if hasattr(gpus_cfg, 'device'):
                device_attr = gpus_cfg.device
            elif hasattr(self.model_cfg, 'device'):
                device_attr = self.model_cfg.device
            else:
                device_attr = 'cuda:0'
            if device_attr == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device(device_attr)

            self.distributed = getattr(gpus_cfg, 'distributed', False) or getattr(self.data_cfg, 'distributed', False)
            self.local_rank = getattr(gpus_cfg, 'local_rank', 0)
            if self.distributed:
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl')
                self.local_rank = dist.get_rank()
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f'cuda:{self.local_rank}')
        except Exception as e:
            warnings.warn(f"Device setup failed; using defaults: {e}")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.distributed = False
            self.local_rank = 0

    def _init_log_file(self):
        if self.local_rank == 0:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n"
                        f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Experiment: {self.name}\n{'='*60}\n")

    def _log_info(self, message: str):
        if self.local_rank == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{timestamp}] {message}\n"
            print(f"[INFO] {message}")
            try:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(log_message)
            except Exception as e:
                print(f"Failed to write log: {e}")

    def _setup_model_parallel(self):
        if self.distributed:
            gpus_cfg = getattr(self.cfg, 'gpus', self.cfg)
            if getattr(gpus_cfg, 'sync_bn', False):
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        elif torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _setup_optimizer_and_scheduler(self):
        model_params = self.model.module.parameters() if isinstance(self.model, (DDP, torch.nn.DataParallel)) else self.model.parameters()
        optimizer_name = getattr(self.cfg, 'optimizer', 'adam').lower()
        lr = getattr(self.cfg, 'lr', 1e-4)
        weight_decay = getattr(self.cfg, 'weight_decay', 1e-5)

        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay, momentum=getattr(self.cfg, 'momentum', 0.9))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler_type = getattr(self.cfg, 'scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=getattr(self.cfg, 'lr_tm', self.max_epoch), eta_min=getattr(self.cfg, 'lr_min', 1e-7))
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=getattr(self.cfg, 'lr_step_size', 30), gamma=getattr(self.cfg, 'lr_gamma', 0.1))
        else:
            self.scheduler = None

    def _setup_data_loaders(self):
        try:
            # self.config.dataset.num_workers = 0
            # self._log_info("DEBUG: DataLoader num_workers forced to 0 for debugging.")
            self._log_info(f"DataLoader num_workers: {getattr(self.config.dataset, 'num_workers', 0)}")
        
            self.train_loader = train_data_loader(self.config)
            self._log_info(f"Training data loader ready; batch size: {getattr(self.data_cfg, 'batch_size', 'unknown')}")
            
            self.val_loader = evaluation_data_loader(self.config)
            self._log_info("Validation data loader ready")
        except Exception as e:
            self._log_info(f"Failed to configure data loaders: {e}")
            raise

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        if self.local_rank == 0:
            model_state = self.model.module.state_dict() if isinstance(self.model, (DDP, torch.nn.DataParallel)) else self.model.state_dict()
            checkpoint = {
                'epoch': self.epoch, 'iteration': self.iteration, 'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(), 'best_loss': self.best_loss,
                'config': self.config if hasattr(self, 'config') else self.cfg
            }
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            try:
                torch.save(checkpoint, filepath)
                self._log_info(f"Checkpoint saved: {filepath}")
                if is_best:
                    torch.save(checkpoint, self.ckpt_dir / "best.pth")
                    self._log_info(f"Best model saved: {self.ckpt_dir / 'best.pth'}")
            except Exception as e:
                self._log_info(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filepath: str, strict: bool = True):
        """Restore model and optional optimizer state from a checkpoint."""
        try:
            if not os.path.exists(filepath):
                self._log_info(f"Warning: checkpoint does not exist: {filepath}. Starting from scratch.")
                return

            checkpoint = torch.load(filepath, map_location=self.device)
            self._log_info(f" Loading checkpoint: {filepath}")

            model_to_load = self.model.module if isinstance(self.model, (DDP, torch.nn.DataParallel)) else self.model
            
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self._log_info("Model weights loaded (non-strict mode).")
            
            if strict:
                self.epoch = checkpoint.get('epoch', 0)
                self.start_epoch = self.epoch
                self.iteration = checkpoint.get('iteration', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                if 'optimizer_state_dict' in checkpoint and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self._log_info("Optimizer state loaded.")

                if 'scheduler_state_dict' in checkpoint and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self._log_info("Scheduler state loaded.")

                if 'scaler_state_dict' in checkpoint and self.scaler:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    self._log_info("GradScaler state loaded.")
            else:
                self.epoch = checkpoint.get('epoch', 0)
                self.start_epoch = self.epoch
                self._log_info(f"Resuming from epoch {self.start_epoch + 1}.")


        except Exception as e:
            self._log_info(f" Critical checkpoint load error: {e}")
            if strict:
                raise

    def _check_model_device(self):
        try:
            model_params = list(self.model.module.parameters()) if isinstance(self.model, (DDP, torch.nn.DataParallel)) else list(self.model.parameters())
            if model_params:
                model_device = model_params[0].device
                self._log_info(f"Model parameter device: {model_device}")
                self._log_info(f"Expected device: {self.device}")
            else:
                self._log_info("Model has no parameters")
        except Exception as e:
            self._log_info(f"Failed to inspect model device: {e}")

    def _check_data_device_dtype(self, data, batch_idx=0):
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    if torch.is_tensor(value):
                        self._log_info(f"Batch {batch_idx} - {key}: device={value.device}, dtype={value.dtype}, shape={value.shape}")
                    elif hasattr(value, '__class__') and 'Data' in str(value.__class__):
                        self._log_info(f"Batch {batch_idx} - {key}: type={type(value)} (PyG Data/Batch)")
                        for attr_name in ['x', 'edge_index', 'edge_attr', 'pos', 'batch']:
                            if hasattr(value, attr_name) and torch.is_tensor(getattr(value, attr_name)):
                                attr_value = getattr(value, attr_name)
                                self._log_info(f"  - {attr_name}: device={attr_value.device}, dtype={attr_value.dtype}, shape={attr_value.shape}")
        except Exception as e:
            self._log_info(f"Failed to inspect data device: {e}")

    def _forward_step(self, data: Dict[str, Any], epoch: int) -> Dict[str, torch.Tensor]:
        """Move a batch to the device and compute model-provided training losses."""
        outputs = self.model(data, epoch=epoch)
        if 'loss' not in outputs:
            raise ValueError("Model forward must return a dictionary containing the 'loss' key.")
        return outputs

    def _backward_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()

            grad_clip_norm = getattr(self.cfg, 'grad_clip_norm', 1.0)
            if grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            grad_clip_norm = getattr(self.cfg, 'grad_clip_norm', 1.0)
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

            self.optimizer.step()

    def _log_training_step(self, loss_info: Dict[str, torch.Tensor]):
        lr = self.optimizer.param_groups[0]['lr']
        loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_info.items() if torch.is_tensor(v) and v.numel() == 1])
        self._log_info(f"Epoch {self.epoch}, Step {self.iteration}, LR: {lr:.2e} | {loss_str}")

    def _debug_forward_step(self, data):
        """Log batch and output devices while computing a diagnostic forward pass."""
        self._log_info("=== Starting forward-pass debug ===")
        self._log_info("Inspecting input data:")
        self._check_data_device_dtype(data, "forward")
        
        self._log_info("Inspecting model state:")
        self._check_model_device()

        self._log_info("Running model forward pass...")
        outputs = self.model(data, epoch=0)
        self._log_info(f"Model output type: {type(outputs)}")
        for key, value in outputs.items():
            if torch.is_tensor(value):
                self._log_info(f"Output {key}: device={value.device}, dtype={value.dtype}, shape={value.shape}")
        
        self._log_info("Model computed the loss internally.")
        self._log_info("=== Forward-pass debug complete ===")
        return outputs

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.training = True
        
        if self.epoch == 0:
            self._check_model_device()
        
        loss_dict = {}
        # Use the train_time_steps from the config, default to 1 if not present
        inner_train_steps = getattr(self.cfg, 'train_time_steps', 1)
        max_train_batches = getattr(self.cfg, 'max_train_batches', 0)
        processed_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", disable=(self.local_rank != 0))

        for batch_idx, data in enumerate(pbar):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            if not data:
                warnings.warn(f"Validation Epoch {self.epoch}: Skipped an empty/invalid batch.")
                continue
            # ==============================
            try:
                data = to_device(data, self.device)
                
                # This loop is essential for stabilizing diffusion model training.
                for step_iteration in range(inner_train_steps):
                    with autocast(enabled=self.use_amp):
                        loss_info = self._forward_step(data, epoch=self.epoch)
                        loss = loss_info['loss']

                    if torch.isnan(loss) or torch.isinf(loss):
                        self._log_info(f"Detected abnormal loss value: {loss.item()} in inner step, skipping batch.")
                        break # Exit inner loop for this batch
                    
                    # A full backward pass and optimizer step occurs in EACH inner iteration
                    self._backward_step(loss)

                    # Accumulate loss for logging purposes
                    for key, value in loss_info.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            loss_dict.setdefault(key, 0.0)
                            loss_dict[key] += value.item()
                
                self.iteration += 1 # Outer iteration count increments once per batch
                processed_batches += 1

                if self.local_rank == 0:
                    # Update progress bar with the loss from the last inner step
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"})
                
                if self.iteration % self.log_freq == 0 and self.local_rank == 0:
                    self._log_training_step(loss_info)

            except Exception as e:
                self._log_info(f"Batch {batch_idx} training failed: {e}\n{traceback.format_exc()}")
                if batch_idx < 5: raise

        # Normalize the accumulated loss by the total number of inner steps performed
        total_steps = processed_batches * inner_train_steps
        return {key: value / total_steps for key, value in loss_dict.items()} if total_steps > 0 else {}

    def visualize_epoch_end(self):
        if self.val_loader is None or self.local_rank != 0:
            return

        self._log_info(f"--- Epoch {self.epoch} | Generating visualization sample ---")
        self.model.eval()

        try:
            data = next(iter(self.val_loader))
            data = to_device(data, self.device)

            with torch.no_grad():
                output_dict = self.model.sample(data, num_samples=1)

            pred_traj_local = release_cuda(output_dict['prediction'])[0]
            pose = release_cuda(data['pose'])[0]
            gt_traj_world = release_cuda(data['traj_gt_world'])[0]
            point_cloud_local = release_cuda(data['local_graph'].pos)
            
            pred_traj_world = apply_transformation(pred_traj_local, pose)
            point_cloud_world = apply_transformation(point_cloud_local, pose)

            save_path_2d = self.output_dir / "visualizations" / f"epoch_{self.epoch}_2D_comparison.png"
            os.makedirs(save_path_2d.parent, exist_ok=True)
            plot_trajectory_comparison(
                gt_traj=gt_traj_world, 
                pred_trajs=[pred_traj_world],
                labels=["Prediction"],
                save_path=str(save_path_2d)
            )
            self._log_info(f" Epoch-end 2D visualization saved to: {save_path_2d}")

            save_path_3d = self.output_dir / "visualizations" / f"epoch_{self.epoch}_3D_context.png"
            plot_trajectory_on_pointcloud(
                pc=point_cloud_world,
                traj=gt_traj_world,
                save_path=str(save_path_3d),
                title=f"Epoch {self.epoch} Trajectory in Point Cloud"
            )
            self._log_info(f" Epoch-end 3D context visualization saved to: {save_path_3d}")

        except Exception as e:
            self._log_info(f" Failed to generate epoch-end visualization: {e}\n{traceback.format_exc()}")
        finally:
            self.model.train()

    def run(self):
        self._log_info(f"Starting training from epoch {self.start_epoch + 1} through epoch {self.max_epoch}")
        self._log_info("=== Pre-training system check ===")
        self._log_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): self._log_info(f"CUDA version: {torch.version.cuda}")
        self._check_model_device()

        self._log_info("=== Testing data loading and forward pass ===")
        try:
            test_batch = next(iter(self.train_loader))
            self._log_info(" Test batch loaded")
            test_batch = to_device(test_batch, self.device)
            self._log_info(" Data moved to target device")

            self.model.train()
            with autocast(enabled=self.use_amp):
                loss_info = self._debug_forward_step(test_batch)
                if 'loss' not in loss_info or not loss_info['loss'].requires_grad:
                    raise RuntimeError("System check failed: invalid loss or missing gradients.")
                self._log_info(" Forward-pass test succeeded")
                self._log_info(f"Total loss 'loss' requires_grad: {loss_info['loss'].requires_grad}")
            self._log_info("=== System check complete ===")
        except Exception as e:
            self._log_info(f"First batch test failed: {e}\n{traceback.format_exc()}")
            raise

        self._log_info("=== Starting full training ===")
        for epoch in range(self.start_epoch + 1, self.max_epoch):
            self.epoch = epoch
            train_loss_dict = self.train_epoch()

            if self.scheduler: self.scheduler.step()
            
            self.visualize_epoch_end()
            
            val_loss_dict = {}
            if epoch % self.evaluation_freq == 0 or epoch == self.max_epoch - 1:
                val_loss_dict = self.validate()
                current_loss = val_loss_dict.get('loss', float('inf'))
                if current_loss < self.best_loss - self.min_delta:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                    self.save_checkpoint(self.ckpt_dir / "best.pth", is_best=True)
                else:
                    self.patience_counter += 1

            self._log_epoch_results(epoch, train_loss_dict, val_loss_dict)
            
            self.save_checkpoint(self.ckpt_dir / f"checkpoint_epoch_{epoch}.pth")
            self._log_info(f"Epoch {epoch} complete; checkpoint checkpoint_epoch_{epoch}.pth saved.")


            if self.early_stopping and self.patience_counter >= self.patience:
                self._log_info(f" Early stopping after {self.patience} epochs without improvement")
                break
            clear_cuda_cache()

        self._log_info(" Training complete")

    def validate(self, success_radius: float = 1.0, collision_radius: float = 0.5) -> Dict[str, float]:
        """Run validation inference and aggregate losses and trajectory metrics."""
        if self.val_loader is None: return {}
        self.model.eval()
        self.training = False
        
        all_losses = {}
        inference_times = []
        path_length_ratios = []
        final_displacement_errors = []
        jerks = []
        collision_count = 0
        total_samples = 0
        max_val_batches = getattr(self.cfg, 'max_val_batches', 0)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.epoch}", disable=(self.local_rank != 0))
            for batch_idx, data in enumerate(pbar):
                if max_val_batches > 0 and batch_idx >= max_val_batches:
                    break
                if not data:
                    warnings.warn(f"Validation Epoch {self.epoch}: Skipped an empty/invalid batch.")
                    continue
                # ==============================
                data = to_device(data, self.device)
                
                start_time = time.time()
                with autocast(enabled=self.use_amp):
                    output_dict = self.model.sample(data, num_samples=1)
                inference_times.append(time.time() - start_time)

                pred_traj_local = output_dict['prediction']
                gt_traj_local = data['traj_gt_local']
                point_cloud_local = data['local_graph'].pos
                batch_map = data['local_graph'].batch
                batch_size = pred_traj_local.shape[0]
                total_samples += batch_size

                for i in range(batch_size):
                    pred_traj_i = release_cuda(pred_traj_local[i])
                    gt_traj_i = release_cuda(gt_traj_local[i])
                    
                    fde = np.linalg.norm(pred_traj_i[-1] - gt_traj_i[-1])
                    final_displacement_errors.append(fde)
                    
                    pc_mask = (batch_map == i)
                    pc_i = point_cloud_local[pc_mask]
                    if check_collisions(torch.from_numpy(pred_traj_i).to(self.device), pc_i, collision_radius):
                        collision_count += 1
                    
                    len_pred = np.linalg.norm(pred_traj_i[1:] - pred_traj_i[:-1], axis=-1).sum()
                    len_gt = np.linalg.norm(gt_traj_i[1:] - gt_traj_i[:-1], axis=-1).sum()
                    if len_gt > 1e-6:
                        path_length_ratios.append(len_pred / len_gt)
                    
                    jerks.append(calculate_jerk(pred_traj_i))

        results = {}
        if total_samples > 0:
            success_count = sum(1 for fde in final_displacement_errors if fde < success_radius)
            results['Success_Rate_%'] = (success_count / total_samples) * 100
            results['Collision_Rate_%'] = (collision_count / total_samples) * 100
            results['Avg_Inference_Time_ms'] = np.mean(inference_times) * 1000
            results['FDE_mean'] = np.mean(final_displacement_errors)
            results['Path_Ratio_mean'] = np.mean(path_length_ratios) if path_length_ratios else 0.0
            results['Jerk_mean'] = np.mean(jerks) if jerks else 0.0
            results['FDE_std'] = np.std(final_displacement_errors)
            results['Path_Ratio_std'] = np.std(path_length_ratios) if path_length_ratios else 0.0

        self.model.train()
        self.training = True
        return results
    
    def _log_epoch_results(self, epoch: int, train_loss: Dict[str, float], val_loss: Dict[str, float]):
        if self.local_rank == 0:
            train_loss_str = " | ".join([f"Train_{k}: {v:.4f}" for k, v in train_loss.items()])
            val_loss_str = " | ".join([f"Val_{k}: {v:.4f}" for k, v in val_loss.items()]) if val_loss else "No Val"
            lr = self.optimizer.param_groups[0]['lr']
            self._log_info(f"Epoch {epoch} complete | LR: {lr:.2e} | {train_loss_str} | {val_loss_str}")

    def test_and_visualize(self, ckpt_path: str, n_samples: int = 100) -> Dict[str, Any]:
        self.load_checkpoint(ckpt_path, strict=False)
        self.model.eval()
        results = {}
        if self.val_loader:
            with torch.no_grad():
                eval_results = self.validate() 
                results.update(eval_results)
        self._log_info(f"Test complete; results: {results}")
        return results
