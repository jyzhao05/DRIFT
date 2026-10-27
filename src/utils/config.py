"""Default configuration schemas, enumerations, and lightweight preset constructors for DRIFT."""

import math
from os.path import join
import numpy as np
from easydict import EasyDict as edict


#########################################################################
# Data Dictionary Keys 
#########################################################################
class DataDict:
    # Core data fields
    pose = "pose"
    time = "time"
    camera = "camera"
    scan = "scan"
    lidar = "lidar"
    lidar2d = "lidar2d"
    local_map = "local_map"
    vel = "vel"
    imu = "imu"
    target = "target"
    raw_target = "raw_target"
    path = "path"
    heuristic = "heuristic"
    
    # Multi-path and trajectory fields
    all_paths = "all_paths"
    targets = "targets"
    trajectories = "trajectories"
    ori_trajectories = "ori_trajectories"
    all_trajectories = "all_trajectories"
    
    # File and metadata
    file_names = "file_names"
    all_positions = "all_positions"
    frame_name = "frame_name"
    images = "images"
    running_time = "running_time"
    
    # Network outputs and predictions
    network = "network"
    prediction = "prediction"
    predict_path = "predict_path"
    ground_truth = "ground_truth"
    embedding = "embedding"
    
    # Graph and attention specific
    local_graph = "local_graph"
    batch_graphs = "batch_graphs"
    node_embeddings = "node_embeddings"
    c_cmag = "c_cmag"
    attention_weights = "attention_weights"
    
    # Diffusion specific
    time_steps = "time_steps"
    noise = "noise"
    noise_target = "noise_target"
    zmu = "mu"
    zvar = "logvar"
    all_variances = "all_variances"
    
    # Traversability
    traversable_step = "traversable_step"
    traversability_gt = "traversability_gt"
    traversability_pred = "traversability_pred"
    traversability_mu = "traversability_mu"
    traversability_var = "traversability_var"
    
    # Loss computation matrices
    A = "A"
    b = "b"


#########################################################################
# Enum Classes
#########################################################################
class CameraType:
    realsense_d435i = 0
    realsense_l515 = 1


class DatasetType:
    test = "test"
    train = "train"
    val = "val"


class GeneratorType:
    diffusion = 0
    cvae = 1


class RNNType:
    gru = 0
    lstm = 1


class DiffusionModelType:
    unet = "unet"
    crnn = "crnn"
    cmagnn = "cmagnn"


class CRNNType:
    lstm = "lstm"
    gru = "gru"


class Hausdorff:
    average = "average"
    max = "max"


class ScheduleMethods:
    step = "step"
    cosine = "cosine"
    exponential = "exponential"


#########################################################################
# Loss and Log Names
#########################################################################
class LossNames:
    # Basic losses
    kld = "kld"
    path_dis = "path_dis"
    last_dis = "last_dis"
    traversability = "traversability"
    
    # Diffusion specific
    l_simple = "l_simple"
    main = "main"
    total = "total"
    
    # Evaluation metrics
    evaluate_last_dis = "evaluate_last_dis"
    evaluate_path_dis = "evaluate_path_dis"
    evaluate_traversability = "evaluate_traversability"
    
    # Total loss
    loss = "loss"


class LogNames:
    step_time = "step_time"
    lr = "learning_rate"
    memory_usage = "memory_usage"
    graph_stats = "graph_statistics"


class LogTypes:
    train = "train"
    val = "validation"
    test = "test"
    others = "evaluation"


#########################################################################
# Base Configuration Templates
#########################################################################

# LiDAR Configuration Template
def create_lidar_config():
    """Create the LiDAR preprocessing configuration template."""
    return edict({
        'threshold': 50.0,
        'channels': 64,
        'horizons': 1024,
        'angle_range': 360,
        
        'max_frames': 3,
        'frame_time_interval': 0.1,
        'use_motion_compensation': True,
        'motion_estimation_method': 'pose_based',  # 'pose_based', 'icp', 'simple'
        'fallback_to_single_frame': True,
        
        'min_points_per_frame': 100,
        'max_points_per_frame': 50000,
        'remove_ground': False,
        'height_filter': None,
    })


def create_graph_config():
    """Define voxelization, neighborhood, and graph feature settings."""
    return edict({
        'voxel_size': 0.2,
        'k_nn': 8,
        'max_nodes': 2000,
        'max_edges': 16000,
        'edge_feature_dim': 6,
        'node_feature_dim': 4,
        
        'use_edge_features': True,
        'use_self_loops': False,
        'radius_graph': False,
        'radius': 1.0,
        'max_neighbors': 16,
        
        'downsample_method': 'voxel',  # 'voxel', 'random', 'fps'
        'voxel_method': 'centroid',    # 'centroid', 'average', 'random'
        'preserve_structure': True,
        
        'node_features': ['position', 'normal', 'density', 'height'],
        'edge_features': ['distance', 'direction', 'angle'],
        'use_relative_positions': True,
        'normalize_features': True,
        
        'remove_isolated_nodes': True,
        'min_node_degree': 1,
        'graph_connectivity_check': True,
    })


def create_dataset_config():
    """Create the dataset and preprocessing configuration template."""
    return edict({
        'name': "DRIFT_Dataset",
        'root': "",
        'batch_size': 16,
        'num_workers': 8,
        'shuffle': True,
        'distributed': False,
        'training_data_percentage': 0.8,
        'validation_data_percentage': 0.1,
        
        'processed_data_dir_name': "processed_data",
        'traj_len': 16,
        'history_len': 0,
        'vel_num': 10,
        'imu_num': 0,
        
        'pose_config': edict({
            'format': 'auto_detect',  # 'auto_detect', '3x4', '4x4'
            'coordinate_system': 'robot_frame',  # 'robot_frame', 'world_frame'
            'validate_pose': True,
            'pose_smoothing': False,
            'fallback_to_identity': True,
        }),
        
        'motion_config': edict({
            'use_motion_compensation': True,
            'estimation_method': 'velocity_based',  # 'velocity_based', 'constant_velocity', 'zero_motion'
            'estimated_speed': 1.0,
            'max_speed': 10.0,
            'angular_velocity': 0.1,
            'use_icp_fallback': False,
        }),
        
        'validation_config': edict({
            'check_data_integrity': True,
            'min_trajectory_length': 2,
            'max_trajectory_length': 200,
            'check_pose_validity': True,
            'check_lidar_validity': True,
            'skip_invalid_samples': True,
            'log_invalid_samples': True,
        }),
        
        'augmentation_config': edict({
            'enable_augmentation': False,
            'rotation_range': 0.1,
            'translation_range': 0.5,
            'noise_level': 0.01,
            'random_dropout': 0.05,
        }),
        
        'lidar_cfg': None,
        'graph_cfg': None
    })

# Perception Configuration Template
def create_perception_config():
    """Create the graph perception configuration template."""
    return edict({
        'fix_perception': False,
        'cache_embeddings': True,
        
        # GNN backbone configuration
        'gnn_cfg': edict({
            'in_feats': 4,  # x,y,z + density
            'hidden_dims': [64, 128, 256, 128],
            'out_dim': 128,
            'gnn_type': 'edgeconv',  # or 'gat'
            'pool_method': 'max',
            'heads': 4,  # for GAT
            'dropout': 0.1
        }),
        
        # Velocity encoder configuration
        'vel_cfg': edict({
            'input_dim': 5,
            'hidden_dim': 64,
            'out_dim': 32,
            'encoder_type': 'cnn'  # 'cnn', 'lstm', or 'mlp'
        }),
        
        # Target encoder configuration
        'target_cfg': edict({
            'input_dim': 2,
            'hidden_dim': 64,
            'out_dim': 32
        }),
        
        # Feature fusion configuration
        'fusion_cfg': edict({
            'hidden_dim': 256,
            'out_dim': 128  # c_cmag_dim
        }),
        
        # Graph processing parameters
        'voxel_size': 0.2,
        'k_nn': 8,
        'max_nodes': 2000,
        
        # Output dimensions
        'output_dim': 128,  # c_cmag_dim
        'gnn_output_dim': 128  # node embeddings dimension
    })


# CVAE Configuration Template
def create_cvae_config():
    """Build the retained CVAE configuration template."""
    return edict({
        'activation_func': "relu",
        'rnn_type': RNNType.gru,
        'perception_in': 130,  # Will be updated by factory
        'vae_zd': 512,
        'vae_output_threshold': 1,
        'paths_num': 5,
        'waypoints_num': 50,
        'waypoint_dim': 2,
        'estimate_traversability': True,
        'fix_first': False,
        'cvae_file': None
    })


# CMAGNN Generator Configuration Template
def create_generator_config():
    """Define attention, recurrent, and conditioning settings for the generator."""
    return edict({
        # The GNN now predicts a smaller set of control points
        'num_control_points': 16,
        # Input/utput dimensions
        'hidden_dim': 256,
        'time_emb_dim': 128,
        
        # Cross-modal attention parameters
        'attention_heads': 4,
        'k_env': 16,  # Number of environment nodes per trajectory point
        # Architecture ablation switches. The defaults keep canonical DRIFT
        # unchanged. Non-dynamic conditioning modes are default-off controls
        # for mechanism ablations.
        'conditioning_mode': 'dynamic',
        'denoiser_cell': 'gtgru',
        
        # GNN parameters
        'num_gnn_layers': 3,
        
        # I/O dimensions
        'in_feats': 2,  # trajectory point dimension (x, y)
        'out_feats': 2,  # output dimension
        
        'vel_input_dim': 5,
        'target_input_dim': 2,
        'vel_embed_dim': 32,
        'target_embed_dim': 32

    })


# Diffusion Configuration Template
def create_diffusion_config():
    """Create the diffusion model configuration template."""
    return edict({
        'traversable_steps': 10,
        # Basic diffusion parameters
        'timesteps': 100,
        'sampling_timesteps': 6,
        'loss_type': 'l2',  # 'l1', 'l2', 'huber'
        'schedule_type': 'cosine',  # 'cosine', 'linear'
        
        # Linear schedule parameters
        'beta_start': 0.0001,
        'beta_end': 0.02,
        
        # Classifier-free guidance
        'classifier_free_guidance_scale': 1.0,
        'classifier_free_dropout': 0.1,
        
        # Auxiliary loss weights
        # 'aux_loss_weights': edict({
        #     'smoothness': 0.1,
        #     'feasibility': 0.05,
        #     'target': 0.1,
        #     'velocity': 0.02
        # }),
        'aux_loss_weights': edict({
            'smoothness': 0.1,
            'feasibility': 1.0,
            'target': 0.5,
            'velocity': 0.05
        }),
        
        # Legacy parameters for compatibility
        'model_type': DiffusionModelType.cmagnn,
        'perception_in': 130,  # Will be updated by factory
        'diffusion_zd': 512,
        'waypoint_dim': 2,
        'waypoints_num': 50,  # Will be updated by factory
        'use_traversability': True,
        'estimate_traversability': True,
        'clip_sample': True,
        'clip_sample_range': 1.0,

        'crnn': edict({
            'in_dim': 2,
            'out_dim': 2,
            'hidden_dim': 256,
            'diffusion_step_embed_dim': 128,
            'steps': 1,
            'rnn_type': RNNType.gru,
            'output_threshold': 1.0, # 
            # 'output_threshold': None,
        }),
    })


# Model Configuration Template
def create_model_config():
    """Create the model-level configuration template."""
    return edict({
        'generator_type': GeneratorType.diffusion,
        'scale_waypoints': 1.0,
        
        # Component configurations (will be populated by factory)
        'perception': None,
        'cvae': None,
        'diffusion': None,
        'generator': None,
        
        # Device configuration
        'device': "cuda:0",
        'mixed_precision': False
    })


def create_loss_config():
    """Create the auxiliary loss and trajectory objective configuration."""
    return edict({
        'generator_type': GeneratorType.diffusion,
        'ablation_profile': 'full',
        'stage_ratio_overrides': edict({
            'imitation': edict(),
            'planning': edict(),
        }),
        'train_poses': True,
        'scale_waypoints': 1.0,
        'distance_type': Hausdorff.average,

        'distance_ratio': 10.0,
        
        'last_ratio': 50,
        'start_ratio': 50,

        'traversability_ratio': 0.0,
        'use_traversability': False,
        'collision_distance': 0.5,  # meters
        'map_resolution': 0.05,     # meters/pixel
        'map_range': 64,            # map center offset
        
        'output_dir': None
    })
# Training Configuration Template
def create_training_config():
    """Create the optimizer, schedule, validation, and checkpoint configuration."""
    return edict({
        'adaptive_training': True,
        'aux_loss_warmup_epochs': 10,
        'traversability_threshold': 1e-7,
        
        'name': "DRIFT_Training",
        'only_model': False,
        'output_dir': "./results",
        'snapshot': None,
        'resume_from_checkpoint': True,
        
        # Training schedule
        'max_epoch': 150,
        'stage1_epochs': 2,
        'evaluation_freq': 5,
        'save_freq': 10,
        'log_freq': 100,
        'max_train_batches': 0,
        'max_val_batches': 0,

        'adaptive_training': True,
        'aux_loss_warmup_epochs': 10,
        # =================================
        
        # Optimizer settings
        'optimizer': "adam",
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip_norm': 1.0,
        
        
        # Scheduler settings
        'scheduler': ScheduleMethods.cosine,
        'lr_t0': 1,
        'lr_tm': 30,
        'lr_min': 1e-7,
        
        # Diffusion specific
        'train_time_steps': 32,
        'traversability_threshold': 1e-7,
        
        # Early stopping
        'early_stopping': True,
        'patience': 20,
        'min_delta': 1e-4,
        
        # Validation settings
        'val_batch_size': 8,
        'val_num_samples': 5,
        'visualize_on_val': True,
        'num_viz_batches': 1,
        
        # GPU settings
        'gpus': edict({
            'channels_last': False,
            'local_rank': 0,
            'sync_bn': False,
            'no_ddp_bb': False,
            'device': "cuda:0",  # Will be synced by factory
            'mixed_precision': False  # Will be synced by factory
        })
    })


# Inference Configuration Template
def create_inference_config():
    """Create the inference and sampling configuration template."""
    return edict({
        'batch_size': 1,
        'num_samples': 10,
        'use_cache': True,
        'visualize_results': True,
        'save_trajectories': True,
        'output_format': "png",
        
        # Diffusion sampling parameters
        'num_inference_steps': 100,
        'guidance_scale': 1.0,
        'eta': 0.0,
        
        # Post-processing
        'smooth_trajectories': True,
        'collision_check': True,
        'max_trajectory_length': 50  # Will be updated by factory
    })


#########################################################################
# Base Configuration Factory
#########################################################################
def create_base_config():
    """Construct the complete lightweight base configuration."""
    # Create all sub-configurations
    lidar_cfg = create_lidar_config()
    graph_cfg = create_graph_config()
    dataset_cfg = create_dataset_config()
    perception_cfg = create_perception_config()
    cvae_cfg = create_cvae_config()
    generator_cfg = create_generator_config()
    diffusion_cfg = create_diffusion_config()
    model_cfg = create_model_config()
    loss_cfg = create_loss_config()
    training_cfg = create_training_config()
    inference_cfg = create_inference_config()
    
    # Populate sub-configuration references
    dataset_cfg.lidar_cfg = lidar_cfg
    dataset_cfg.graph_cfg = graph_cfg
    
    model_cfg.perception = perception_cfg
    model_cfg.cvae = cvae_cfg
    model_cfg.diffusion = diffusion_cfg
    model_cfg.generator = generator_cfg
    
    # Create main configuration
    config = edict({
        'dataset': dataset_cfg,
        'model': model_cfg,
        'loss': loss_cfg,
        'training': training_cfg,
        'inference': inference_cfg,
        
        # Global settings
        'seed': 42,
        'debug': False,
        'experiment_name': "drift_experiment",
        'version': "1.0.0"
    })
    
    return config


#########################################################################
# Default Configuration (lightweight creation only)
#########################################################################
# Only create the base configuration - no validation or heavy processing
Config = create_base_config()

# Configuration presets (lightweight creation only)
DiffusionConfigTemplate = create_base_config()
DiffusionConfigTemplate.model.generator_type = GeneratorType.diffusion
DiffusionConfigTemplate.loss.generator_type = GeneratorType.diffusion

CVAEConfigTemplate = create_base_config()
CVAEConfigTemplate.model.generator_type = GeneratorType.cvae
CVAEConfigTemplate.loss.generator_type = GeneratorType.cvae
