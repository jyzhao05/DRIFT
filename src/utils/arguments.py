import argparse
import os

import torch
from src.utils.config import GeneratorType, DiffusionModelType, CRNNType
from src.utils.config_utils import (
    ConfigManager, 
    create_default_config,
    create_diffusion_config,
    create_cvae_config,
    ConfigPrinter
)


def get_args():
    parser = argparse.ArgumentParser(description='DRIFT trajectory generation training')
    parser.add_argument('--name', type=str, default="hnav", help="name of project")
    # parser.add_argument('--wandb_api', type=str, default="", help="Your wandb api")
    parser.add_argument('--only_load_model', action='store_true', default=False,
                        help='only load model to continue training')
    parser.add_argument('--snapshot', type=str, default="", help='snapshot')
    parser.add_argument('--evaluation_freq', type=int, default=1, help="evaluation frequency")
    parser.add_argument('--train_time_steps', type=int, default=32, help="time steps for training")
    parser.add_argument('--training_type', type=int, default=1, help="0: 20 epochs; 1: 10 epochs")
    parser.add_argument('--debug_output', type=str, default=None, help='snapshot')
    parser.add_argument(
        '--loss_ablation',
        type=str,
        default='full',
        help="auxiliary loss profile: full, no_smoothness, no_collision, simple_only, simple_plus_endpoint",
    )
    parser.add_argument('--output_dir', type=str, default=None, help='training output directory override')
    parser.add_argument('--max_epoch_override', type=int, default=0, help='override training max_epoch when > 0')
    parser.add_argument('--stage1_epochs_override', type=int, default=-1, help='override stage1_epochs when >= 0')
    parser.add_argument('--train_fraction_override', type=float, default=0.0, help='override training_data_percentage when in (0, 1)')
    parser.add_argument('--max_train_batches', type=int, default=0, help='limit train batches per epoch when > 0')
    parser.add_argument('--max_val_batches', type=int, default=0, help='limit val batches per epoch when > 0')
    parser.add_argument('--seed', type=int, default=42, help='random seed for training/evaluation')

    # data args:
    parser.add_argument('--data_root', type=str, help='root of the dataset', 
                       default="./data")
    parser.add_argument('--processed_dir', type=str, default="processed_data_global", 
                        help="Name of the preprocessed data directory inside data_root")
    parser.add_argument('--batch_size', type=int, default=16, 
                       help="samples per training batch")
    parser.add_argument('--workers', type=int, default=16, 
                       help="the worker number in the dataloader")

    # model args:
    parser.add_argument('--generator_type', type=int, default=0, help="0: diffusion; 1: cvae")
    parser.add_argument('--diffusion_model', type=int, default=0, help="0: rnn; 1: unet")
    parser.add_argument('--crnn_type', type=int, default=0, help="0: gru; 1: lstm")
    parser.add_argument(
        '--conditioning_mode',
        type=str,
        default='dynamic',
        choices=['dynamic', 'static', 'pooled_scene', 'pooled_node_global'],
        help=(
            "Generator conditioning mode: dynamic=DRIFT retrieval, "
            "static=pooled node context at each step, pooled_scene=global scene only, "
            "pooled_node_global=legacy noSCA pooled-node context"
        ),
    )
    parser.add_argument(
        '--denoiser_cell',
        type=str,
        default='gtgru',
        choices=['gtgru', 'plain_gru'],
        help="Denoiser recurrent cell: gtgru=canonical DRIFT, plain_gru=standard GRU ablation",
    )
    parser.add_argument('--train_poses', action='store_true', default=False, 
                       help="if train poses or increments")
    parser.add_argument('--use_traversability', action='store_true', default=False, 
                       help="if train traversability")
    parser.add_argument('--traversable_steps', type=int, default=10, 
                       help="time steps used for traversability training")
    parser.add_argument('--diffusion_time_steps', type=int, default=100, help="")

    # Configuration file options
    parser.add_argument('--config_file', type=str, default=None, 
                       help="path to YAML configuration file")
    parser.add_argument('--save_config', type=str, default=None, 
                       help="path to save final configuration")
    parser.add_argument('--print_config', action='store_true', default=True,
                       help="print configuration summary")

    # GPUs
    parser.add_argument('--channels-last', action='store_true', default=False, 
                       help='Use channels_last memory layout')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true', default=False,
                       help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--split-bn', action='store_true', 
                       help='Enable separate BN layers per augmentation split.')
    parser.add_argument('--device', type=int, default=-1, help="the gpu id")
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                       help='Force broadcast buffers for native DDP to off.')
    return parser.parse_args()


def get_configuration():
    """Build the final configuration from command-line arguments and optional YAML input."""
    args = get_args()
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Determine how to create the initial configuration
    if args.config_file:
        # Load from YAML file
        print(f"Loading configuration from {args.config_file}")
        config = config_manager.load_config(args.config_file)
    else:
        # Create configuration based on generator type
        if args.generator_type == 0:
            print("Creating diffusion model configuration")
            config = create_diffusion_config()
        elif args.generator_type == 1:
            print("Creating CVAE model configuration")
            config = create_cvae_config()
        else:
            raise ValueError(f"Unknown generator type: {args.generator_type}")
    
    # Apply command line argument overrides
    config = apply_arg_overrides(config, args)
    
    # Synchronize dependent values and validate
    config = config_manager.synchronizer.synchronize_config(config)
    config_manager.validator.validate_config(config)
    
    # Print configuration summary if requested
    if args.print_config:
        ConfigPrinter.print_summary(config)
    
    # Save configuration if requested
    if args.save_config:
        config_manager.save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    return config


def apply_arg_overrides(config, args):
    """Apply command-line values to the corresponding configuration sections."""
    
    #########################################
    # Basic project configurations
    #########################################
    config.experiment_name = args.name
    config.dataset.name = args.name
    config.training.name = args.name
    config.seed = args.seed
    
    #########################################
    # Data configurations
    #########################################
    config.dataset.root = args.data_root
    config.dataset.batch_size = args.batch_size
    config.dataset.num_workers = args.workers
    config.dataset.processed_dir_name = args.processed_dir
    
    # Update loss config root if it exists (legacy compatibility)
    if hasattr(config.loss, 'root'):
        config.loss.root = args.data_root
    
    #########################################
    # Model configurations
    #########################################
    # Generator type (already set during config creation, but ensure consistency)
    if args.generator_type == 0:
        config.model.generator_type = GeneratorType.diffusion
        config.loss.generator_type = GeneratorType.diffusion
    elif args.generator_type == 1:
        config.model.generator_type = GeneratorType.cvae
        config.loss.generator_type = GeneratorType.cvae
    
    # Diffusion model specific settings
    if config.model.generator_type == GeneratorType.diffusion:
        if hasattr(config.model, 'generator'):
            config.model.generator.conditioning_mode = args.conditioning_mode
            config.model.generator.denoiser_cell = args.denoiser_cell
        if args.diffusion_model == 0:
            config.model.diffusion.model_type = DiffusionModelType.crnn
            # Handle CRNN type if the config structure supports it
            if hasattr(config.model.diffusion, 'crnn'):
                if args.crnn_type == 0:
                    config.model.diffusion.crnn.type = CRNNType.gru
                else:
                    config.model.diffusion.crnn.type = CRNNType.lstm
        elif args.diffusion_model == 1:
            config.model.diffusion.model_type = DiffusionModelType.unet
        else:
            raise ValueError(f"Unknown diffusion model type: {args.diffusion_model}")
        
        # Diffusion timesteps
        config.model.diffusion.timesteps = args.diffusion_time_steps
        
        # Handle legacy traversable_steps if exists
        if hasattr(config.model.diffusion, 'traversable_steps'):
            config.model.diffusion.traversable_steps = args.traversable_steps
    
    # Training configuration for poses
    config.loss.train_poses = args.train_poses
    if args.train_poses:
        config.loss.scale_waypoints = 20.0
        config.model.scale_waypoints = 20.0
    else:
        config.loss.scale_waypoints = 1.0
        config.model.scale_waypoints = 1.0
    
    # Traversability settings
    if args.use_traversability:
        config.loss.use_traversability = True
        if hasattr(config.model.diffusion, 'use_traversability'):
            config.model.diffusion.use_traversability = True
        if hasattr(config.model.cvae, 'estimate_traversability'):
            config.model.cvae.estimate_traversability = True
    
    #########################################
    # Training configurations
    #########################################
    config.training.only_model = args.only_load_model
    config.training.snapshot = args.snapshot if args.snapshot else None
    config.training.evaluation_freq = args.evaluation_freq
    config.training.train_time_steps = args.train_time_steps
    
    # Output directory for loss (debugging)
    config.loss.output_dir = args.debug_output
    config.loss.ablation_profile = args.loss_ablation
    
    # Training schedule based on training type
    if args.training_type == 0:
        config.training.max_epoch = 150
        config.training.lr_tm = 10
    elif args.training_type == 1:
        config.training.max_epoch = 100
        config.training.lr = 2e-5
        config.training.lr_tm = 30
        config.training.lr_min = 1e-8
    else:
        raise ValueError(f"Unknown training type: {args.training_type}")

    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.max_epoch_override > 0:
        config.training.max_epoch = args.max_epoch_override
    if args.stage1_epochs_override >= 0:
        config.training.stage1_epochs = args.stage1_epochs_override
    if 0.0 < args.train_fraction_override < 1.0:
        config.dataset.training_data_percentage = args.train_fraction_override
    if args.max_train_batches > 0:
        config.training.max_train_batches = args.max_train_batches
    if args.max_val_batches > 0:
        config.training.max_val_batches = args.max_val_batches
    
    #########################################
    # GPU/Device configurations
    #########################################
    # Device setting
    if args.device >= 0:
        device = f"cuda:{args.device}"
    elif args.device == -1:
        device = "cuda"
        print("------------------- CUDA: ", torch.cuda.is_available(), "-----------------------")
    else:
        device = "cpu"
    
    config.model.device = device
    config.training.gpus.device = device
    
    # Other GPU settings
    config.training.gpus.channels_last = args.channels_last
    config.training.gpus.local_rank = args.local_rank
    config.training.gpus.sync_bn = args.sync_bn
    config.training.gpus.no_ddp_bb = args.no_ddp_bb
    
    # Handle split_bn if it exists in the config
    if hasattr(config.training.gpus, 'split_bn'):
        config.training.gpus.split_bn = args.split_bn
    
    return config


def create_config_from_args_only():
    """Create a configuration using command-line arguments without YAML input."""
    args = get_args()
    
    # Create base configuration
    if args.generator_type == 0:
        config = create_diffusion_config()
    elif args.generator_type == 1:
        config = create_cvae_config()
    else:
        config = create_default_config()
    
    # Apply overrides
    config = apply_arg_overrides(config, args)
    
    return config

