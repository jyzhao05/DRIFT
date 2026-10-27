"""Factory helpers that create, synchronize, summarize, and validate DRIFT configuration objects."""

from src.utils.config import (
    create_base_config, 
    GeneratorType, 
    ScheduleMethods,
    DiffusionModelType,
    CRNNType
)


def sync_configuration_attributes(config):
    """Synchronize dependent configuration fields across dataset, model, loss, and training sections."""
    
    training_cfg = config.training
    
    if not hasattr(training_cfg, 'max_epoch') and hasattr(training_cfg, 'max_epochs'):
        training_cfg.max_epoch = training_cfg.max_epochs
    elif not hasattr(training_cfg, 'max_epoch'):
        training_cfg.max_epoch = 150
    
    if not hasattr(training_cfg, 'gpus'):
        training_cfg.gpus = training_cfg
    
    if hasattr(config.model, 'device'):
        training_cfg.gpus.device = config.model.device
        training_cfg.device = config.model.device
    
    if hasattr(config.model, 'mixed_precision'):
        training_cfg.gpus.mixed_precision = config.model.mixed_precision
    
    dataset_cfg = config.dataset
    
    if not hasattr(dataset_cfg, 'voxel_size') and hasattr(dataset_cfg, 'graph_cfg'):
        dataset_cfg.voxel_size = dataset_cfg.graph_cfg.voxel_size
        dataset_cfg.k_nn = dataset_cfg.graph_cfg.k_nn
        dataset_cfg.max_nodes = dataset_cfg.graph_cfg.max_nodes
    
    model_cfg = config.model
    
    if hasattr(model_cfg, 'perception'):
        perception_cfg = model_cfg.perception
        
        vel_dim = dataset_cfg.vel_num if hasattr(dataset_cfg, 'vel_num') else 5
        target_dim = 2  # x, y target
        imu_dim = dataset_cfg.imu_num if hasattr(dataset_cfg, 'imu_num') else 0
        
        if hasattr(perception_cfg, 'vel_cfg'):
            perception_cfg.vel_cfg.input_dim = vel_dim
        if hasattr(perception_cfg, 'target_cfg'):
            perception_cfg.target_cfg.input_dim = target_dim
            
        perception_in = perception_cfg.output_dim + vel_dim + target_dim + imu_dim
        
        if hasattr(model_cfg, 'cvae'):
            model_cfg.cvae.perception_in = perception_in
        if hasattr(model_cfg, 'diffusion'):
            model_cfg.diffusion.perception_in = perception_in
    
    traj_len = dataset_cfg.traj_len if hasattr(dataset_cfg, 'traj_len') else 50
    
    if hasattr(model_cfg, 'cvae'):
        model_cfg.cvae.waypoints_num = traj_len
    if hasattr(model_cfg, 'diffusion'):
        model_cfg.diffusion.waypoints_num = traj_len
    if hasattr(config, 'inference'):
        config.inference.max_trajectory_length = traj_len
    
    generator_type = model_cfg.generator_type
    if hasattr(config, 'loss'):
        config.loss.generator_type = generator_type
    if hasattr(model_cfg, 'generator'):
        if hasattr(dataset_cfg, 'vel_num'):
            model_cfg.generator.vel_input_dim = dataset_cfg.vel_num
        
        model_cfg.generator.target_input_dim = 2
    # =================================================
    if hasattr(config, 'loss') and not hasattr(config.loss, 'output_dir'):
        config.loss.output_dir = training_cfg.output_dir
    
    required_training_attrs = {
        'name': 'DRIFT_Training',
        'evaluation_freq': 5,
        'save_freq': 1, 
        'log_freq': 100,
        'early_stopping': True,
        'patience': 3,
        'min_delta': 1e-4,
        'optimizer': 'adam',
        'lr': 2e-5,
        'weight_decay': 1e-5,
        'grad_clip_norm': 1.0,
        'scheduler': 'cosine',
        'lr_tm': 31,
        'lr_min': 1e-7
    }
    
    for attr, default_value in required_training_attrs.items():
        if not hasattr(training_cfg, attr):
            setattr(training_cfg, attr, default_value)
    
    required_dataset_attrs = {
        # 'num_workers': 0,
        'shuffle': True,
        'distributed': False
    }
    
    for attr, default_value in required_dataset_attrs.items():
        if not hasattr(dataset_cfg, attr):
            setattr(dataset_cfg, attr, default_value)

    if not hasattr(training_cfg, 'adaptive_training'):
        training_cfg.adaptive_training = True
    
    if not hasattr(training_cfg, 'aux_loss_warmup_epochs'):
        training_cfg.aux_loss_warmup_epochs = 5

    
    return config


def create_drift_config(
    dataset_name="hnav",
    batch_size=100,
    traj_len=50,
    lr=2e-5,
    max_epochs=50,
    device="cuda",
    mixed_precision=False,
    generator_type=GeneratorType.diffusion,
    root_path=None,
    **kwargs
):
    """Create the default DRIFT configuration and apply requested overrides."""
    config = create_base_config()
    
    config.dataset.name = dataset_name
    config.dataset.batch_size = batch_size
    config.dataset.traj_len = traj_len
    if root_path:
        config.dataset.root = root_path
    
    config.model.generator_type = generator_type
    config.model.device = device
    config.model.mixed_precision = mixed_precision
    
    config.training.lr = lr
    config.training.max_epochs = max_epochs
    config.training.max_epoch = max_epochs
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            if hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.dataset, key):
                setattr(config.dataset, key, value)
    
    config = sync_configuration_attributes(config)
    
    if config.model.generator_type == GeneratorType.s2tnet:
        print("INFO: Configuring model for S2TNet.")
        config.model.name = 's2tnet'
        
        s2tnet_cfg = config.model.s2tnet_cfg = {}
        
        s2tnet_cfg.d_model = 64
        s2tnet_cfg.nhead = 8
        s2tnet_cfg.encoder_num_layers = 4
        s2tnet_cfg.decoder_num_layers = 4
        s2tnet_cfg.feedforward_ratio = 1
        s2tnet_cfg.drop_rate = 0.1
        
        s2tnet_cfg.in_chans = 2 
        s2tnet_cfg.output_chans = 2
        
        s2tnet_cfg.hist_len = 10 
        
        s2tnet_cfg.use_spatial = False 
        s2tnet_cfg.spatial_depth = 4
    return config


def print_config_summary(config):
    """Print a concise summary of the active configuration."""
    print("=" * 60)
    print("DRIFT Configuration Summary")
    print("=" * 60)
    
    print(f"Generator Type: {config.model.generator_type}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Batch Size: {config.dataset.batch_size}")
    print(f"Trajectory Length: {config.dataset.traj_len}")
    print(f"Learning Rate: {config.training.lr}")
    print(f"Max Epochs: {config.training.max_epoch}")
    print(f"Device: {config.model.device}")
    print(f"Mixed Precision: {config.model.mixed_precision}")
    
    if hasattr(config.model, 'diffusion'):
        print(f"Diffusion Timesteps: {config.model.diffusion.timesteps}")
        print(f"Sampling Timesteps: {config.model.diffusion.sampling_timesteps}")
        print(f"Schedule Type: {config.model.diffusion.schedule_type}")
    
    if hasattr(config.model, 'perception'):
        print(f"Perception Output Dim: {config.model.perception.output_dim}")
        
    if hasattr(config.dataset, 'graph_cfg'):
        print(f"Graph Max Nodes: {config.dataset.graph_cfg.max_nodes}")
        print(f"Voxel Size: {config.dataset.graph_cfg.voxel_size}")
    
    print("=" * 60)


def validate_config(config):
    """Validate required configuration fields and report actionable errors or warnings."""
    errors = []
    warnings = []
    
    required_fields = [
        ('training.max_epoch', 'training.max_epoch'),
        ('training.lr', 'training.lr'),
        ('dataset.batch_size', 'dataset.batch_size'),
        ('dataset.traj_len', 'dataset.traj_len'),
        ('model.generator_type', 'model.generator_type'),
        ('model.device', 'model.device')
    ]
    
    for field_path, display_name in required_fields:
        try:
            obj = config
            for attr in field_path.split('.'):
                obj = getattr(obj, attr)
        except AttributeError:
            errors.append(f"Missing required configuration field: {display_name}")
    
    if hasattr(config.dataset, 'root') and config.dataset.root:
        import os
        if not os.path.exists(config.dataset.root):
            warnings.append(f"Dataset path does not exist: {config.dataset.root}")
    
    if hasattr(config.model, 'device'):
        import torch
        if config.model.device.startswith('cuda') and not torch.cuda.is_available():
            warnings.append("CUDA requested but unavailable; falling back to CPU")
    
    if errors:
        print(" Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print(" Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print(" Configuration validation passed")
    return True
