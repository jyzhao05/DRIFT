"""Configuration synchronization, validation, YAML loading, and command-line override utilities."""

import os
import yaml
from copy import deepcopy
from easydict import EasyDict as edict

from src.utils.config import (
    GeneratorType, Config, DiffusionConfigTemplate, CVAEConfigTemplate,
    create_base_config
)


class ConfigurationError(Exception):
    """Raised when configuration fields are missing or inconsistent."""
    pass


class ConfigValidator:
    """Validate dimensions, generator settings, and graph configuration."""
    
    @staticmethod
    def validate_dimensions(config):
        """Check trajectory lengths and perception output dimensions."""
        # Basic dimension consistency
        dataset_traj_len = config.dataset.traj_len
        cvae_waypoints = config.model.cvae.waypoints_num
        diffusion_waypoints = config.model.diffusion.waypoints_num
        
        if dataset_traj_len != cvae_waypoints:
            raise ConfigurationError(
                f"Trajectory length mismatch: dataset {dataset_traj_len} vs CVAE {cvae_waypoints}"
            )
        
        if dataset_traj_len != diffusion_waypoints:
            raise ConfigurationError(
                f"Trajectory length mismatch: dataset {dataset_traj_len} vs Diffusion {diffusion_waypoints}"
            )
        
        # Perception output dimensions
        perception_out = config.model.perception.output_dim
        fusion_out = config.model.perception.fusion_cfg.out_dim
        if perception_out != fusion_out:
            raise ConfigurationError(
                f"Perception output dimension mismatch: {perception_out} vs {fusion_out}"
            )
    
    @staticmethod
    def validate_generator_consistency(config):
        """Check generator selection and the required model configuration."""
        model_gen_type = config.model.generator_type
        loss_gen_type = config.loss.generator_type
        
        if model_gen_type != loss_gen_type:
            raise ConfigurationError(
                "Generator type mismatch between model and loss"
            )
        
        # Check required components exist
        if model_gen_type == GeneratorType.diffusion:
            if not hasattr(config.model, 'diffusion') or config.model.diffusion is None:
                raise ConfigurationError("Diffusion config missing for diffusion generator")
            if not hasattr(config.model, 'generator') or config.model.generator is None:
                raise ConfigurationError("Generator config missing for diffusion generator")
        elif model_gen_type == GeneratorType.cvae:
            if not hasattr(config.model, 'cvae') or config.model.cvae is None:
                raise ConfigurationError("CVAE config missing for CVAE generator")
    
    @staticmethod
    def validate_graph_consistency(config):
        """Check that data and perception use matching graph settings."""
        graph_cfg = config.dataset.graph_cfg
        perception_cfg = config.model.perception
        
        if graph_cfg.voxel_size != perception_cfg.voxel_size:
            raise ConfigurationError(
                f"Voxel size mismatch: dataset {graph_cfg.voxel_size} vs perception {perception_cfg.voxel_size}"
            )
        
        if graph_cfg.k_nn != perception_cfg.k_nn:
            raise ConfigurationError(
                f"K-NN mismatch: dataset {graph_cfg.k_nn} vs perception {perception_cfg.k_nn}"
            )
    
    @classmethod
    def validate_config(cls, config):
        """Validate required configuration fields and report actionable errors or warnings."""
        try:
            cls.validate_dimensions(config)
            cls.validate_generator_consistency(config)
            cls.validate_graph_consistency(config)
            return True
        except ConfigurationError as e:
            raise e
        except Exception as e:
            raise ConfigurationError(f"Unexpected validation error: {str(e)}")


class ConfigSynchronizer:
    """Keep dependent configuration values consistent across components."""
    
    @classmethod
    def synchronize_config(cls, config):
        """Synchronize base settings, feature dimensions, and graph parameters."""
        cls._sync_base_settings(config)
        
        cls._sync_dimensions_and_lengths(config)
        
        cls._sync_graph_and_perception(config)

        # cls._sync_loss_and_training(config)

        return config
    
        
    @staticmethod
    def _sync_base_settings(config):
        """Propagate device, precision, and generator settings to training and loss."""
        device = config.model.device
        mixed_precision = config.model.mixed_precision
        
        config.training.gpus.device = device
        config.training.gpus.mixed_precision = mixed_precision
    
        """Propagate device, precision, and generator settings to training and loss."""
        gen_type = config.model.generator_type
        config.loss.generator_type = gen_type
    
        config.loss.generator_type = config.model.generator_type
        config.training.device = config.model.device
            
    @staticmethod
    def _sync_dimensions_and_lengths(config):
        dataset_cfg = config.dataset
        model_cfg = config.model

        traj_len = dataset_cfg.traj_len
        if hasattr(model_cfg, 'cvae'): model_cfg.cvae.waypoints_num = traj_len
        model_cfg.diffusion.waypoints_num = traj_len
        config.inference.max_trajectory_length = traj_len

        if hasattr(model_cfg, 'perception') and hasattr(model_cfg, 'generator'):
            perception_cfg = model_cfg.perception
            generator_cfg = model_cfg.generator
            
            generator_cfg.cond_dim = perception_cfg.output_dim
            
            generator_cfg.attention_key_dim = perception_cfg.gnn_output_dim
            generator_cfg.attention_value_dim = perception_cfg.gnn_output_dim
            
            generator_cfg.attention_query_dim = generator_cfg.hidden_dim

            if hasattr(dataset_cfg, 'vel_num'):
                generator_cfg.vel_input_dim = dataset_cfg.vel_num
            generator_cfg.target_input_dim = 2
            # ===============================================
        
    @staticmethod
    def _sync_graph_and_perception(config):
        graph_cfg = config.dataset.graph_cfg
        perception_cfg = config.model.perception
        
        perception_cfg.voxel_size = graph_cfg.voxel_size
        perception_cfg.k_nn = graph_cfg.k_nn
        perception_cfg.max_nodes = graph_cfg.max_nodes
        
        perception_cfg.gnn_cfg.in_feats = graph_cfg.node_feature_dim    
    
    @staticmethod
    def sync_dimensions(config):
        """Synchronize trajectory and feature dimensions across model components."""
        traj_len = config.dataset.traj_len
        
        # Update CVAE waypoints
        config.model.cvae.waypoints_num = traj_len
        
        # Update diffusion waypoints
        config.model.diffusion.waypoints_num = traj_len
        
        # Update inference max trajectory length
        config.inference.max_trajectory_length = traj_len
        
        # Update perception input dimensions (perception_out + target_dim)
        perception_out = config.model.perception.output_dim
        target_dim = 2  # (x, y) target coordinates
        config.model.cvae.perception_in = perception_out + target_dim
        config.model.diffusion.perception_in = perception_out + target_dim

    @staticmethod
    def sync_scale_settings(config):
        """Synchronize trajectory scaling settings with the loss configuration."""
        scale = config.model.scale_waypoints
        config.loss.scale_waypoints = scale
    
class ConfigIO:
    """Read, write, and merge YAML configuration dictionaries."""
    
    @staticmethod
    def load_from_yaml(yaml_path):
        """Load configuration values from a YAML file and validate them."""
        if not os.path.exists(yaml_path):
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML file: {str(e)}")
        
        return yaml_config
    
    @staticmethod
    def save_to_yaml(config, yaml_path):
        """Serialize the current configuration to a YAML file."""
        try:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            
            with open(yaml_path, 'w') as f:
                yaml.dump(dict(config), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {yaml_path}: {str(e)}")
    
    @staticmethod
    def deep_merge(default_dict, override_dict):
        """Recursively merge nested dictionaries, preferring values from the update."""
        result = deepcopy(default_dict)
        
        def _deep_update(base, update):
            for key, value in update.items():
                if (key in base and 
                    isinstance(base[key], dict) and 
                    isinstance(value, dict)):
                    _deep_update(base[key], value)
                else:
                    base[key] = value
        
        _deep_update(result, override_dict)
        return result


class ConfigManager:
    """Create, load, update, and validate model configurations."""
    
    def __init__(self):
        self.validator = ConfigValidator()
        self.synchronizer = ConfigSynchronizer()
        self.io = ConfigIO()
    
    def create_config(self, generator_type=None, **overrides):
        """Create and synchronize a configuration for the selected generator."""
        # Start with base configuration
        config = create_base_config()
        
        # Apply generator type if specified
        if generator_type is not None:
            config.model.generator_type = generator_type
        
        # Apply any additional overrides
        if overrides:
            config = edict(self.io.deep_merge(dict(config), overrides))
        
        # Synchronize dependent values
        config = self.synchronizer.synchronize_config(config)
        
        return config
    
    def load_config(self, yaml_path, validate=True):
        """Load a YAML configuration and synchronize dependent fields."""
        # Load YAML content
        yaml_config = self.io.load_from_yaml(yaml_path)
        
        # Merge with base configuration
        base_config = create_base_config()
        merged_config = edict(self.io.deep_merge(dict(base_config), yaml_config))
        
        # Synchronize dependent values
        merged_config = self.synchronizer.synchronize_config(merged_config)
        
        # Validate if requested
        if validate:
            self.validator.validate_config(merged_config)
        
        return merged_config
    
    def save_config(self, config, yaml_path, validate=True):
        """Write the configuration to a YAML file."""
        if validate:
            self.validator.validate_config(config)
        
        self.io.save_to_yaml(config, yaml_path)
    
    def update_from_args(self, config, args):
        """Apply command-line overrides and synchronize the resulting configuration."""
        config = deepcopy(config)
        
        # Common argument mappings
        arg_mappings = {
            'batch_size': ('dataset', 'batch_size'),
            'lr': ('training', 'lr'),
            'max_epoch': ('training', 'max_epoch'),
            'device': ('model', 'device'),
            'generator_type': ('model', 'generator_type'),
            'output_dir': ('training', 'output_dir'),
            'debug': ('debug',),
            'seed': ('seed',),
        }
        
        # Apply argument updates
        for arg_name, config_path in arg_mappings.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    # Navigate to the config location
                    current = config
                    for key in config_path[:-1]:
                        current = current[key]
                    current[config_path[-1]] = value
        
        # Synchronize after updates
        config = self.synchronizer.synchronize_config(config)
        
        return config
    
    def get_preset_config(self, generator_type):
        """Build a preset configuration for the selected generator type."""
        if generator_type == GeneratorType.diffusion:
            config = deepcopy(DiffusionConfigTemplate)
            # Optimization for diffusion
            config.training.lr = 2e-5
            config.training.max_epoch = 200
            config.dataset.batch_size = 16
        elif generator_type == GeneratorType.cvae:
            config = deepcopy(CVAEConfigTemplate)
            # Optimization for CVAE
            config.training.lr = 1e-4
            config.training.max_epoch = 150
            config.dataset.batch_size = 32
        else:
            raise ValueError(f"Unsupported generator type: {generator_type}")
        
        # Synchronize and validate
        config = self.synchronizer.synchronize_config(config)
        self.validator.validate_config(config)
        
        return config


class ConfigPrinter:
    """Format concise and detailed configuration reports."""
    
    @staticmethod
    def print_summary(config):
        """Print the main experiment, model, data, and training settings."""
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
        
        if config.model.generator_type == GeneratorType.diffusion:
            print(f"Diffusion Timesteps: {config.model.diffusion.timesteps}")
            print(f"Sampling Timesteps: {config.model.diffusion.sampling_timesteps}")
            print(f"Schedule Type: {config.model.diffusion.schedule_type}")
        
        print(f"Perception Output Dim: {config.model.perception.output_dim}")
        print(f"Graph Max Nodes: {config.dataset.graph_cfg.max_nodes}")
        print(f"Voxel Size: {config.dataset.graph_cfg.voxel_size}")
        print("=" * 60)
    
    @staticmethod
    def print_detailed(config, max_depth=3):
        """Print every configuration section and its values."""
        def _print_dict(d, indent=0, depth=0):
            if depth > max_depth:
                return
            
            for key, value in d.items():
                spaces = "  " * indent
                if isinstance(value, dict):
                    print(f"{spaces}{key}:")
                    _print_dict(value, indent + 1, depth + 1)
                else:
                    print(f"{spaces}{key}: {value}")
        
        print("=" * 60)
        print("DRIFT Detailed Configuration")
        print("=" * 60)
        _print_dict(dict(config))
        print("=" * 60)


# Convenience functions for common operations
def create_default_config():
    """Create the default configuration used by the application."""
    manager = ConfigManager()
    return manager.create_config()


def create_diffusion_config():
    """Create the diffusion model configuration template."""
    manager = ConfigManager()
    return manager.get_preset_config(GeneratorType.diffusion)


def create_cvae_config():
    """Build the retained CVAE configuration template."""
    manager = ConfigManager()
    return manager.get_preset_config(GeneratorType.cvae)


def load_config_from_yaml(yaml_path):
    """Load a configuration through the shared configuration manager."""
    manager = ConfigManager()
    return manager.load_config(yaml_path)


def save_config_to_yaml(config, yaml_path):
    """Save a configuration through the shared configuration manager."""
    manager = ConfigManager()
    return manager.save_config(config, yaml_path)


def update_config_from_args(config, args):
    """Apply argument overrides through the shared configuration manager."""
    manager = ConfigManager()
    return manager.update_from_args(config, args)


def validate_config(config):
    """Validate required configuration fields and report actionable errors or warnings."""
    validator = ConfigValidator()
    return validator.validate_config(config)


def print_config_summary(config):
    """Print a concise summary of the active configuration."""
    printer = ConfigPrinter()
    return printer.print_summary(config)
