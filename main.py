import sys
import os
import traceback
import torch


torch.set_anomaly_enabled(True)
import torch.multiprocessing as mp
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
from src.utils.arguments import get_configuration
from src.utils.config_utils import (
    ConfigManager,
    ConfigPrinter,
    ConfigurationError
)
from config_factory import create_drift_config, print_config_summary, validate_config
from src.train import Trainer

from src.utils.device import set_device_and_seed
from src.utils.config import GeneratorType

class DRIFTApp:
    """Coordinate configuration, device setup, and trainer creation."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = None
        self.trainer = None
    
    def setup_configuration(self):
        """Load and validate the application configuration."""
        try:
            self.config = get_configuration()
            print(" Configuration loaded successfully")
            
            self.config_manager.validator.validate_config(self.config)
            print(" Configuration validation passed")
            
        except ConfigurationError as e:
            print(f" Configuration error: {e}")
            raise
        except Exception as e:
            print(f" Failed to load configuration: {e}")
            traceback.print_exc()
            raise

    def setup_device_and_seed(self):
        """Select the compute device and initialize random seeds."""
        try:
            device = set_device_and_seed(
                device=self.config.model.device,
                seed=getattr(self.config, 'seed', 42)
            )
            print(f"Device: {device}")
            return device
        except Exception as e:
            print(f" Failed to configure device: {e}")
            raise
    
    def setup_trainer(self):
        """Construct the trainer using a supported configuration argument."""
        try:
            trainer_args = [
                {'config': self.config},
                {'cfg': self.config},
            ]
            
            for kwargs in trainer_args:
                try:
                    self.trainer = Trainer(**kwargs)
                    print(" Trainer created successfully")
                    return
                except TypeError as e:
                    continue
            
            raise RuntimeError("Unable to create trainer; check the Trainer constructor")
            
        except Exception as e:
            print(f" Failed to create trainer: {e}")
            traceback.print_exc()
            raise
    
    def print_startup_info(self):
        """Display the experiment configuration before execution."""
        print(" DRIFT - Cross-Modal Attention Graph Neural Network")
        print("=" * 60)
        
        if hasattr(ConfigPrinter, 'print_summary'):
            ConfigPrinter.print_summary(self.config)
        else:
            print(f"Experiment: {getattr(self.config, 'experiment_name', 'Unknown')}")
            print(f"Generator type: {self.config.model.generator_type}")
            print(f"Batch size: {self.config.dataset.batch_size}")
            print(f"Max epochs: {self.config.training.max_epoch}")
            print("-" * 40)


def main():
    """Initialize the application and run training with cleanup on exit."""
    app = DRIFTApp()
    
    try:
        app.setup_configuration()
        
        
        app.print_startup_info()
        
        app.setup_device_and_seed()
        
        app.setup_trainer()
        if app.trainer and app.trainer.model and app.trainer.device:
            app.trainer.model.to(app.trainer.device)
            print(f" Model synchronized to device: {app.trainer.device}")
        # if getattr(app.config, 'debug', False):
        #     torch.autograd.set_detect_anomaly(True)
        
        print(" Starting training...")
        app.trainer.run()
        print(" Training complete")
    
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n Training failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        try:
            is_anom_fn = getattr(torch.autograd, "is_anomaly_enabled", None)
            set_det_fn = getattr(torch.autograd, "set_detect_anomaly", None)
            if callable(set_det_fn):
                if callable(is_anom_fn):
                    if is_anom_fn():
                        set_det_fn(False)
                else:
                    set_det_fn(False)
        except Exception:
            pass
    return 0


def test_mode():
    """Load the configured checkpoint and run the trainer test interface."""
    app = DRIFTApp()
    
    try:
        print(" Test mode")
        
        app.setup_configuration()
        app.setup_device_and_seed()
        app.setup_trainer()
        
        if app.config.training.snapshot:
            checkpoint_path = app.config.training.snapshot
        else:
            checkpoint_path = os.path.join(
                app.config.training.output_dir, 
                "checkpoints", 
                "best.pth"
            )
        
        if not os.path.exists(checkpoint_path):
            print(f" Checkpoint file not found: {checkpoint_path}")
            return 1
        
        print(f" Load checkpoint: {checkpoint_path}")
        
        if hasattr(app.trainer, 'test_and_visualize'):
            results = app.trainer.test_and_visualize(
                ckpt_path=checkpoint_path,
                n_samples=getattr(app.config.inference, 'num_samples', 100)
            )
        elif hasattr(app.trainer, 'test'):
            results = app.trainer.test(checkpoint_path)
        else:
            print(" Trainer does not support testing")
            return 1
        
        print(f" Test results: {results}")
        print(" Test complete")
        return 0
    
    except Exception as e:
        print(f" Test failed: {e}")
        traceback.print_exc()
        return 1


def resume_training():
    """Restore a checkpoint and continue the training loop."""
    app = DRIFTApp()
    
    try:
        print("Resuming training")
        
        app.setup_configuration()
        
        latest_checkpoint = os.path.join(
            app.config.training.output_dir, 
            "checkpoints", 
            "latest.pth"
        )
        
        if not os.path.exists(latest_checkpoint):
            print(f" Checkpoint not found: {latest_checkpoint}")
            print("Searching for another checkpoint...")
            
            checkpoint_dir = os.path.join(app.config.training.output_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoints:
                    latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
                    print(f"Using checkpoint: {latest_checkpoint}")
                else:
                    print(" No checkpoint found")
                    return 1
            else:
                print(" Checkpoint directory does not exist")
                return 1
        
        app.config.training.snapshot = latest_checkpoint
        app.config.training.resume_from_checkpoint = True
        
        print(f" Resuming from checkpoint: {latest_checkpoint}")
        
        app.config_manager.validator.validate_config(app.config)
        
        app.print_startup_info()
        app.setup_device_and_seed()
        app.setup_trainer()
        
        print(" Resume training...")
        app.trainer.run()
        print(" Training complete")
        return 0
    
    except Exception as e:
        print(f" Resume failed: {e}")
        traceback.print_exc()
        return 1


def inference_mode():
    """Run inference when the trainer exposes an inference interface."""
    app = DRIFTApp()
    
    try:
        print(" Inference mode")
        
        app.setup_configuration()
        app.setup_device_and_seed()
        app.setup_trainer()
        
        if hasattr(app.trainer, 'inference'):
            results = app.trainer.inference()
            print(f" Inference results: {results}")
        else:
            print(" Trainer does not support inference")
            return 1
        
        print(" Inference complete")
        return 0
    
    except Exception as e:
        print(f" Inference failed: {e}")
        traceback.print_exc()
        return 1


def print_help():
    """Display available application modes and command examples."""
    print(" DRIFT - Cross-Modal Attention Graph Neural Network")
    print("=" * 60)
    print("Usage:")
    print("  python main.py [mode]")
    print()
    print("Available modes:")
    print("  train     - Training mode (default)")
    print("  test      - Test mode")
    print("  resume    - Resume training")
    print("  inference - Inference mode")
    print("  help      - Show this help message")
    print()
    print("Examples:")
    print("  python main.py                    # default training")
    print("  python main.py train              # training mode")
    print("  python main.py test               # test mode")
    print("  python main.py resume             # resume training")
    print("  python main.py --config_file my.yaml  # use a configuration file")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        mode = sys.argv[1].lower()
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    else:
        mode = 'train'
    
    mode_functions = {
        'train': main,
        'training': main,
        'test': test_mode,
        'resume': resume_training,
        'inference': inference_mode,
        'infer': inference_mode,
        'help': print_help,
        '--help': print_help,
        '-h': print_help,
    }
    
    if mode in mode_functions:
        try:
            exit_code = mode_functions[mode]()
            sys.exit(exit_code if exit_code is not None else 0)
        except Exception as e:
            print(f" Application failed: {e}")
            sys.exit(1)
    else:
        print(f" Unknown mode: {mode}")
        print_help()
        sys.exit(1)
