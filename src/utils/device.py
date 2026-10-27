"""Device management helpers for CPU/GPU placement, random seeds, and tensor cleanup."""

import torch
from typing import Union, Any, Dict, List, Tuple
import numpy as np

try:
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    Data = None
    Batch = None


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """Resolve a requested device and fall back safely when CUDA is unavailable."""
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    if isinstance(device, str):
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                if ':' in device:
                    gpu_id = int(device.split(':')[1])
                    if gpu_id >= torch.cuda.device_count():
                        print(f"Warning: CUDA device {gpu_id} not available, falling back to cuda:0")
                        return torch.device('cuda:0')
                return torch.device(device)
            else:
                print("Warning: CUDA not available, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    if isinstance(device, torch.device):
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return torch.device('cpu')
        return device
    
    raise ValueError(f"Invalid device specification: {device}")


def to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors and graph data to the requested device."""
    if torch.is_tensor(x):
        return x.to(device)
    
    elif HAS_TORCH_GEOMETRIC and isinstance(x, (Data, Batch)):
        return x.to(device)
    
    elif hasattr(x, 'to') and hasattr(x, '__class__') and 'Data' in str(x.__class__):
        try:
            return x.to(device)
        except Exception as e:
            print(f"Warning: Failed to move {type(x)} to device: {e}")
            return _move_graph_like_object(x, device)
    
    elif isinstance(x, dict):
        return {key: to_device(value, device) for key, value in x.items()}
    
    elif isinstance(x, (list, tuple)):
        moved_items = [to_device(item, device) for item in x]
        return type(x)(moved_items)
    
    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
        return tensor.to(device)
    
    else:
        return x


def _move_graph_like_object(obj, device):
    """Move tensor attributes of a graph-like object to the requested device."""
    try:
        attrs = dir(obj)
        
        for attr_name in attrs:
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
                
                if torch.is_tensor(attr_value):
                    setattr(obj, attr_name, attr_value.to(device))
                
                elif isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                    if torch.is_tensor(attr_value[0]):
                        moved_list = [item.to(device) for item in attr_value]
                        setattr(obj, attr_name, moved_list)
                
            except (AttributeError, TypeError):
                continue
        
        return obj
    
    except Exception as e:
        print(f"Warning: Failed to manually move graph object: {e}")
        return obj


def release_cuda(x: Any) -> Any:
    """Detach tensors and convert nested values to CPU or NumPy representations."""
    if torch.is_tensor(x):
        x_cpu = x.detach().cpu()
        
        if x_cpu.numel() == 1:
            return x_cpu.item()
        else:
            return x_cpu.numpy()
    
    elif HAS_TORCH_GEOMETRIC and isinstance(x, (Data, Batch)):
        return x.to('cpu')
    
    elif isinstance(x, dict):
        return {key: release_cuda(value) for key, value in x.items()}
    
    elif isinstance(x, (list, tuple)):
        released_items = [release_cuda(item) for item in x]
        return type(x)(released_items)
    
    else:
        return x


def get_memory_info() -> Dict[str, float]:
    """Report CUDA memory usage in GiB, or CUDA availability."""
    if not torch.cuda.is_available():
        return {"message": "CUDA not available"}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    cached = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    
    return {
        "allocated_gb": allocated,
        "cached_gb": cached,
        "max_allocated_gb": max_allocated,
        "device_count": torch.cuda.device_count()
    }


def clear_cuda_cache():
    """Release cached CUDA memory when a CUDA device is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_device_and_seed(device: Union[str, torch.device, None] = None, seed: int = 42) -> torch.device:
    """Configure the runtime device and initialize reproducible random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    target_device = get_device(device)
    
    if target_device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return target_device


def check_tensor_devices(*tensors, expected_device=None):
    """Inspect tensor and graph devices and report consistency."""
    results = {}
    devices = set()
    
    for i, tensor in enumerate(tensors):
        if torch.is_tensor(tensor):
            device = tensor.device
            devices.add(device)
            results[f'tensor_{i}'] = {
                'device': device,
                'dtype': tensor.dtype,
                'shape': tensor.shape
            }
        elif HAS_TORCH_GEOMETRIC and isinstance(tensor, (Data, Batch)):
            graph_devices = set()
            for attr_name in ['x', 'edge_index', 'edge_attr', 'pos', 'batch']:
                if hasattr(tensor, attr_name):
                    attr_value = getattr(tensor, attr_name)
                    if torch.is_tensor(attr_value):
                        graph_devices.add(attr_value.device)
            
            results[f'graph_{i}'] = {
                'type': type(tensor).__name__,
                'devices': list(graph_devices)
            }
            devices.update(graph_devices)
    
    results['summary'] = {
        'all_devices': list(devices),
        'consistent': len(devices) <= 1,
        'expected_device': expected_device,
        'matches_expected': all(d == expected_device for d in devices) if expected_device else None
    }
    
    return results
