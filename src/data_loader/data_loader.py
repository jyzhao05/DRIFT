import os
import random
import warnings
from typing import Dict, List, Any
import torch
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from src.data_loader.dataset import TrainData

def reset_seed_worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

def drift_online_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate valid samples into batched graph, tensor, and metadata fields."""
    batch = [s for s in batch if s is not None]
    if not batch:
        return {}

    collated_batch = {}

    collated_batch['local_graph'] = Batch.from_data_list([s['local_graph'] for s in batch])

    tensor_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor)]

    for key in tensor_keys:
        collated_batch[key] = torch.stack([s[key] for s in batch], dim=0)

    if 'meta' in batch[0]:
        meta_keys = batch[0]['meta'].keys()
        collated_batch['meta'] = {key: [s['meta'][key] for s in batch] for key in meta_keys}

    return collated_batch

def get_data_loader(cfg, split='train'):
    dataset = TrainData(cfg=cfg, split=split)
    distributed = getattr(cfg.dataset, 'distributed', False)
    sampler = DistributedSampler(dataset) if distributed else None

    batch_size = getattr(cfg.dataset, 'batch_size', 16)

    num_workers = getattr(cfg.dataset, 'num_workers', 0)

    shuffle = (getattr(cfg.dataset, 'shuffle', True) and (sampler is None)) and (split == 'train')

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=drift_online_collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return data_loader

def train_data_loader(cfg):
    return get_data_loader(cfg=cfg, split='train')

def evaluation_data_loader(cfg):
    return get_data_loader(cfg=cfg, split='val')
