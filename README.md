# DRIFT

DRIFT is a diffusion-based framework for local robot trajectory generation. Given a point cloud, robot velocity, a start point, and a goal, it generates a sequence of 2D waypoints in the robot's local coordinate frame.

## Overview

DRIFT combines graph-based environment perception with an autoregressive diffusion denoiser:

- **Graph perception** encodes a voxelized point cloud using a graph neural network.
- **Sparse cross-attention** retrieves nearby environment features for each waypoint as the trajectory is denoised.
- **GT-GRU** incorporates local environment context, goal information, and diffusion time into recurrent waypoint prediction.

Training combines a DDPM noise-prediction objective with auxiliary trajectory losses. At inference time, reverse diffusion generates a trajectory from Gaussian noise.

## Requirements

The main dependencies are PyTorch, PyTorch Geometric, torch-cluster, diffusers, and FAISS. Package versions are listed in [requirements.txt](requirements.txt).

Install PyTorch and the corresponding PyG and torch-cluster builds for your environment. The torch-cluster version listed in `requirements.txt` uses PyTorch 2.0 and CUDA 11.8.

## Data

Organize the dataset as follows:

```text
data/
├── data.pkl
└── data_folder/
    └── ...
```

`data.pkl` indexes the sequence files in `data_folder/`. The loader reads point clouds, poses, velocities, trajectories, and goals, then constructs local graphs and normalizes trajectory coordinates.

## Training

Run from the repository root:

```bash
python main.py train \
    --data_root ./data \
    --device 0 \
    --conditioning_mode dynamic \
    --denoiser_cell gtgru
```

Common options:

| Argument | Description |
| --- | --- |
| `--data_root` | Dataset directory |
| `--device` | GPU index |
| `--batch_size` | Training batch size |
| `--output_dir` | Directory for training outputs |
| `--config_file` | YAML configuration file |
| `--snapshot` | Checkpoint to load |
| `--seed` | Random seed |

## Project structure

```text
.
├── main.py                 # Training entry point
├── config_factory.py       # Configuration helpers
├── requirements.txt
└── src/
    ├── data_loader/        # Dataset loading and batching
    ├── models/
    │   ├── backbones/      # Graph encoder and recurrent cell
    │   ├── perception.py   # Environment and state encoding
    │   ├── generator_cmag.py
    │   ├── diffusion.py    # DDPM training and sampling
    │   └── model.py        # Model assembly
    ├── loss.py             # Auxiliary trajectory losses
    ├── train.py            # Training and validation
    └── utils/              # Preprocessing, configuration, visualization
```
