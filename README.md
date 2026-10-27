# DRIFT

Core implementation of DRIFT for graph-conditioned diffusion trajectory generation.
The model encodes a local point-cloud graph and generates waypoints conditioned on the environment, velocity, start point, and goal.

## Method

Local point clouds are converted into voxel-based neighborhood graphs. An environment encoder extracts node and scene features, while separate encoders represent the start and goal. At each diffusion step, sparse cross-attention retrieves nearby environment nodes for each noisy waypoint. A goal- and time-conditioned GRU (GT-GRU) combines this local context with the conditioning features to predict a clean trajectory. The prediction is converted to a noise estimate for the DDPM objective and reverse-diffusion updates.

## Code guide

| File | Purpose |
| --- | --- |
| `src/models/model.py` | `HNav`: model assembly, training forward pass, and sampling interface |
| `src/models/perception.py` | Environment, velocity, goal, and start-point encoding |
| `src/models/backbones/perception_gnn.py` | EdgeConv / GAT environment features |
| `src/models/generator_cmag.py` | Sparse local cross-attention and recurrent waypoint generation |
| `src/models/backbones/gt_gru.py` | Goal- and time-conditioned recurrent cell |
| `src/models/diffusion.py` | DDPM training objective and reverse-diffusion sampling |
| `src/loss.py`, `src/models/diff_hausdorf.py` | Trajectory matching and geometric auxiliary losses |
| `main.py`, `src/train.py` | Application entry point, optimization, validation, and checkpoints |
| `src/data_loader/` | Dataset access, batching, and sample normalization |
| `src/utils/integrated_processor.py`, `transform.py`, `voxel_graph.py` | Point-cloud processing, coordinate transforms, and graph construction |
| `config_factory.py`, `src/utils/config*.py`, `arguments.py` | Configuration definitions and command-line options |

Start with `HNav`, then follow `CMAGNNGenerator`, `GTGRUCell`, and `DiffusionModel`.
The recurrent generator uses `SinusoidalPosEmb` from `src/models/backbones/unet.py` for diffusion timestep embeddings.

## Environment and data

Dependencies are listed in `requirements.txt`. Use compatible PyTorch, PyG, and torch-cluster builds for your CPU or CUDA environment. The listed torch-cluster build targets PyTorch 2.0 / CUDA 11.8.

Launch training from the repository root:

```bash
python main.py train --data_root /path/to/dataset --device 0 --conditioning_mode dynamic --denoiser_cell gtgru
```

`--data_root` defaults to `./data`. The directory must contain `data.pkl` and `data_folder/`; see `TrainData.load_raw_data` and `DataDict` for the expected schema.

The model interfaces are `HNav.forward(batch, epoch=...)` and `HNav.sample(batch)`. Sampling expects graph data, velocity, goal, the trajectory field supplying the start point, and normalization parameters.
