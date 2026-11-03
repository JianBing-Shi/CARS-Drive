# BEVFormer preprocessing and training environment

Use `FRESH-Drive` for all preprocessing, experiments, tests and training.
Activate it with `conda activate FRESH-Drive`, or use `conda run -n FRESH-Drive python`.
Workspace interpreter: `/home/shi-jia-liang/miniforge3/envs/FRESH-Drive/bin/python`.

## Environment setup status

Installed package metadata verified on 2026-11-02:

- Python 3.10.22
- torch 1.13.1+cu117 / torchvision 0.14.1+cu117
- mmcv-full 1.7.0 / mmdet 2.28.2 / mmdet3d 1.0.0rc6
- lightning 2.0.9 / pytorch-lightning 2.0.9 / torchmetrics 1.2.1
- torch-geometric 2.3.1

The training imports have been checked in this local environment.
Package presence does not verify CUDA extensions or full training compatibility.
Keep PyTorch compatible with the BEVFormer compiled extensions.

For cached BEV training on the remote L40S image (PyTorch 2.1.0 / CUDA 11.8),
use the root `requirements.txt` and [remote setup guide](remote_training_environment.md).
That environment does not install or reuse the local BEVFormer compiled extensions.

Before full preprocessing, validate a real eight-camera frame: checkpoint
loading, CUDA deformable attention, finite BEV output and calibrated projection.

## Preprocessing files

Paths below are relative to the nuplan-v1.1 directory passed as `root`:

```
preprocessed_dataset/
  features/<recording>/<camera>/<own-image-name>.pt
  bev_features/<recording>/<CAM_F0-image-name>.pt
  events/<recording>/<event-type>/<anchor-timestamp>.pt
  mini-train.pt
  mini-val.pt
```

Manifests contain filenames relative to `events`. Each event references BEV
files relative to `root`. Tensors are saved on CPU. Preserve this directory
structure when moving files between machines.

## Training contract

The current NuplanEventDataset/NuplanDataModule/FRESHDrive interface loads cached
BEV events directly. The training DataModule disables preprocessing and does not
load the image encoders. Training needs events, BEVs and split manifests, not the
per-camera `features` cache.

The current defaults load [31,C,H,W] maps at 0.1-second intervals: 20 history
frames, an anchor and 10 future frames. At anchor time, only history and anchor
observations are available; later observations become available as replay advances.
Future expert poses and object annotations are supervision/replay data. Future
logged ego poses are expert demonstrations, not recorded planner output. Scenario
tags are not navigation routes.

Build BEV visual tokens in the training model. First train a trajectory model
using historical BEV, ego history/state and available conditions; then evaluate
refresh decisions by comparing continued old plans with replanned trajectories.
Token changes alone are not refresh labels. The current route condition has
no navigation geometry, and the preprocessing does not produce refresh labels.

The current event-stratified train/val split can share overlapping frames from
one recording. It is useful for pipeline checks; final evaluation should use
held-out recordings to measure generalization without shared windows.
