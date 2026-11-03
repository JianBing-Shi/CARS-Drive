# 🚗 DriveWorld-VLA: Unified Latent-Space World Modeling with Vision–Language–Action for Autonomous Driving

Feiyang Jia*, Lin Liu*, Ziying Song, Caiyan Jia†, Hangjun Ye, Xiaoshuai Hao† and Long Chen⊥
[[📄 Paper (arXiv)]]()

We present **DriveWorld-VLA**, a tightly coupled framework where a world model serves as the reasoning engine bridging action and prospective imagination.

<div align="center">
<img src="assets/main.jpg" width="1000">
</div>

---

## News

* **` Feb. 01th, 2026`:** We released our paper on [Arxiv](https://arxiv.org/abs/2506.08052). NavSim Code/Models are released!

## Updates

- [X] Release Paper
- [X] Release NavSim Models and Training/Evaluation Framework
- [ ] Release NuScenes Models and Training/Evaluation Framework

## 📊 1. Results & Checkpoints

|          Method          |       NC       |      DAC      |       EP       |      TTC      |    Comfort    |      PDMS      |  Training Time  |   GPU Memory   |                                            Checkpoint                                            |
| :----------------------: | :------------: | :------------: | :------------: | :------------: | :-----------: | :------------: | :--------------: | :-------------: | :----------------------------------------------------------------------------------------------: |
| **DriveWorld-VLA** | **99.1** | **98.2** | **81.9** | **96.1** | **100** | **91.3** | **24 hrs** | **80 GB** | [📥 Download](https://pan.baidu.com/s/1DuGLsuK6pAfT7loUb8pm3g?pwd=dvla) |

*Training conducted on 8 NVIDIA H20 GPUs.*

> **Legend**
> • NC: No Collision
> • DAC: Drivable Area Compliance
> • EP: Ego Progress
> • TTC: Time to Collision
> • Comfort: Comfort
> • PDMS: Predictive Driver Model Score

---

## 📦 2. Dataset & File Structure

```bash
root/
├── ckpts/
│   └── resnet34.pth
├── internvl_chat/
│   └── Internvlm checkpoint
├── dataset/
│   ├── maps/
│   ├── navsim_logs/
│   │   ├── test/
│   │   └── trainval/
│   ├── sensor_blobs/
│   │   ├── test/
│   │   └── trainval/
└── exp/
    └── metric_cache/
```

### 📁 a. Download NAVSIM Dataset

To obtain the [navsim dataset](https://github.com/autonomousvision/navsim/tree/main):

```bash
bash download/download_maps.sh
bash download/download_navtrain.sh
bash download/download_test.sh
```

### 📁 b. Prepare the Internvl checkpoint

```bash
refer to https://github.com/xiaomi-research/recogdrive to download checkpoint
```

### 📁 c. Precompute Metric Cache

```bash
bash scripts/evaluation/run_metric_caching.sh
```

---

## ⚙️ 3. Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate Driveworld-vla
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/motional/nuplan-devkit.git@nuplan-devkit-v1.2#egg=nuplan-devkit
```

Add environment variables to `~/.bashrc` (modify paths as needed):

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

---

## 🚀 4. Training & Evaluation

Update paths in:

```
——navsim/agents/WoTE/configs/default_stage1.py
——navsim/agents/WoTE/configs/default_stage2.py
——navsim/agents/WoTE/configs/default_stage3.py
```

Then launch training stage 1:

```bash
bash scripts/training/run_ImagineWorld_stage1.sh # stage1_training
```

Then launch training stage 2:

```bash
bash scripts/training/run_ImagineWorld_stage2.sh # stage2_training
```

Then launch training stage 3:

```bash
bash scripts/training/run_ImagineWorld_stage3.sh # stage3_training
```

Evaluation (stage 3):

```bash
bash scripts/evaluation/eval_driveworld_vla.sh
```

---

## 🔍 5.Qualitative Results on Navsim

<div align="center">
  <img src="assets/nav_vis.jpg" width="1000">
</div>
<p align="left">
Visualization examples of navsim dataset. Top label: source of trajectory.

## 🔍 6.Qualitative Results on Nuscenes

<div align="center">
  <img src="assets/nus_vis.jpg" width="1000">
</div>
<p align="left">
Visualization examples of nuScenes validation dataset. Top label: source of trajectory.

## Acknowledgement

DriveWorld-VLA is greatly inspired by the following outstanding contributions to the open-source community: [NAVSIM](https://github.com/autonomousvision/navsim), [DPPO](https://github.com/irom-princeton/dppo), [LightningDiT](https://github.com/hustvl/LightningDiT), [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), [WOTE](https://github.com/liyingyanUCAS/WoTE).

## Citation

If you find DriveWorld-VLA is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@article{
update soon
}
```
