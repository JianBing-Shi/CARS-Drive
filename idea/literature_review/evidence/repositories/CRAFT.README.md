<div align="center">
  <img src="./assets/craft_name.png" alt="CRAFT" width="92%" style="vertical-align: bottom;">
  <h2>Counterfactual-to-Interactive Reinforcement Fine-Tuning for Driving Policies</h2>
  <p><b>A closed-loop post-training framework for autonomous driving policies.</b></p>
  <p>
    <a href="https://currychen77.github.io/CRAFT">Project Page</a> ·
    <a href="https://arxiv.org/abs/2605.04470">arXiv</a>
  </p>
</div>
<div align="center">
  <a href="https://currychen77.github.io/"><b>Keyu Chen</b></a><sup>1</sup> ·
  <a href="https://scholar.google.com/citations?user=Jo7TvUMAAAAJ&hl=zh-CN"><b>Nanfei Ye</b></a><sup>2</sup> ·
  <a href="https://wangyida.github.io/"><b>Yida Wang</b></a><sup>2</sup> ·
  <a href="https://scholar.google.com/citations?user=yd-sMoQAAAAJ&hl=zh-CN&oi=ao"><b>Wenchao Sun</b></a><sup>1</sup> ·
  <a href="https://scholar.google.com.hk/citations?user=n-iGIBkAAAAJ&hl=zh-CN"><b>Danqi Zhao</b></a><sup>1</sup> ·
  <a href="https://github.com/AutoChengh"><b>Hao Cheng</b></a><sup>1</sup> ·
  <a href="http://www.svm.tsinghua.edu.cn/essay/80/1835.html"><b>Sifa Zheng</b></a><sup>1</sup>
  <br>
  <sup>1</sup>School of Vehicle and Mobility, Tsinghua University ·
  <sup>2</sup>Li Auto Inc
</div>

<br>

<div align="center">
  <img src="./assets/intro.png" alt="CRAFT overview" width="100%">
</div>

:dizzy: *CRAFT improves driving policies by combining dense counterfactual proxy supervision with residual correction from true closed-loop interaction.*

If you find this repository useful, please consider giving it a star.

## :sparkles: News

- **`2026-09-25`** CRAFT **was accepted at NeurIPS 2026**!
- **`2026-05-07`** Our paper is available on [arXiv](https://arxiv.org/abs/2605.04470)📄!
- **`2026-05-06`** Explore our project page, now live [here](https://currychen77.github.io/CRAFT)🔗!

## :calendar: TODO List

- [x] Training & Eval Pipeline
- [x] Fine-Tuning Support (SparseDriveV2 / HiP-AD / MindDrive)
- [x] Visualization Utilities
- [ ] Official fine-tuned checkpoints (Post-Acceptance Release)

## :clipboard: Outline

- [:sparkles: News](#sparkles-news)
- [:calendar: TODO List](#calendar-todo-list)
- [:clipboard: Outline](#clipboard-outline)
- [:hammer: Setup](#hammer-setup)
- [:gift: Data and Checkpoints](#gift-data-and-checkpoints)
- [:fire: Usage](#fire-usage)
  - [Training / Evaluation](#training--evaluation)
  - [Visualization](#visualization)
- [:bookmark: Citation](#bookmark-citation)
- [:heart: Acknowledgements](#heart-acknowledgements)

## :hammer: Setup

**Recommended system: Ubuntu 20.04 or 22.04**

**Step 1: Download CARLA 0.9.16**

```bash
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz
tar -xvf CARLA_0.9.16.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.16.tar.gz
cd .. && bash ImportAssets.sh
```

Update your system's PYTHONPATH with the following paths:
```bash
export CARLA_ROOT=YOUR_CARLA_PATH
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
```

**Step 2: Set up the conda environment and install CARLA**

```bash
conda create -n Craft python=3.10
conda activate Craft
## install CARLA
pip install YOUR_CARLA_PATH/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
```

**Step 3: Clone this git repo in an appropriate folder**

```bash
git clone git@github.com:CurryChen77/CraftPolicy.git
cd CraftPolicy
```

**Step 4: Install packages**

```bash
pip install -r requirements.txt
pip install -e .
```

**Step 5: Install E2E packages**

Follow the [E2E installation guide](./rift/ego/b2d/README.md) to install E2E-related packages.

## :gift: Data and Checkpoints

* **CARLA Map Data**

[Doc of HD Map](docs/map_api.md)

Please ensure the HD map and speed limit data are downloaded before starting fine-tuning.

|       Name        |                         Google Drive                         | Approx. Size |         Storage Place         |
| :---------------: | :----------------------------------------------------------: | :----------: | :---------------------------: |
|    HD Map Data    | [Link](https://drive.google.com/drive/folders/1CWdcO2Gd-Qd9cF-bBTC1oK6jVZtFGDj3?usp=drive_link) |    714 MB    |   [Folder](./data/map_data)   |
| Speed Limits Data | [Link](https://drive.google.com/drive/folders/142JqYn4h8KDgHGxR1JOogHGRbynyLlBg?usp=drive_link) |    79 MB     | [Folder](./data/speed_limits) |

* **AV Checkpoints**

|   AV Name    |                         Google Drive                         | Approx. Size |                Storage Place                 |
| :----------: | :----------------------------------------------------------: | :----------: | :------------------------------------------: |
| PlanT-V2 | [Link](https://drive.google.com/drive/folders/1E-n2nffVOMHleKFCokdHa93MllEv1QFf?usp=drive_link) |    732 MB    | [Folder](./rift/ego/model_ckpt/PlanT_medium) |

For E2E AV checkpoints, see the [E2E installation guide](./rift/ego/b2d/README.md).

* **CBV Checkpoints**

| CBV Name |                         Google Drive                         | Approx. Size |                 Storage Place                  |
| :------: | :----------------------------------------------------------: | :----------: | :--------------------------------------------: |
|  Pluto   | [Link](https://drive.google.com/drive/folders/18bXqs8nT1gr3-xnTUlxGN2jxnQV5BqFQ?usp=drive_link) |   51.4 MB    | [Folder](./rift/cbv/planning/model_ckpt/pluto) |

## :fire: Usage

The main workflows are launched from `scripts/`. For most experiments, start with the two high-level launchers below:

- `scripts/run_train_and_eval.sh`: collect rollouts, train one fine-tuner, then evaluate it.
- `scripts/run_multi_finetuners.sh`: run several fine-tuners sequentially with the same settings.

A typical multi-GPU run uses one CARLA worker per collector/evaluator GPU and Lightning DDP for the trainer GPUs. In the examples below, GPUs `0,1,2,3` run CARLA workers and GPUs `4,5,6,7` run training.

### Training / Evaluation

Train and evaluate one fine-tuner:

```bash
bash scripts/run_train_and_eval.sh \
  -e sparsedrive_scorer \
  -c scenario \
  -f craft \
  -p bench2drive220 \
  -s 0 \
  -t 1 \
  -r 1 \
  -g 0,1,2,3 \
  -u 4,5,6,7 \
  -S \
  -V
```

Train and evaluate multiple fine-tuners sequentially:

```bash
bash scripts/run_multi_finetuners.sh \
  -e sparsedrive_scorer \
  -c scenario \
  -f craft,grpo,ppo,reinforce_plus \
  -p bench2drive220 \
  -s 0 \
  -t 1 \
  -r 1 \
  -g 0,1,2,3 \
  -u 4,5,6,7 \
  -S
```

Recommended launcher flags:

| Flag | Meaning |
|------|---------|
| `-e` | Ego policy config name without `.yaml`, such as `sparsedrive_scorer`, `sparsedrive`, `hip_ad`, `pdm_lite`, `uniad`, or `vad`. |
| `-c` | CBV / traffic policy config name without `.yaml`, such as `scenario`, `autopilot`, or `pluto`. |
| `-f` | Fine-tuner name. Supported values include `craft`, `grpo`, `ppo`, `reinforce_plus`, `rift`, and `soft_align`. `run_multi_finetuners.sh` accepts a comma-separated list. |
| `-p` | Route set under `rift/scenario/route`; pass either `bench2drive220` or `bench2drive220.xml`. |
| `-s` | Random seed. |
| `-t` | Training repetitions for the checkpoint/output tag. |
| `-r` | Evaluation runs used after training. |
| `-g` | Collector GPU list. One CARLA worker is launched for each listed GPU. |
| `-u` | Trainer GPU list. Multiple trainer GPUs automatically use Lightning DDP. |
| `-S` | Start from scratch by clearing the matching logs, rollout buffers, and checkpoints before running. |
| `-V` | Record videos during the evaluation stage. |

Evaluation uses the union of `-g` and `-u` as evaluator GPUs in `run_train_and_eval.sh`. For lower-level collection, training, evaluation-only runs, pretrained-checkpoint evaluation, and output path conventions, see the [detailed usage guide](docs/detail_usage.md).

Common outputs:

| Output | Path pattern |
|--------|--------------|
| Rollout buffers | `data/rollout_data/<ego>-<cbv>-seed<seed>/<route>x<train_repetitions>-<finetuner>_finetuner` |
| Collection logs | `log/collect_data/<ego>-<cbv>-seed<seed>/<route>x<train_repetitions>-<finetuner>_finetuner` |
| Training logs | `log/train_ego/<ego>-<cbv>-seed<seed>/<route>x<train_repetitions>-<finetuner>_finetuner` |
| Fine-tuned checkpoints | `rift/ego/model_ckpt/<ego>/<route>x<train_repetitions>-<finetuner>_finetuner-<cbv>-seed<seed>` |
| Evaluation logs | `log/eval/<ego>-<cbv>-seed<seed>/<route>x<train_repetitions>-<finetuner>_finetuner` for the default `eval_runs=1` same-route case. For `eval_runs>1` or cross-route evaluation, the suffix becomes `<eval_route_tag>-train_<train_route>x<train_repetitions>-<finetuner>_finetuner`. |

### Visualization

Plotting scripts are under `tools/plot`. The [notebook](paper_figure.ipynb) provides an executable gallery for the main figures below.

Aggregate ego evaluation metrics and export a CSV:

```bash
python tools/plot/plot_ego_eval_result.py
```

Render the scaling-and-reward combo figure. This command expects `assets/results_ego.csv`, collection logs, and matching evaluation logs:

```bash
python tools/plot/plot_scaling_reward_combo.py
```

Render policy distribution figures from a rollout snapshot and available pretrained / fine-tuned checkpoints:

```bash
python tools/plot/plot_policy_distribution.py
```

Render training-stability figures from offline W&B logs:

```bash
python tools/plot/plot_train_stability.py
```

## :bookmark: Citation

If you find our paper useful, please kindly cite us via:

```
@misc{chen2026craft,
      title={CRAFT: Counterfactual-to-Interactive Reinforcement Fine-Tuning for Driving Policies}, 
      author={Keyu Chen and Nanfei Ye and Yida Wang and Wenchao Sun and Danqi Zhao and Hao Cheng and Sifa Zheng},
      year={2026},
      eprint={2605.04470},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.04470}, 
}
```

## :heart: Acknowledgements

This implementation is based on code from several repositories. We sincerely thank the authors for their awesome work.
- [RIFT](https://github.com/CurryChen77/RIFT)
- [HiP-AD](https://github.com/nullmax-vision/HiP-AD)
- [CaRL](https://github.com/autonomousvision/CaRL)
- [LEAD](https://github.com/kesai-labs/lead)
- [SparseDriveV2](https://github.com/swc-17/SparseDriveV2)
- [MindDrive](https://github.com/xiaomi-mlab/Minddrive)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [DriveLM-CARLA](https://github.com/OpenDriveLab/DriveLM/tree/DriveLM-CARLA/pdm_lite)
- [PlanTV2](https://github.com/autonomousvision/plant2)
- [nuPlan](https://github.com/motional/nuplan-devkit)
