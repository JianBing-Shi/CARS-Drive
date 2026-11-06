# FRESH-Drive 远程服务器阶段性总结

整理日期：2026-11-04（北京时间）。本文依据本次会话和本地文件检查整理；最新连接与 refresh 权重目录已按用户提供的信息更新，未在整理时连接服务器，远端实时状态需登录后核实。

## 1. 当前路径与连接

本地项目已经从 `E2EAD` 改名为 `FRESH-Drive`，旧本地项目路径不再存在。

| 用途 | 当前路径或参数 |
| --- | --- |
| 本地项目 | `/home/shi-jia-liang/Projects/E2Eprojects/FRESH-Drive` |
| 本地数据根目录 | `/home/shi-jia-liang/Projects/E2Eprojects/datasets/nuplan/nuplan-v1.1` |
| 远端项目（最后确认） | `/root/E2EAD` |
| 远端数据根目录 | `/root/rivermind-data/nuplan/nuplan-v1.1` |
| 远端 action 日志 | `/root/E2EAD/lightning_logs/action/` |
| 远端 refresh 日志与权重目录（用户最新提供） | `/root/E2EAD/lightning_logs/refresh/` |
| SSH 主机 | `sh03-ssh.gpuhome.cc` |
| SSH 用户 | `root` |
| 最新确认端口 | `30158` |
| 最新出现的远端实例名称 | `jupyter-m7j0i4p14fp2p9r4` |

本地改名不会自动更改远端目录，上传与下载命令应分别使用两边实际路径。

```bash
ssh -p 30158 root@sh03-ssh.gpuhome.cc
```

历史端口依次出现过 `30117`、`30141`、`30175`，最新更新为 `30158`；此前实例名为 `jupyter-ap4zjs41g0tcvh1g`，当前用户提供的实例名为 `jupyter-m7j0i4p14fp2p9r4`。重启或更换实例后以平台提供的最新 SSH 登录命令为准。HTTPS webservice 地址不能直接作为 rsync over SSH 的地址；上传下载不需要先开启 8088 端口转发。

用户最新终端提示为 `root@jupyter-m7j0i4p14fp2p9r4:~/E2EAD/lightning_logs/refresh#`，并说明该目录存放 refresh 训练权重。具体 `.ckpt` 文件名、训练是否结束及验证指标尚未提供；数据路径、GPU 与环境记录来自此前实例，需在当前实例重新核实。

## 2. 数据迁移范围

远端数据盘约 170GB；用户确认足以容纳需要的四项。`features/` 超过 800GB，不上传。四项的完整性与当前占用未在本次整理时重新验证。

```text
/root/rivermind-data/nuplan/nuplan-v1.1/preprocessed_dataset/
├── events/
├── bev_features/
├── mini-train.pt
└── mini-val.pt
```

训练直接读取事件和引用的 BEV 缓存，无需上传每路相机的 `features/`、原始图像或原始 DB。保留目录层级，事件中的 BEV 路径相对于 nuPlan 版本根目录解析。

远端训练 YAML 应使用：

```yaml
datamodule:
  root: /root/rivermind-data/nuplan/nuplan-v1.1
```

不要将 `root` 设置为下面的 `preprocessed_dataset/`。

## 3. 环境与依赖

用户报告的服务器镜像为 `PyTorch/2.1.0-CUDA11.8-Py3.10-Ubuntu20`，GPU 为单卡 L40S。后续更换实例后，应重新确认 GPU、驱动和解释器。

根目录 `requirements.txt` 面向远端缓存训练，主要版本：

| 包 | 版本 |
| --- | --- |
| torch | 2.1.0+cu118 |
| torchvision | 0.16.0+cu118 |
| lightning / pytorch-lightning | 2.0.9 |
| torchmetrics | 1.2.1 |
| torch-geometric | 2.3.1 |
| numpy | 1.23.5 |
| scipy | 1.10.1 |

缓存训练不需要安装 `mmcv-full`、`mmdet`、`mmdet3d` 或 BEVFormer 编译扩展。不要将本机 PyTorch 1.13.1 / CUDA 11.7 的预处理扩展直接搬到远端 PyTorch 2.1.0 / CUDA 11.8 环境。

曾建议创建独立 `FRESH-Drive` Conda 环境，但用户提供的 traceback 使用 `/opt/conda/lib/python3.10/site-packages/`；不能据此认定已切换到独立环境。登录后先核实实际解释器：

```bash
which python
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
python -m pip check
nvidia-smi
```

若需要新建环境，在远端项目根目录执行；已有环境时跳过创建：

```bash
conda create -n FRESH-Drive python=3.10 pip -y
conda activate FRESH-Drive
python -m pip install --upgrade pip "setuptools==69.5.1" wheel
python -m pip install -r requirements.txt
```

详细说明见本地 `docs/remote_training_environment.md`。

## 4. 训练进展与 DataLoader 问题

此前 action 训练的运行日志表明（以下不是当前 refresh 训练状态）：

- 使用多进程 DataLoader 时出现 `DataLoader worker exited unexpectedly`。
- 将 `num_workers` 改为 `0` 后，通过 sanity check 并进入真实训练。
- 当时显示 `Epoch 0`，进度 `6/177`，`train/loss=0.0822`，平均约 `17.69 s/it`。
- 该日志仅证明训练已开始，不代表已完成一轮或已得到有效最终模型。
- 日志对应的约 52 分钟/轮只是当时平均速度的估计，不含后续速度变化和完整验证耗时。

`num_workers: 0` 成功说明单进程数据加载路径可运行；尚不能确认多进程失败的具体原因。主机 RAM、容器内存限制和 `/dev/shm` 是优先排查项，不应直接归因于显存不足。

当前本地 `config/train/action.yaml` 和 `refresh.yaml` 仍保留 `num_workers: 10`。直接覆盖远端配置可能重新引入故障；远端已经修改的 batch size、pin memory 等值未获用户逐项确认。

可先使用以下保守配置（batch size 等属于建议，只有 `num_workers: 0` 已获用户确认成功）：

```yaml
datamodule:
  root: /root/rivermind-data/nuplan/nuplan-v1.1
  train_batch_size: 1
  val_batch_size: 1
  num_workers: 0
  pin_memory: false
  persistent_workers: false
```

Lightning 建议 176 workers 是根据可见 CPU 数量生成的通用提示，不是适合该容器的推荐值。若要调优，在停止当前训练并保存进度后，先测试 `num_workers: 1`、`prefetch_factor: 1`，稳定后再试 `2`，并比较实际吞吐。

```bash
free -h
df -h /dev/shm
watch -n 1 nvidia-smi
```

GPU 利用率低可能与存储读取、CPU 数据处理、同步或小 batch 有关，需结合 CPU、磁盘与 GPU 指标判断。

## 5. 上传项目代码（在本地执行）

以下将本地新目录的内容同步到远端原来的 `/root/E2EAD/`。排除日志和本机缓存，避免上传日志覆盖远端同名结果；未使用 `--delete`。

```bash
rsync -avh --partial --info=progress2 \
  --exclude='lightning_logs/' \
  --exclude='runs/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='.codex/' \
  --exclude='.agents/' \
  --exclude='.aws/' \
  --exclude='.vscode/' \
  --exclude='config/train/' \
  -e "ssh -p 30158 -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /home/shi-jia-liang/Projects/E2Eprojects/FRESH-Drive/ \
  root@sh03-ssh.gpuhome.cc:/root/E2EAD/
```

这里有意排除 `config/train/`，保护远端已经调整的数据路径和 worker 设置。该命令适用于远端已存在训练配置的更新场景；首次部署时需要单独上传、调整训练配置。上传前可加 `--dry-run` 预览文件列表。不要在运行中的实验期间更新它正在使用的代码。

## 6. 只上传四项预处理数据（在本地执行）

```bash
local_src="/home/shi-jia-liang/Projects/E2Eprojects/datasets/nuplan/nuplan-v1.1/preprocessed_dataset"

ssh -p 30158 root@sh03-ssh.gpuhome.cc \
  'mkdir -p /root/rivermind-data/nuplan/nuplan-v1.1/preprocessed_dataset'

rsync -avh --partial --info=progress2 \
  -e "ssh -p 30158 -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  "$local_src/events" \
  "$local_src/bev_features" \
  "$local_src/mini-train.pt" \
  "$local_src/mini-val.pt" \
  root@sh03-ssh.gpuhome.cc:/root/rivermind-data/nuplan/nuplan-v1.1/preprocessed_dataset/
```

四个源项明确列出，不会传输 `features/`。这里目录源项末尾没有 `/`，目标是共同父目录，所以远端得到一层 `events/` 和 `bev_features/`，重复执行也不会多嵌套一层。

## 7. 下载训练日志与 checkpoint（在本地执行）

本地路径已统一更新为 `FRESH-Drive`。下载 action 日志与权重：

```bash
mkdir -p /home/shi-jia-liang/Projects/E2Eprojects/FRESH-Drive/lightning_logs/action

rsync -avh --partial --info=progress2 \
  -e "ssh -p 30158 -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  root@sh03-ssh.gpuhome.cc:/root/E2EAD/lightning_logs/action/ \
  /home/shi-jia-liang/Projects/E2Eprojects/FRESH-Drive/lightning_logs/action/
```

会下载远端实际存在的日志和 checkpoint。此前 `mkdir ... No such file or directory` 是本地父目录不存在造成的，使用 `mkdir -p` 解决。

下载当前实例的 refresh 日志与权重，使用带实例标识的本地目录保存：

```bash
mkdir -p /home/shi-jia-liang/Projects/E2Eprojects/FRESH-Drive/lightning_logs/refresh_m7j0i4p14fp2p9r4

rsync -avh --partial --info=progress2 \
  -e "ssh -p 30158 -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  root@sh03-ssh.gpuhome.cc:/root/E2EAD/lightning_logs/refresh/ \
  /home/shi-jia-liang/Projects/E2Eprojects/FRESH-Drive/lightning_logs/refresh_m7j0i4p14fp2p9r4/
```

上述命令仅为下载示例，本次更新没有执行传输。

下载正在写入的 checkpoint 不保证得到一致文件；优先在 checkpoint 保存完成后同步。不同实例可能都使用 `version_0`，如需同时保留结果，应使用独立备份目录，避免同名文件被覆盖。

rsync 的通用规则：

```text
rsync [选项] 源路径 目标路径
上传：本地路径 → 用户@主机:远端路径
下载：用户@主机:远端路径 → 本地路径
```

源目录末尾 `/` 表示同步内容；不带 `/` 时把源目录本身放入已存在的目标目录。中断后重跑相同命令会跳过未变化文件并重新同步差异；`--partial` 保留未完成文件。

## 8. SSH 指纹问题记录

旧端口 `30117` 曾报告服务器指纹变化。在当时的旧实例中，网页终端文件指纹与容器内 22 端口实际返回的指纹一致，但公网旧端口不同；之后发现平台已将 SSH 端口改为 `30141`。

此记录解释当时问题，不应将旧实例指纹直接作为当前 `30158` 实例的身份依据。遇到指纹冲突时，先确认平台最新地址和端口，并通过当前实例可信网页终端或平台核验；不要关闭主机密钥校验。

## 9. 下一阶段核对项

1. 登录最新 `30158` 实例，确认远端代码、数据、日志仍在上述路径。
2. 确认四项数据传输完整；出现缺失文件或反序列化异常时，针对相关文件重传。
3. 核实实际 Python 环境、GPU 和依赖版本。
4. 保持已验证的 `num_workers: 0`，检查训练是否仍在运行及最新验证指标。
5. 列出 `/root/E2EAD/lightning_logs/refresh/` 下的具体 checkpoint，确认训练状态与验证指标，在保存完成后下载；本次仅确认用户提供的权重目录，未核实文件或执行回传。
6. 检查 refresh 实验的配置和 action 初始化来源，确认使用五步滚动规划协议；后续同协议恢复可使用 refresh checkpoint。
