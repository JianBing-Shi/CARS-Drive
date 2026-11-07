# FRESH-Drive 远程训练环境

适用环境：Linux x86_64、Python 3.10、单卡 NVIDIA L40S，服务器镜像为
PyTorch 2.1.0 / CUDA 11.8。项目根目录的 `requirements.txt` 用于训练已经生成的
`events/` 与 `bev_features/`，固定 PyTorch 的 CUDA 11.8 构建。

本机预处理环境仍为 PyTorch 1.13.1 / CUDA 11.7。两者使用不同的 PyTorch 版本；
不要将本机 BEVFormer 的编译扩展复制到远程训练环境。

## 安装

远端训练直接使用服务器镜像提供的 Python/PyTorch 环境，不创建或激活 Conda 环境。
将项目代码和 `requirements.txt` 上传到服务器，在远端项目根目录执行：

```bash
python --version
python -m pip install --upgrade pip "setuptools==69.5.1" wheel
python -m pip install -r requirements.txt
```

如果镜像已经提供匹配的 PyTorch 2.1.0/CUDA 11.8，可以先执行 `python -m pip check`；
只有依赖缺失或版本不匹配时再执行完整安装。CUDA 运行库由镜像和 PyTorch 提供，
缓存训练无需另外安装 CUDA Toolkit 或编译器。

## 验证

仍在项目根目录和服务器默认 Python 环境中执行：

```bash
python -m pip check
python - <<'PY'
import sys
import torch
import torchvision
import lightning.pytorch as pl
import torch_geometric
from datamodules import NuplanDataModule
from model import CARSDirve

print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("Lightning:", pl.__version__, "PyG:", torch_geometric.__version__)
assert torch.cuda.is_available(), "CUDA 不可用，请检查 GPU 分配和驱动"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda", requires_grad=True)
(x @ x.T).square().mean().backward()
torch.cuda.synchronize()
print("CUDA forward/backward and training imports: OK")
PY
```

还可以运行项目已有的 CPU 小样本训练测试，验证缓存读取、两个训练阶段和 checkpoint 恢复：

```bash
python -m unittest discover -s tools -p test_train.py
python -m unittest discover -s tools -p test_nuplan_datamodule.py
```

生成依赖文件时已核对本机已安装包版本和当前训练导入链；远端 PyTorch 2.1.0 / L40S
的安装和 CUDA 验证需要在服务器完成，本机验证不能代替该检查。

## 数据与启动

远程数据应保持以下结构，无需上传 `features/`：

```text
/root/rivermind-data/nuplan/nuplan-v1.1/preprocessed_dataset/
├── events/
├── bev_features/
├── mini-train.pt
└── mini-val.pt
```

将所用 `config/train/pred.yaml` 或 `config/train/plan.yaml` 中的数据根目录设为
nuPlan 版本目录，而不是它下面的 `preprocessed_dataset/`：

```yaml
datamodule:
  root: /root/rivermind-data/nuplan/nuplan-v1.1
```

```bash
python train.py --config config/train/pred.yaml
```

刷新阶段使用 `config/train/plan.yaml`，并将 `trainer.ckpt_path` 指向已经训练好的
action checkpoint 或需要恢复的 refresh checkpoint。

## 依赖范围

训练 DataModule 以 `preprocess=False` 加载缓存；数据加载器的 `get()` 读取事件与
其引用的 BEV 文件，不运行图像编码器。BEVFormer/OpenMMLab 的导入位于编码器加载
函数内部，因此当前训练不需要 `mmcv-full`、`mmdet`、`mmdet3d` 或 `nuplan-devkit`，
也不需要 `torch-scatter`、`torch-sparse` 等可选 PyG 扩展。日志使用 CSVLogger，
不要求安装 TensorBoard。

这份文件没有锁定全部间接依赖。远端安装和验证成功后，可保存该服务器的完整版本快照：

```bash
python -m pip freeze > requirements.remote.lock.txt
```

若需要在远端重新从图像预处理 BEV，应单独配置与 PyTorch/CUDA ABI 匹配的
BEVFormer 环境，参见 `docs/bevformer_environment.md`。
