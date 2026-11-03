# 项目运行环境

- 预处理、实验、测试和训练统一使用 Conda 环境 `FRESH-Drive`，不再使用 `E2EAD` 虚拟环境。
- Python 命令使用 `conda run -n FRESH-Drive python ...`，或先 `conda activate FRESH-Drive`。
- 本机解释器：`/home/shi-jia-liang/miniforge3/envs/FRESH-Drive/bin/python`。
- 依赖状态见 `docs/bevformer_environment.md`；保持 BEVFormer 编译扩展与 PyTorch/CUDA 兼容。
