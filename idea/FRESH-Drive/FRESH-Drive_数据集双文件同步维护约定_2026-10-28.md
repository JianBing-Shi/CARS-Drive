# FRESH-Drive 数据集双文件同步维护约定

日期：2026-10-28（Asia/Shanghai）

## 1. 维护要求

[`preprocess_dataset.py`](../../preprocess_dataset.py) 与 [`datasets/nuplanevent_dataset.py`](../../datasets/nuplanevent_dataset.py) **始终保持完整文件内容一致**，包括导入、类定义、默认参数、属性、预处理函数、样本读取函数和 `__main__` 入口。

这是项目采用的双文件维护方式。后续涉及任一文件的修改，必须同步更新另一文件，并在完成修改后检查两者内容一致。两个文件均保留完整的 `NuplanEventDataset` 实现，不将其中一个精简为仅导入另一个文件的入口。

## 2. 使用入口

| 使用场景 | 入口 | 行为 |
|---|---|---|
| 手动预处理 | `python preprocess_dataset.py` | 按脚本入口初始化数据集，缓存缺失时执行预处理。 |
| Lightning 数据加载 | `NuplanDataModule` 从 `datasets` 导入 `NuplanEventDataset` | 传入 `preprocess=False`，读取已生成的事件清单和共享 BEV 缓存；缺少所需清单时提示先预处理。 |

两个文件通过调用参数区分预处理与训练读取行为，不通过维护不同版本的类实现区分。当前 `event_types` 参数用于读取清单后的类别筛选；预处理候选事件仍由 `datasets/dataset_utils.py` 中的 `EVENT_TYPES` 控制。

同样参数和缓存条件下，两份实现应具有相同的数据路径、事件格式、划分规则和读取结果。训练加载不需要重新提取特征，也不要求原始 DB 和编码器权重存在。

## 3. 修改与检查

先完成一份实现的修改，再将修改同步至另一份；两份文件作为同一次变更维护。同步检查可在项目根目录执行：

```bash
cmp preprocess_dataset.py datasets/nuplanevent_dataset.py
```

命令退出状态为 `0` 表示文件逐字节一致；存在差异时使用以下命令查看：

```bash
diff -u preprocess_dataset.py datasets/nuplanevent_dataset.py
```

同步维护的是代码内容，不要求分别执行两份脚本来重复生成数据。当前约定依靠修改时同步及文件比较检查，不包含后台自动同步机制。

涉及数据加载行为的修改，使用项目指定的 `FRESH-Drive` 环境运行相应检查：

```bash
conda run -n FRESH-Drive python -m unittest discover -s tools -p test_nuplan_datamodule.py
```

## 4. 约定范围

本约定只确定两个代码文件的同步维护方式，不改变现有预处理数据格式、事件选择、训练验证划分或模型输入接口。后续若修改这些行为，也应同时更新两份实现及对应说明，避免预处理输出与训练读取出现差异。
