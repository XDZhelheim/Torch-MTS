# Torch-MTS

A PyTorch framework for multivariate time series forecasting (MTS), mainly for personal research and experiments.

## 1. 环境搭建 🧩

本项目依赖尽量保持精简，安装常用库即可。

使用 conda 或 pip 安装：

```text
torch numpy pandas pyyaml matplotlib torchinfo
```

## 2. 快速上手 🚀

> **重要**：脚本内部使用了大量 `../` 相对路径，建议先进入 `scripts/` 目录再执行命令。

```bash
cd scripts
```

### 2.1 生成数据

```bash
python generate_training_data.py -d METRLA -s <history_seq_len> -p <future_seq_len>
```

会在 `data/<DATASET>/` 下生成：

- `data.npz`
- `index_<history_seq_len>_<future_seq_len>.npz`

常用设置下的数据文件已提前生成并随仓库提供。  
TODO：改成动态加载，避免为每种历史长度和预测长度都生成一遍数据。

### 2.2 启动训练

```bash
python train.py -m <model> -d <dataset> -g <gpu_id>
```

常见用法：

```bash
# 指定配置文件
python train.py -m STGCN -d METRLA --config ../configs/STGCN.yaml

# LTSF 任务覆盖输入/预测长度
python train.py -m DLinear -d ETTH1 -g 0 -s 336 -p 96
```

### 2.3 后台训练

```bash
bash run_nohup.sh STGCN METRLA 0
```

### 2.4 查看结果

- 日志：`logs/<ModelName>/<ModelName>-<DATASET>-<timestamp>.log`
- 权重：`saved_models/<ModelName>/<ModelName>-<DATASET>-<timestamp>.pt`

批量汇总日志指标：

```bash
python get_metrics.py
```

会生成 `logs/results.csv` 和 `logs/results_long.csv`

## 3. 实验结果总表 📊

运行环境：PyTorch 1.13.1 + NVIDIA RTX 5000 Ada 32G

![TMTS-Table-250225](https://github.com/user-attachments/assets/25ab75af-b6d9-4fff-aa87-cb76ed1f969b)

## 4. 目录结构 🗂️

```text
Torch-MTS/
├── configs/        # 各模型配置文件（按数据集分节）
├── data/           # 数据集目录（原始文件 + 预处理后的 data/index）
├── lib/            # 数据加载、损失、指标、工具函数
├── logs/           # 训练日志和结果汇总
├── models/         # 模型实现与模型选择
├── runners/        # 训练/验证/测试流程（STF/LTSF + 特定模型 runner）
├── saved_models/   # 模型权重保存目录
└── scripts/        # 预处理、训练、指标汇总等脚本
```

## 5. 已支持模型和数据集 🧠

**模型：**

- Baseline：`HistoricalInertia`, `MLP`, `LSTM`, `GRU`, `Transformer`, `WaveNet`, `Mamba`, `GCLSTM`, `GCGRU`, `GCRN`
- 时空/交通：`STGCN`, `DCRNN`, `AGCRN`, `GWNET`, `MTGNN`, `GMAN`, `STNorm`, `StemGNN`, `STID`, `STWA`, `MegaCRN`, `GTS`, `STAEformer`, `HimNet`, `STDN`
- 长序列：`DLinear`, `PatchTST`

**数据集：**

- 时空预测（STF）：`METRLA`, `PEMSBAY`, `PEMS03/04/07/08`, `PEMSD7M/L`
- 长序列预测（LTSF）：`ELECTRICITY`, `WEATHER`, `TRAFFIC`, `EXCHANGE`, `ILI`, `ETTH1/ETTH2/ETTM1/ETTM2`

## 6. 主要命令参数 ⌨️

### 6.1 `train.py`

- `-d, --dataset`：数据集名（默认 `METRLA`）
- `-m, --model`：模型名，大小写不敏感（默认 `LSTM`）
- `-g, --gpu_num`：GPU 编号（默认 `0`）
- `-c, --compile`：启用 `torch.compile`（要求 torch >= 2.0）
- `--seed`：随机种子（默认 `233`）
- `--cpus`：限制 CPU 线程数（默认 `1`）
- `--config`：指定配置文件路径
- `-s, --seq_len`：覆盖 LTSF 输入长度
- `-p, --pred_len`：覆盖 LTSF 预测长度

### 6.2 `generate_training_data.py`

- `-d, --dataset`：数据集名
- `-s, --history_seq_len`：历史窗口长度
- `-p, --future_seq_len`：预测窗口长度
- `--target_channel`：目标特征（默认 `[0]`）

## 7. 训练流程说明 ⚙️

`scripts/train.py` 主流程如下：

1. 解析参数（数据集、模型、GPU、配置路径、随机种子等）
2. 读取配置文件 `configs/<ModelName>.yaml`（或 `--config` 指定）
3. 根据配置选择 dataloader、loss、runner
4. 使用 Adam + MultiStepLR 训练，并用早停保存最佳权重
5. 在测试集输出指标并记录日志

### 7.1 Runner 机制

- `STFRunner`：用于交通/时空预测，训练与评估前会先对 `y_pred` 做 `inverse_transform` 再计算 loss/metrics
- `LTSFRunner`：用于长序列预测，默认输出 `MSE/MAE`
- 特定模型 runner：`DCRNNRunner`, `MegaCRNRunner`, `HimNetRunner` 等

### 7.2 默认 loss 选择（`lib/losses.py`）

- METRLA/PEMSBAY/PEMSD7M/L：`MaskedMAELoss`
- PEMS03/04/07/08：`HuberLoss`
- LTSF 数据集：`MSELoss`

## 8. 数据准备细节 🧪

`scripts/generate_training_data.py` 支持读取：

- `.h5`（METRLA, PEMSBAY）
- `.npz`（PEMS03/04/07/08, PEMSD7M/L）
- `.csv`（ELECTRICITY, WEATHER, TRAFFIC, EXCHANGE, ILI, ETTH/ETTM）

并可自动添加两类额外特征：

- `time_of_day`
- `day_of_week`

脚本支持三种 train/val/test 切分策略：

- `DEFAULT`：先滑窗再切分（样本更多，但会跨 train/val/test 边界），时空预测常用
- `STRICT`：先切分再滑窗（最严格，样本最少）
- `LTSF`：与主流 LTSF 代码保持一致（但是 val 与 train 之间, test 与 val 之间存在重叠）

**GMAN 额外步骤（SE 生成）**

GMAN 配置依赖 `SE_file_path`（见 `configs/GMAN.yaml`）。  
如需生成结构嵌入，可执行：

```bash
pip install gensim networkx
cd scripts
python gen_SE.py
```

脚本会基于邻接矩阵生成 `SE_*.txt` 文件（node2vec + Word2Vec 流程）。

## 9. 实践建议（重要）✅

1. 不要用 sklearn 的 `StandardScaler`
2. 构建 train-val-test 的时候必须只使用 x_train 的分布构建 scaler
3. 读取原始数据时使用 `.astype(np.float32)`，保证 ndarray/tensor 全程 `float32`
4. **千万不要 transform 任何 y**，可避免 MAPE 爆炸等问题
5. 在 forward 过程中 `inverse_transform(y_pred)` 后再与 `y_true` 计算 loss
6. metrics 中不要修改 `y_pred, y_true` 的值（例如将很小的值置 0）
7. 学习率调度建议使用并细调 `MultiStepLR`

## 10. Changelog 📝

- v1.1：第一个可用版本
- v1.2：参考 BasicTS，重构 dataset 的创建与读取
- v1.3：添加独立的 loss 与 runner 模块
- v1.4：交通预测稳定版本（或许是完全 bug-free），持续使用半年以上
- v1.5：添加长序列数据集支持，并增加新模型与实验结果

> 本文档由 Copilot + GPT-5.3-Codex 生成，我又改了改，效果还行
