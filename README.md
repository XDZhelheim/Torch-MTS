# Torch-MTS
A simple PyTorch framework for multivariate time series forecasting, for personal use.

All the codes are based on my personal coding habit and style.

## 几个关键点

先看 [scaler_check.ipynb](scaler_check.ipynb)

1. 不要用 sklearn 的 `StandardScaler`
2. 构建 train-val-test 的时候必须只使用 x_train 的分布构建 scaler
3. 读取原始数据的时候 `.astype(np.float32)`，保证全程的数据不管是 ndarray 还是 tensor 都是 float32 类型
4. [重点重点重点] 千万不要 transform 任何 y，这一点可以解决各种各样的奇怪问题，例如 MAPE 爆炸
5. 在 forward 过程中 `inverse_transform(y_pred)` 之后再和 `y_true` 算 loss。如果算 loss 时 `y_pred, y_true` 是 transform 后的，会严重影响 performance
6. metrics 中不要修改 `y_pred, y_true` 的任何值，例如把特别小的值置为 0。引起这个操作的情况是 MAPE 分母的 `y_true` 中有一些特别小的值导致算出来的值爆炸，原因在 [scaler_check.ipynb](scaler_check.ipynb) 里分析过了，所以只要按照上面的 1234 的要求做，就不会出现这种情况
7. Learning scheduler: `MultiStepLR` 必须有而且要细调，而且要和课程学习的设置联动
8. 课程学习 对复杂模型很管用
9. 可能有用也可能没用的 trick: l2 reg, grad clip

## Changelog
* v1.1: 一个稳定版本，包含 MetaLSTM
* v1.2: 参考 BasicTS，重构了整个 dataset 的创建和读取

## TODO

- HA
- 自定义 forward，自定义 loss
- 大饼：log 生成表格的脚本；支持输入 1) 多个.log 2) 模型名 3) log文件夹名；支持输出 1) .csv 2) .tex 3) .png

logs:

- Attn: 7 w-spatial
- MTGNN: bay, 3478 +tod
- STMetaGCGRU: bay
- STWA: 3478
