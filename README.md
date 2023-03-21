# Torch-MTS
A simple PyTorch framework for multivariate time series forecasting, for personal use.

All the codes are based on my personal coding habit and style.

## 几个关键点

先看 [scaler_check.ipynb](scaler_check.ipynb)

1. 不要用 sklearn 的 `StandardScaler`
2. 构建 train-val-test 的时候必须只使用 x_train 的分布构建 scaler
3. 读取原始数据的时候 `.astype(np.float32)`，保证全程的数据不管是 ndarray 还是 tensor 都是 float32 类型
4. 千万不要 transform 任何 y。在 forward 过程中 `inverse_transform(y_pred)` 之后再和 `y_true` 算 loss。如果算 loss 时 `y_pred, y_true` 是 transform 后的，会严重影响 performance。
5. metrics 中不要修改 `y_pred, y_true` 的任何值，例如把特别小的值置为 0。引起这个操作的情况是 MAPE 分母的 `y_true` 中有一些特别小的值导致算出来的值爆炸，原因在 [scaler_check.ipynb](scaler_check.ipynb) 里分析过了，所以只要按照上面的 2 和 3 的要求做，就不会出现这种情况
6. Learning scheduler: `MultiStepLR` 必须有而且要细调
7. 课程学习 很管用
8. 可能有用也可能没用的 trick: l2 reg, grad clip
