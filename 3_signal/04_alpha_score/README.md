# Layer 3a: Alpha 合成

## 职责
从 handler 构造 DatasetH（含 train/valid/test 分段），
同时生成等权 baseline 的 ALPHA 信号。

等权 baseline 的意义：LGBM 模型必须显著优于这个简单 baseline 才算有价值。

## 使用的 Qlib 组件
- `DatasetH` — 分段数据集容器
- `DataHandlerLP` — 数据处理器

## 输入
- `outputs/handler/csi500_alpha14_handler.pkl` — 已初始化的 handler
- `configs/_common.yaml` — 日期分段

## 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/dataset/dataset.pkl` | pickle | DatasetH 实例 |
| `outputs/dataset/baseline_equal_weight.csv` | datetime, instrument, ALPHA_BASELINE | 等权 baseline |

## 算法
等权 baseline = 所有 14 个处理后特征等权平均 → 截面 rank percentile → ALPHA_BASELINE ∈ [0, 1]

## 运行
```bash
python 3_signal/04_alpha_score/build_dataset.py
```
