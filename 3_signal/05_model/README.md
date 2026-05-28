# Layer 3b & 3c: 模型训练与预测

## Layer 3b: 训练

### 职责
使用 Qlib LGBModel 在 train/valid 段上训练 LightGBM 模型。

### 使用的 Qlib 组件
- `qlib.contrib.model.gbdt.LGBModel` — LGBM 包装器（自动早停、日志）

### 输入
- `outputs/dataset/dataset.pkl` — DatasetH 实例
- `configs/_common.yaml` — 日期分段

### 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/model/lgbm_model.pkl` | pickle | 训练好的 LGBModel |
| `outputs/model/feature_importance.csv` | CSV | 特征重要性排序 |
| `outputs/model/training_log.txt` | text | LGBM 训练日志 |

### 运行
```bash
python 3_signal/05_model/train_lgbm.py
```

## Layer 3c: 预测

### 职责
加载训练好的模型，在 test 段上生成预测信号。

### 输入
- `outputs/model/lgbm_model.pkl`
- `outputs/dataset/dataset.pkl`

### 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/predictions/predictions_lgbm.csv` | datetime, instrument, ALPHA | LGBM 预测信号 |
| `outputs/predictions/predictions_baseline.csv` | datetime, instrument, ALPHA | 等权 baseline 信号 |

### 运行
```bash
python 3_signal/05_model/predict_lgbm.py
```
