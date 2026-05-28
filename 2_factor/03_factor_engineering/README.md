# Layer 2b: 因子工程

## 职责
从 YAML 配置实例化 CSI500AlphaHandler（含完整 processor 链），
触发数据加载和处理，pickle 持久化为 DatasetH 可用的格式。

## 使用的 Qlib 组件
- `init_instance_by_config` — 从 YAML dict 实例化
- `DataHandlerLP` — 数据加载 + processor 链管理
- `QlibDataLoader` — Qlib 表达式加载器
- `ProcessInf` / `RobustZScoreNorm` / `Fillna` — 内置处理器
- `DropnaLabel` / `CSRankNorm` — 标签处理器
- `IndustryMcapNeutralize` — 自定义截面中性化处理器

## 输入
- `configs/handlers/csi500_alpha14.yaml` — Handler + processor 链配置
- `configs/_common.yaml` — 日期范围

## 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/handler/csi500_alpha14_handler.pkl` | pickle | DataHandlerLP 实例（含数据缓存） |
| `outputs/handler/features_processed_sample.csv` | CSV | 前 30 个交易日特征快照（调试用） |

## 处理链
```
infer:  ProcessInf → RobustZScoreNorm → IndustryMcapNeutralize → Fillna
learn:  infer + DropnaLabel → CSRankNorm
```

## 运行
```bash
python 2_factor/03_factor_engineering/build_handler.py --config configs/handlers/csi500_alpha14.yaml
```
