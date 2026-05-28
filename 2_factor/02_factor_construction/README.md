# Layer 2a: 因子构建

## 职责
从 alpha14.yaml 读取因子定义，用 Qlib 表达式引擎一次性计算 14 个因子的原始值。
产出供人肉眼检查和单因子 IC 探索，**不被 Layer 2b 消费**（Layer 2b 独立重算）。

## 使用的 Qlib 组件
- `D.features` + Qlib 表达式引擎

## 输入
- `configs/factors/alpha14.yaml` — 14 因子表达式
- `configs/_common.yaml` — 日期范围
- `outputs/data/universe_csi500.csv` — 股票池

## 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/factors/all_factors_raw.csv` | datetime, instrument, factor1..14 | 原始未处理因子值 |
| `outputs/factors/factor_coverage.csv` | factor, coverage, first_valid_date | 每个因子的非空率和首个有效日 |

## 运行
```bash
python 2_factor/02_factor_construction/build_factors.py --config configs/factors/alpha14.yaml
```
