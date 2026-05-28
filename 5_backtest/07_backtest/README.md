# Layer 5a: 回测

## 职责
基于日频持仓和成本模型计算组合 NAV，输出 Qlib 标准格式报告。

## 使用的 Qlib 组件
- `D.features` — 拉取前向收益和基准收益
- `qlib.contrib.evaluate.risk_analysis` — Qlib 标准风险分析

## 输入
- `outputs/positions/positions_lgbm.csv` — 日频持仓
- `configs/_common.yaml` — 基准、成本参数

## 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/backtest/nav_lgbm.csv` | datetime, NAV, return, bench, cost, ... | NAV 序列 |
| `outputs/backtest/report_normal.csv` | datetime, return, cost, bench, turnover | Qlib 标准报告 |
| `outputs/backtest/report_long_short.csv` | datetime, long, short, long_short | 多空拆分 |
| `outputs/backtest/trades_lgbm.csv` | datetime, turnover, cost, n_long, n_short | 调仓换手明细 |
| `outputs/backtest/qlib_risk_analysis.csv` | — | Qlib risk_analysis 输出 |

## 成本模型
- open_cost=0.0005, close_cost=0.0015, min_cost=5
- 调仓日在收益后扣除成本

## 运行
```bash
python 5_backtest/07_backtest/run_backtest.py
```
