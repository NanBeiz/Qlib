# Layer 5b: 分析

## 职责
单因子 IC / Rank IC / ICIR 分析 + 模型预测 IC + 组合绩效指标。

## 使用的 Qlib 组件
- `qlib.contrib.evaluate.risk_analysis` — Qlib 标准风险指标

## 输入
- `outputs/factors/all_factors_raw.csv` — 原始因子 + LABEL0
- `outputs/predictions/predictions_lgbm.csv` — LGBM 预测信号
- `outputs/backtest/report_normal.csv` — 回测报告

## 输出
| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/analysis/factor_ic.csv` | factor, ic_mean, ic_std, icir, rank_ic_mean, rank_icir | 因子 IC 汇总 |
| `outputs/analysis/factor_ic_series.csv` | datetime × factor | 逐日 IC 序列 |
| `outputs/analysis/model_ic.csv` | factor(ALPHA), ic_mean, icir | 模型预测 IC |
| `outputs/analysis/performance_metrics.csv` | total_return, sharpe, max_dd, calmar, ... | 绩效指标 |

## 运行
```bash
python 5_backtest/08_analysis/factor_analysis.py
```
