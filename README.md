# CSI500 多因子市场中性策略（Beta + Risk 完整版）

基于 Microsoft Qlib 构建的 CSI500 成分股多因子市场中性策略，集成 Beta 对冲 + 风险模型 + 风险约束优化。

---

## 1. 策略名称

**quant_csi500_mn** — CSI500 Multi-Factor Market Neutral Strategy (Beta + Risk Edition)

---

## 2. 策略类型

| 属性 | 值 |
|------|-----|
| 策略类型 | 多因子选股 + 市场中性（多空对冲） |
| 选股池 | 中证 500 指数成分股（CSI500） |
| 多空结构 | 月末调仓，Top 50 / Bottom 50 |
| 中性维度 | 美元中性 + **Beta 中性**（IC 期货对冲）+ **风险约束**（协方差优化） |
| 权重方法 | 等权 / 逆波动率 / 风险平价 / GMV / MVO |
| 数据频率 | 日频 |
| 基准 | 沪深 300（SH000300，绩效对比） |

---

## 3. 四层策略入口

| 入口 | 选股 | 权重 | Beta 对冲 | 风险模型 |
|------|------|------|-----------|----------|
| `generate_positions.py` | alpha | 等权 | — | — |
| `generate_positions_with_hedge.py` | alpha | 等权 | IC 期货 | — |
| `generate_positions_risk.py` | alpha | **风险优化** | — | 协方差矩阵 |
| `generate_positions_risk_hedge.py` | alpha | **风险优化** | **IC 期货** | 协方差矩阵 |

### 策略流程图

```
predict_lgbm.py
    ↓
alpha 排序 → top 50 / bottom 50 候选
    ↓
┌──────────────────────────────────┐
│  风险模型 (3_signal/riskmodel/)  │
│  ShrinkCov / StructuredCov / POET│
│  → 协方差矩阵 Σ                  │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│  风险优化器 (risk_optimizer.py)  │
│  inv / rp / gmv / mvo           │
│  → 风险加权持仓                   │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│  Beta 对冲 (hedge_manager.py)    │
│  net_beta × portfolio_notional   │
│  → IC 期货合约手数                │
└──────────────────────────────────┘
    ↓
positions_risk.csv + hedge_risk.csv
    ↓
run_backtest.py
```

---

## 4. 数据说明

### 4.1 数据源

| 项目 | 说明 |
|------|------|
| 行情数据 | Qlib 本地数据 `~/.qlib/qlib_data/cn_data/`，OHLCV + Amount |
| 指数成分股 | CSI500 指数成分股（Qlib `csi500` 内置） |
| 行业分类 | 申万一级行业（Qlib `$industry_sw_level1`） |
| 期货数据 | 中金所 IC/IF/IH/IM 主力连续合约 CSV（待接入，接口已预留） |

### 4.2 时间分段

| 分段 | 时间区间 | 用途 |
|------|----------|------|
| train | 2017-01-01 ~ 2020-12-31 | 模型训练 + 处理器 fit |
| valid | 2021-01-01 ~ 2021-12-31 | 早停验证 |
| test | 2022-01-01 ~ 2024-12-31 | 样本外回测（708 个交易日） |

---

## 5. 因子说明

### 5.1 因子总览（14 因子 / 5 分组）

| 分组 | 数量 | 因子 |
|------|------|------|
| 低波动 `low_volatility` | 3 | lowvol, vol_20, idio_vol_60 |
| 动量 `momentum` | 3 | momentum_12_1, mom_20_5, mom_60_5 |
| 反转 `reversal` | 4 | reversal_20, rev_1, rev_5, resid_rev_5 |
| 流动性 `liquidity` | 2 | amihud_20, liq_20 |
| 量价结构 `volume_price` | 2 | close_pos, ret_amount_corr_20 |

### 5.2 因子处理链

```
infer（train + valid + test）:
  ProcessInf → RobustZScoreNorm(clip_outlier=true, fit 仅 train 段)
  → IndustryMcapNeutralize（行业+市值截面中性化） → Fillna(0)

learn（仅 train，叠加 infer）:
  DropnaLabel → CSRankNorm（截面 rank 归一化）
```

---

## 6. 模型说明

| 项目 | 说明 |
|------|------|
| 模型 | LightGBM（Qlib `LGBModel`） |
| 损失函数 | MSE |
| 超参数 | num_leaves=64, lr=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.7 |
| 早停 | 30 rounds |
| 标签 | `LABEL0 = Ref($close, -1) / $close - 1`（T+1 收益率） |

---

## 7. 风险模型

### 7.1 协方差估计

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `shrink` | Ledoit-Wolf 收缩估计（默认） | 稳健，样本少时表现好 |
| `structured` | PCA/FA 因子模型 | 可分解因子/特质风险 |
| `poet` | POET 主成分阈值估计 | 高维稀疏场景 |

### 7.2 使用方法

```bash
# 构建风险模型报告
python 3_signal/riskmodel/build_riskmodel.py

# 切换协方差方法
python 3_signal/riskmodel/build_riskmodel.py --method structured
```

### 7.3 风险指标

| 指标 | 函数 | 说明 |
|------|------|------|
| 组合波动率 | `compute_portfolio_volatility()` | 年化 σ = √(w'Σw) × √252 |
| VaR | `compute_var()` | 参数法 Value-at-Risk |
| CVaR | `compute_cvar()` | 条件 VaR / Expected Shortfall |
| 风险贡献 | `compute_risk_decomposition()` | 每只股票的边际/成分风险贡献 |
| 因子分解 | `compute_variance_decomposition()` | 因子风险 vs 特质风险占比 |

---

## 8. 风险优化器

### 8.1 优化方法

| 方法 | 说明 | 特点 |
|------|------|------|
| `inv` | 逆波动率 | 最稳定，只用对角线，低波高权重 |
| `rp` | 风险平价 | 等风险贡献，需 scipy 优化 |
| `gmv` | 全局最小方差 | 纯防御，最小化组合方差 |
| `mvo` | 均值-方差 | 最大化 alpha - λ × 风险 |

### 8.2 实现原理

多空组合拆为两条腿分别优化（复用 Qlib `PortfolioOptimizer`，不修改源码）：

- **多头腿**：top 50 股票 → `w >= 0, sum(w) = 1` → 取正
- **空头腿**：bottom 50 股票 → `w >= 0, sum(w) = 1` → 取负
- Fallback 链：指定方法 → inv → 等权

---

## 9. Beta 对冲

### 9.1 对冲参数

| 参数 | 值 |
|------|-----|
| 对冲工具 | 中证 500 股指期货（IC） |
| 合约乘数 | 200 元/点 |
| 保证金比例 | 12% |
| Beta 估计 | 滚动 252 日 OLS，shift(1) 防未来函数 |

### 9.2 计算公式

```
contracts = net_beta × portfolio_notional / (futures_price × multiplier × futures_beta)
```

正 net_beta → 做空期货（负手数），负 net_beta → 做多期货（正手数）。

### 9.3 输出

`outputs/positions/hedge_risk.csv`，含 `action` 字段：

| action | 含义 |
|--------|------|
| `rebalance` | 调仓日新计算的对冲仓位 |
| `carry` | 非调仓日沿用上期 |
| `skip_no_data` | 数据缺失跳过 |
| `skip_threshold` | 合约数低于阈值跳过 |

---

## 10. 回测结果

**测试期：2022-01-01 ~ 2024-12-31（708 个交易日）**

| 指标 | 等权版 | 风险优化版 (inv) |
|------|--------|-----------------|
| 累计收益 | +55.37% | +37.79% |
| 年化收益 | +16.98% | +11.59% |
| 年化波动率 | 16.40% | 15.62% |
| Sharpe Ratio | 1.04 | 0.74 |
| 最大回撤 | -18.61% | -18.94% |
| Calmar Ratio | 0.91 | 0.61 |

---

## 11. 风险控制

| 维度 | 措施 |
|------|------|
| Alpha 因子层 | `IndustryMcapNeutralize` 截面去均值 |
| 权重层 | 风险约束优化（inv/rp/gmv/mvo）替代等权 |
| Beta 层 | IC 期货对冲，目标 Beta = 0 |
| 回测层 | `Exchange.limit_threshold=0.095`，涨跌停过滤 |
| 流动性层 | 排除零成交，Amihud/成交额因子约束 |
| 协方差层 | Ledoit-Wolf 收缩估计，NaN 填充 |

---

## 12. 项目结构

```
quant_csi500_mn_beta_risk/
├── configs/
│   ├── _common.yaml
│   ├── factors/alpha14.yaml
│   ├── handlers/
│   ├── riskmodel/default.yaml       # 风险模型配置
│   └── strategies/
│       ├── top_bottom_neutral.yaml  # 等权策略
│       ├── top_bottom_hedged.yaml   # 等权 + Beta 对冲
│       └── risk_based.yaml          # 风险优化策略
├── src/quant_csi500_mn/
│   ├── handlers/                    # CSI500AlphaHandler
│   ├── processors/                  # IndustryMcapNeutralize
│   ├── strategies/                  # TopBottomNeutralStrategy
│   ├── hedging/                     # Beta 对冲 re-export 层
│   └── utils/
├── 1_data/                          # Layer 1: 数据准备
├── 2_factor/                        # Layer 2: 因子构建 + 工程
├── 3_signal/
│   ├── 04_alpha_score/              # Alpha 合成
│   ├── 05_model/                    # LGBM 训练 + 预测
│   └── riskmodel/                   # ★ 风险模型（协方差 + 风险指标）
│       ├── covariance.py            #   PortfolioCovEstimator
│       ├── risk_metrics.py          #   纯函数风险指标
│       └── build_riskmodel.py       #   CLI 入口
├── 4_strategy/06_strategy_top_bottom/
│   ├── generate_positions.py        # 等权
│   ├── generate_positions_with_hedge.py  # 等权 + Beta 对冲
│   ├── generate_positions_risk.py        # 风险优化
│   ├── generate_positions_risk_hedge.py  # ★ 风险优化 + Beta 对冲
│   ├── portfolio_beta.py            # Beta 估计
│   ├── hedge_manager.py             # 期货合约计算
│   ├── futures_data_interface.py    # 期货数据抽象层
│   └── risk_optimizer.py            # ★ 多空风险优化器
├── 5_backtest/                      # Layer 5: 回测 + 分析
├── scripts/
│   ├── run_pipeline_no_beta.sh
│   ├── run_pipeline_with_beta.sh
│   ├── run_pipeline_risk_based.sh
│   └── run_pipeline_risk_hedge.sh   # ★ 完整版（风险 + 对冲）
└── tests/
    ├── test_hedging.py
    ├── test_riskmodel.py
    └── test_risk_optimizer.py
```

---

## 13. 快速开始

```bash
cd quant_csi500_mn_beta_risk
pip install -e .

# 等权，无对冲
bash scripts/run_pipeline_no_beta.sh

# 等权 + Beta 对冲
bash scripts/run_pipeline_with_beta.sh

# 风险优化权重
METHOD=inv bash scripts/run_pipeline_risk_based.sh

# 完整版：风险优化 + Beta 对冲
METHOD=inv bash scripts/run_pipeline_risk_hedge.sh

# 有期货数据时
FUTURES_CSV=outputs/data/futures_csi500.csv METHOD=inv bash scripts/run_pipeline_risk_hedge.sh

# 测试
pytest tests/ -v
```
