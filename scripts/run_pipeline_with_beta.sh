#!/bin/bash
# run_pipeline_with_beta.sh —— 美元中性 + Beta 对冲（股指期货）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Python 解释器（默认 conda qlib_final，可通过环境变量覆盖）
PYTHON="${PYTHON:-/home/hcl/miniconda3/envs/qlib_final/bin/python}"

# 期货数据路径（可通过环境变量覆盖）
FUTURES_CSV="${FUTURES_CSV:-outputs/data/futures_csi500.csv}"

echo "===== Pipeline (With Beta Hedge) Start: $(date) ====="
echo ""

# ---- Layer 1: 数据准备 ----
echo "[1/8] Layer 1: 数据准备..."
${PYTHON} 1_data/01_data_processing/prepare_data.py
echo ""

# ---- Layer 2a: 因子构建 ----
echo "[2/8] Layer 2a: 因子构建..."
${PYTHON} 2_factor/02_factor_construction/build_factors.py
echo ""

# ---- Layer 2b: 因子工程 ----
echo "[3/8] Layer 2b: 因子工程..."
${PYTHON} 2_factor/03_factor_engineering/build_handler.py
echo ""

# ---- Layer 3a: Alpha 合成 ----
echo "[4/8] Layer 3a: Alpha 合成..."
${PYTHON} 3_signal/04_alpha_score/build_dataset.py
echo ""

# ---- Layer 3b: 模型训练 ----
echo "[5/8] Layer 3b: 模型训练..."
${PYTHON} 3_signal/05_model/train_lgbm.py
echo ""

# ---- Layer 3c: 预测 ----
echo "[6/8] Layer 3c: 预测..."
${PYTHON} 3_signal/05_model/predict_lgbm.py
echo ""

# ---- Layer 4: 策略（美元中性 + Beta 对冲） ----
echo "[7/8] Layer 4: 策略（美元中性 + Beta 对冲）..."
if [ -f "$FUTURES_CSV" ]; then
    echo "  futures data found: $FUTURES_CSV"
    ${PYTHON} 4_strategy/06_strategy_top_bottom/generate_positions_with_hedge.py \
        --futures-data "$FUTURES_CSV"
else
    echo "  WARNING: futures data not found at $FUTURES_CSV"
    echo "  falling back to no-hedge mode"
    ${PYTHON} 4_strategy/06_strategy_top_bottom/generate_positions_with_hedge.py \
        --no-hedge
fi
echo ""

# ---- Layer 5: 回测 + 分析 ----
echo "[8/8] Layer 5: 回测 + 分析..."
${PYTHON} 5_backtest/07_backtest/run_backtest.py
${PYTHON} 5_backtest/08_analysis/factor_analysis.py
echo ""

echo "===== Pipeline (With Beta Hedge) Complete: $(date) ====="
