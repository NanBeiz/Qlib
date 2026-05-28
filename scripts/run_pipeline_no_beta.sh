#!/bin/bash
# run_pipeline_no_beta.sh —— 纯美元中性，不做 Beta 对冲
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Python 解释器（默认 conda qlib_final，可通过环境变量覆盖）
PYTHON="${PYTHON:-/home/hcl/miniconda3/envs/qlib_final/bin/python}"

echo "===== Pipeline (No Beta Hedge) Start: $(date) ====="
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

# ---- Layer 4: 策略（纯股票，无 Beta 对冲） ----
echo "[7/8] Layer 4: 策略（纯美元中性）..."
${PYTHON} 4_strategy/06_strategy_top_bottom/generate_positions.py
echo ""

# ---- Layer 5: 回测 + 分析 ----
echo "[8/8] Layer 5: 回测 + 分析..."
${PYTHON} 5_backtest/07_backtest/run_backtest.py
${PYTHON} 5_backtest/08_analysis/factor_analysis.py
echo ""

echo "===== Pipeline (No Beta Hedge) Complete: $(date) ====="
