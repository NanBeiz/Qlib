#!/usr/bin/env python3
"""Layer 4 风险约束持仓生成（编排层）。

在 alpha 选股基础上，用风险模型协方差矩阵 + 优化器替换等权配置。

Usage:
    python 4_strategy/06_strategy_top_bottom/generate_positions_risk.py
    python 4_strategy/06_strategy_top_bottom/generate_positions_risk.py --method rp
    python 4_strategy/06_strategy_top_bottom/generate_positions_risk.py --method mvo --lamb 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import qlib
from qlib.data import D
from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_csv, write_csv

from generate_positions import generate_positions

# riskmodel 模块路径
_RISKMODEL_DIR = str(Path(__file__).resolve().parents[2] / "3_signal")
if _RISKMODEL_DIR not in sys.path:
    sys.path.insert(0, _RISKMODEL_DIR)

from riskmodel.covariance import PortfolioCovEstimator  # noqa: E402
from riskmodel.risk_metrics import (  # noqa: E402
    compute_cvar,
    compute_portfolio_volatility,
    compute_var,
)

from risk_optimizer import LongShortRiskOptimizer  # noqa: E402

logger = get_module_logger(__name__)


# ── Qlib 防御式数据加载 ──────────────────────────────────────────

def load_stock_returns(
    instruments: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """Qlib D.features 加载个股日收益矩阵 (date × instrument)。"""
    expr = "Ref($close, -1)/$close - 1"
    raw = D.features(
        instruments=instruments,
        fields=[expr],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    raw = raw.reset_index()
    assert {"instrument", "datetime"}.issubset(set(raw.columns))
    ret_col = raw.columns[-1]
    result = raw.pivot(index="datetime", columns="instrument", values=ret_col)
    logger.info("loaded stock returns: %d dates × %d instruments", len(result), len(result.columns))
    return result


# ── 风险持仓生成 ────────────────────────────────────────────────

def generate_risk_positions(
    predictions: pd.DataFrame,
    topk: int,
    bottomk: int,
    stock_returns: pd.DataFrame,
    optimizer: LongShortRiskOptimizer,
    cov_estimator: PortfolioCovEstimator,
    lookback: int = 252,
    min_periods: int = 60,
    alpha_col: str = "ALPHA",
) -> pd.DataFrame:
    """生成日频风险优化持仓。

    流程：
    1. generate_positions() 选出每个调仓日的 long/short 候选
    2. 对每个调仓日，取 lookback 窗口收益估计协方差
    3. LongShortRiskOptimizer 计算风险权重
    4. 非调仓日延续上期权重（carry-forward）

    返回
    ----
    pd.DataFrame
        columns: datetime, instrument, weight
    """
    # 1. 用原版 generate_positions 的选股逻辑得到候选股票（但不取其等权权重）
    positions_eq = generate_positions(predictions, topk, bottomk, alpha_col)
    if positions_eq.empty:
        return positions_eq

    # 2. 重建调仓日 → 候选股票集合
    dates = sorted(positions_eq["datetime"].unique())
    rebalance_dates: list[pd.Timestamp] = []
    candidate_stocks: dict[pd.Timestamp, dict[str, float]] = {}

    for i, d in enumerate(dates):
        d = pd.Timestamp(d)
        day = positions_eq[positions_eq["datetime"] == d]
        w = dict(zip(day["instrument"].astype(str), day["weight"].astype(float)))
        if i == 0:
            rebalance_dates.append(d)
            candidate_stocks[d] = w
        else:
            prev = pd.Timestamp(dates[i - 1])
            prev_day = positions_eq[positions_eq["datetime"] == prev]
            prev_w = dict(zip(prev_day["instrument"].astype(str), prev_day["weight"].astype(float)))
            if set(w.keys()) != set(prev_w.keys()):
                rebalance_dates.append(d)
                candidate_stocks[d] = w

    if not rebalance_dates:
        logger.warning("no rebalance dates found")
        return positions_eq

    # 3. 对每个调仓日做风险优化
    all_dates = pd.DatetimeIndex(sorted(stock_returns.index))
    rebalance_set = set(rebalance_dates)
    risk_weights: dict[pd.Timestamp, dict[str, float]] = {}

    for rb in rebalance_dates:
        # 提取候选股票
        stock_set = list(candidate_stocks[rb].keys())
        long_stocks = [s for s in stock_set if candidate_stocks[rb][s] > 0]
        short_stocks = [s for s in stock_set if candidate_stocks[rb][s] < 0]

        # 回看窗口
        end_idx = all_dates.searchsorted(rb)
        if isinstance(end_idx, np.ndarray):
            end_idx = end_idx[0] if len(end_idx) > 0 else 0
        start_idx = max(0, end_idx - lookback)
        if end_idx - start_idx < min_periods:
            logger.debug("rb=%s: insufficient data", rb.date())
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        window_dates = all_dates[start_idx:end_idx]
        window_returns = stock_returns.loc[window_dates]

        # 只取有足够数据的股票
        valid = window_returns.dropna(axis=1, thresh=min_periods).columns
        all_candidates = long_stocks + short_stocks
        usable = [s for s in all_candidates if s in valid]
        if len(usable) < 10:
            logger.debug("rb=%s: too few usable stocks (%d)", rb.date(), len(usable))
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        # 协方差估计
        try:
            cov = cov_estimator.estimate(window_returns[usable])
        except Exception:
            logger.warning("rb=%s: cov estimation failed", rb.date())
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        # alpha 序列（从 predictions 提取该调仓日的 alpha）
        alpha_scores = {}
        pred_day = predictions[predictions["datetime"] == rb]
        for _, row in pred_day.iterrows():
            inst = str(row["instrument"])
            if inst in usable and pd.notna(row.get(alpha_col)):
                alpha_scores[inst] = float(row[alpha_col])

        if not alpha_scores:
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        alpha_series = pd.Series(alpha_scores)

        # 运行优化
        opt_weights = optimizer.optimize(
            alpha_scores=alpha_series,
            cov_matrix=cov,
            topk=len(long_stocks),
            bottomk=len(short_stocks),
        )

        if not opt_weights:
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        # 如果优化后某条腿为空，从候选等权持仓中补齐
        has_long = any(w > 0 for w in opt_weights.values())
        has_short = any(w < 0 for w in opt_weights.values())
        if not has_long or not has_short:
            for s, w in candidate_stocks[rb].items():
                if s not in opt_weights:
                    opt_weights[s] = w

        risk_weights[rb] = opt_weights
        logger.debug(
            "rb=%s: optimized %d stocks, sum_w=%.6f",
            rb.date(), len(opt_weights), sum(opt_weights.values()),
        )

    # 4. 展开到日频（carry-forward）
    all_rows = []
    last_w: dict[str, float] = {}

    for d in dates:
        d = pd.Timestamp(d)
        if d in risk_weights:
            last_w = risk_weights[d]
        for inst, w in last_w.items():
            all_rows.append({"datetime": d, "instrument": inst, "weight": w})

    if not all_rows:
        return pd.DataFrame(columns=["datetime", "instrument", "weight"])

    result = pd.DataFrame(all_rows).sort_values(["datetime", "instrument"]).reset_index(drop=True)
    logger.info("risk positions: %d rows, %d instruments", len(result), result["instrument"].nunique())
    return result


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 4: 风险约束持仓生成")
    parser.add_argument("--predictions", default=None, help="预测信号 CSV")
    parser.add_argument("--config", default="configs/strategies/risk_based.yaml", help="策略配置")
    parser.add_argument("--common-config", default="configs/_common.yaml", help="全局配置")
    parser.add_argument("--output", default=None, help="持仓输出路径")
    parser.add_argument("--method", default=None, help="优化方法: inv | rp | gmv | mvo")
    parser.add_argument("--lamb", type=float, default=None, help="MVO 风险厌恶系数")
    parser.add_argument("--cov-method", default=None, help="协方差方法: shrink | structured | poet")
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    qlib.init(
        provider_uri=common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
        region=common_cfg.get("region", "cn"),
    )

    # 策略配置
    strategy_path = root / args.config
    if strategy_path.exists():
        with open(strategy_path, "r", encoding="utf-8") as f:
            strategy_cfg = yaml.safe_load(f) or {}
        kwargs = strategy_cfg.get("kwargs", {})
        opt_cfg = strategy_cfg.get("risk_optimizer", {})
    else:
        old_cfg = root / "configs" / "strategies" / "top_bottom_neutral.yaml"
        with open(old_cfg, "r", encoding="utf-8") as f:
            strategy_cfg = yaml.safe_load(f) or {}
        kwargs = strategy_cfg.get("kwargs", {})
        opt_cfg = {}

    topk = kwargs.get("topk", 50)
    bottomk = kwargs.get("bottomk", 50)
    test_start = common_cfg["test_start"]
    test_end = common_cfg["test_end"]
    train_start = common_cfg.get("train_start", test_start)

    # CLI 覆盖
    method = args.method or opt_cfg.get("method", "rp")
    lamb = args.lamb or opt_cfg.get("lamb", 1.0)
    cov_method = args.cov_method or opt_cfg.get("cov_method", "shrink")
    lookback = opt_cfg.get("lookback", 252)
    min_periods = opt_cfg.get("min_periods", 60)

    # 加载预测
    pred_path = args.predictions or str(root / "outputs" / "predictions" / "predictions_lgbm.csv")
    logger.info("loading predictions from %s...", pred_path)
    predictions = read_csv(pred_path)

    all_instruments = sorted(predictions["instrument"].astype(str).unique())

    # Qlib 加载收益
    logger.info("loading stock returns via Qlib...")
    stock_returns = load_stock_returns(all_instruments, train_start, test_end)

    if stock_returns.empty or len(stock_returns.columns) == 0:
        logger.warning("stock returns empty, falling back to equal weight")
        positions = generate_positions(predictions, topk, bottomk)
        write_csv(positions, args.output or str(root / "outputs" / "positions" / "positions_risk.csv"))
        return

    # 初始化组件
    cov_estimator = PortfolioCovEstimator(method=cov_method)
    method_kwargs = {}
    if method == "mvo":
        method_kwargs["lamb"] = lamb
    optimizer = LongShortRiskOptimizer(method=method, method_kwargs=method_kwargs)

    # 生成风险优化持仓
    positions = generate_risk_positions(
        predictions=predictions,
        topk=topk,
        bottomk=bottomk,
        stock_returns=stock_returns,
        optimizer=optimizer,
        cov_estimator=cov_estimator,
        lookback=lookback,
        min_periods=min_periods,
    )

    out_path = args.output or str(root / "outputs" / "positions" / "positions_risk.csv")
    write_csv(positions, out_path)
    logger.info("risk positions exported: %s", out_path)

    # 摘要
    if not positions.empty:
        rebalance_dates = sorted(positions["datetime"].unique())
        print("\n" + "=" * 56)
        print("  Risk-Optimized Positions Summary")
        print("=" * 56)
        print(f"  Optimizer:  {method} (cov: {cov_method})")
        print(f"  Positions:  {len(positions)} rows, {positions['instrument'].nunique()} instruments")
        first, last = rebalance_dates[0], rebalance_dates[-1]
        print(f"  Date range: {str(first)[:10]} ~ {str(last)[:10]}")

        # 有多少调仓日
        n_rb = 0
        for i, d in enumerate(rebalance_dates):
            if i == 0:
                n_rb += 1
            else:
                prev_w = dict(zip(
                    positions[positions["datetime"] == rebalance_dates[i - 1]]["instrument"],
                    positions[positions["datetime"] == rebalance_dates[i - 1]]["weight"],
                ))
                cur_w = dict(zip(
                    positions[positions["datetime"] == d]["instrument"],
                    positions[positions["datetime"] == d]["weight"],
                ))
                if set(prev_w.items()) != set(cur_w.items()):
                    n_rb += 1

        # 检查权重集中度
        first_day = positions[positions["datetime"] == rebalance_dates[0]]
        long_w = first_day[first_day["weight"] > 0]["weight"]
        short_w = first_day[first_day["weight"] < 0]["weight"]
        print(f"  Rebalances: {n_rb}")
        print(f"  First day:  {len(long_w)} long (max w={long_w.max():.4f}), {len(short_w)} short (max |w|={abs(short_w.min()):.4f})")
        print("=" * 56)

    logger.info("Layer 4 (risk-based) complete")


if __name__ == "__main__":
    main()
