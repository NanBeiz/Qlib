#!/usr/bin/env python3
"""风险模型构建 CLI。

从 Qlib 加载个股收益 → 估计协方差矩阵 → 计算组合风险指标 → 输出报告。

Usage:
    python 3_signal/riskmodel/build_riskmodel.py
    python 3_signal/riskmodel/build_riskmodel.py --method structured --num-factors 10
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

# 支持脚本运行和模块导入两种模式
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from covariance import PortfolioCovEstimator  # noqa: E402
from risk_metrics import (  # noqa: E402
    compute_cvar,
    compute_portfolio_volatility,
    compute_risk_decomposition,
    compute_var,
    compute_variance_decomposition,
)

logger = get_module_logger(__name__)


# ── Qlib 防御式数据加载 ──────────────────────────────────────────

def load_stock_returns(
    instruments: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """用 Qlib D.features 加载个股日收益矩阵 (date × instrument)。"""
    expr = "Ref($close, -1)/$close - 1"
    raw = D.features(
        instruments=instruments,
        fields=[expr],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    raw = raw.reset_index()
    assert {"instrument", "datetime"}.issubset(
        set(raw.columns)
    ), f"unexpected columns from D.features: {raw.columns.tolist()}"
    ret_col = raw.columns[-1]
    result = raw.pivot(index="datetime", columns="instrument", values=ret_col)
    logger.info(
        "loaded stock returns: %d dates × %d instruments",
        len(result), len(result.columns),
    )
    return result


# ── 持仓加载 ─────────────────────────────────────────────────────

def load_position_snapshots(
    positions_csv: str,
) -> dict[pd.Timestamp, dict[str, float]]:
    """加载持仓 CSV，提取每个调仓日的权重。

    返回 {rebalance_date: {instrument: weight}}。
    """
    df = read_csv(positions_csv)
    if df.empty:
        logger.warning("positions CSV is empty")
        return {}

    dates = sorted(df["datetime"].unique())
    snapshots: dict[pd.Timestamp, dict[str, float]] = {}

    for i, d in enumerate(dates):
        d = pd.Timestamp(d)
        day_pos = df[df["datetime"] == d]
        weights = dict(zip(
            day_pos["instrument"].astype(str),
            day_pos["weight"].astype(float),
        ))
        if not weights:
            continue
        if i == 0:
            snapshots[d] = weights
            continue
        prev = pd.Timestamp(dates[i - 1])
        prev_weights = dict(zip(
            df[df["datetime"] == prev]["instrument"].astype(str),
            df[df["datetime"] == prev]["weight"].astype(float),
        ))
        if weights != prev_weights:
            snapshots[d] = weights

    logger.info("loaded %d position snapshots", len(snapshots))
    return snapshots


# ── 主流程 ────────────────────────────────────────────────────────

def build_risk_report(
    position_snapshots: dict[pd.Timestamp, dict[str, float]],
    stock_returns: pd.DataFrame,
    cov_estimator: PortfolioCovEstimator,
    lookback: int = 252,
    min_periods: int = 60,
    var_confidence: float = 0.95,
) -> pd.DataFrame:
    """对每个调仓日生成风险指标。

    返回
    ----
    pd.DataFrame
        每行一个调仓日，列为风险指标。
    """
    if stock_returns.empty or not position_snapshots:
        return pd.DataFrame()

    all_dates = sorted(stock_returns.index)
    records = []

    for rb_date in sorted(position_snapshots.keys()):
        weights = position_snapshots[rb_date]
        if not weights:
            continue

        # 取调仓日之前的回看窗口收益
        end_idx = all_dates.searchsorted(rb_date)
        if isinstance(end_idx, np.ndarray):
            end_idx = end_idx[0] if len(end_idx) > 0 else 0
        start_idx = max(0, end_idx - lookback)
        window_dates = all_dates[start_idx:end_idx]

        if len(window_dates) < min_periods:
            logger.debug("rb_date=%s: insufficient data (%d < %d)", rb_date.date(), len(window_dates), min_periods)
            continue

        window_returns = stock_returns.loc[window_dates]
        # 只保留有足够数据的股票
        valid_stocks = window_returns.dropna(axis=1, thresh=min_periods).columns
        if len(valid_stocks) < 10:
            continue
        window_returns = window_returns[valid_stocks]

        # 截面当天持仓中存在的股票
        stocks_in_pos = [s for s in weights if s in valid_stocks]
        if len(stocks_in_pos) < 10:
            continue

        # 协方差估计（只对持仓股票）
        try:
            cov = cov_estimator.estimate(window_returns[stocks_in_pos])
        except Exception as e:
            logger.warning("cov estimation failed for %s: %s", rb_date.date(), e)
            continue

        if cov.empty:
            continue

        filtered_weights = {s: weights[s] for s in stocks_in_pos}

        # 风险指标
        vol = compute_portfolio_volatility(filtered_weights, cov)
        var_val = compute_var(filtered_weights, cov, confidence=var_confidence)
        cvar_val = compute_cvar(filtered_weights, cov, confidence=var_confidence)

        # 尝试因子分解
        decomp = cov_estimator.estimate_with_decomposition(window_returns[stocks_in_pos])
        factor_pct = 0.0
        specific_pct = 0.0
        if decomp["factors"] is not None and decomp["factor_cov"] is not None and decomp["specific_vars"] is not None:
            var_decomp = compute_variance_decomposition(
                filtered_weights,
                decomp["factors"],
                decomp["factor_cov"],
                decomp["specific_vars"],
            )
            factor_pct = var_decomp["factor_pct"]
            specific_pct = var_decomp["specific_pct"]

        records.append({
            "datetime": rb_date,
            "portfolio_vol": round(vol, 6),
            "var_95": round(var_val, 6),
            "cvar_95": round(cvar_val, 6),
            "factor_risk_pct": round(factor_pct, 4),
            "specific_risk_pct": round(specific_pct, 4),
            "n_stocks": len(stocks_in_pos),
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="风险模型构建")
    parser.add_argument("--config", default="configs/riskmodel/default.yaml", help="风险模型配置")
    parser.add_argument("--common-config", default="configs/_common.yaml", help="全局配置")
    parser.add_argument("--positions", default=None, help="持仓 CSV 路径")
    parser.add_argument("--output-dir", default="outputs/riskmodel", help="输出目录")
    parser.add_argument("--method", default=None, help="协方差估计方法: shrink | structured | poet")
    parser.add_argument("--lookback", type=int, default=None, help="回看窗口（交易日）")
    parser.add_argument("--sample", type=int, default=1, help="每 N 个调仓日取一次协方差矩阵")
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    qlib.init(
        provider_uri=common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
        region=common_cfg.get("region", "cn"),
    )

    # 加载风险模型配置
    config_path = root / args.config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            rm_cfg = yaml.safe_load(f) or {}
        rm_cfg = rm_cfg.get("riskmodel", rm_cfg)
    else:
        rm_cfg = {}

    method = args.method or rm_cfg.get("method", "shrink")
    estimator_kwargs = rm_cfg.get("estimator_kwargs", {}).copy()
    lookback = args.lookback or rm_cfg.get("lookback", 252)
    min_periods = rm_cfg.get("min_periods", 60)
    var_confidence = rm_cfg.get("var_confidence", 0.95)

    # CLI 额外参数覆盖
    if args.method == "structured":
        estimator_kwargs.setdefault("factor_model", "pca")
        estimator_kwargs.setdefault("num_factors", 10)

    test_start = common_cfg["test_start"]
    test_end = common_cfg["test_end"]
    train_start = common_cfg.get("train_start", test_start)

    cov_estimator = PortfolioCovEstimator(method=method, estimator_kwargs=estimator_kwargs)

    # 加载持仓
    pos_csv = args.positions or str(root / "outputs" / "positions" / "positions_lgbm.csv")
    logger.info("loading positions from %s...", pos_csv)
    snapshots = load_position_snapshots(pos_csv)
    if not snapshots:
        logger.warning("no position snapshots found")
        return

    # 收集所有出现过的股票
    all_instruments = sorted(set(
        inst for w in snapshots.values() for inst in w
    ))
    logger.info("total instruments: %d", len(all_instruments))

    # Qlib 加载收益
    logger.info("loading stock returns via Qlib (%s ~ %s)...", train_start, test_end)
    stock_returns = load_stock_returns(all_instruments, train_start, test_end)
    if stock_returns.empty:
        logger.warning("stock returns empty, cannot build risk model")
        return

    # 生成风险报告
    logger.info("building risk report with method=%s, lookback=%d...", method, lookback)
    report = build_risk_report(
        position_snapshots=snapshots,
        stock_returns=stock_returns,
        cov_estimator=cov_estimator,
        lookback=lookback,
        min_periods=min_periods,
        var_confidence=var_confidence,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not report.empty:
        write_csv(report, str(out_dir / "risk_report.csv"))
        logger.info("risk report exported: %s", out_dir / "risk_report.csv")

        # 终端摘要
        print("\n" + "=" * 56)
        print("  Risk Model Summary")
        print("=" * 56)
        print(f"  Method:            {method}")
        print(f"  Lookback:          {lookback} days")
        print(f"  Report dates:      {len(report)}")
        print(f"  Mean Portfolio Vol: {report['portfolio_vol'].mean():.4f}")
        print(f"  Mean VaR(95%):     {report['var_95'].mean():.4f}")
        print(f"  Mean CVaR(95%):    {report['cvar_95'].mean():.4f}")
        if report["factor_risk_pct"].mean() > 0:
            print(f"  Mean Factor Risk%: {report['factor_risk_pct'].mean():.2%}")
            print(f"  Mean Specific Risk%: {report['specific_risk_pct'].mean():.2%}")
        print("=" * 56)
    else:
        logger.warning("risk report empty, no output written")

    # 输出样本协方差矩阵
    sample_step = max(1, args.sample)
    sampled_dates = sorted(snapshots.keys())[::sample_step]
    for i, rb_date in enumerate(sampled_dates[:5]):
        weights = snapshots[rb_date]
        end_idx = stock_returns.index.searchsorted(rb_date)
        if isinstance(end_idx, np.ndarray):
            end_idx = end_idx[0] if len(end_idx) > 0 else 0
        start_idx = max(0, end_idx - lookback)
        window_dates = stock_returns.index[start_idx:end_idx]
        if len(window_dates) < min_periods:
            continue
        window_returns = stock_returns.loc[window_dates]
        stocks = [s for s in weights if s in window_returns.columns]
        if len(stocks) < 10:
            continue
        try:
            cov = cov_estimator.estimate(window_returns[stocks])
            cov.to_csv(out_dir / f"cov_matrix_{rb_date.strftime('%Y%m%d')}.csv", float_format="%.10f")
        except Exception:
            pass

    logger.info("Risk model build complete")


if __name__ == "__main__":
    main()
