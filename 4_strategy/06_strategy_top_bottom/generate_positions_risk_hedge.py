#!/usr/bin/env python3
"""Layer 4 风险优化 + Beta 对冲策略入口（整合版）。

在 alpha 选股基础上：
1. 用风险模型协方差矩阵 + 优化器替换等权配置
2. 在风险权重之上叠加 IC 期货 Beta 对冲

输出：
- outputs/positions/positions_risk.csv      (风险优化股票持仓)
- outputs/positions/hedge_risk.csv          (期货对冲覆盖层)

Usage:
    python 4_strategy/06_strategy_top_bottom/generate_positions_risk_hedge.py
    python 4_strategy/06_strategy_top_bottom/generate_positions_risk_hedge.py \\
        --method inv --futures-data outputs/data/futures_csi500.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import qlib
from qlib.data import D
from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_csv, write_csv

from generate_positions import generate_positions
from portfolio_beta import compute_long_short_betas, estimate_stock_betas
from hedge_manager import HedgeManager
from futures_data_interface import CsvFuturesDataProvider, FuturesDataProvider

# riskmodel 模块路径
_RISKMODEL_DIR = str(Path(__file__).resolve().parents[2] / "3_signal")
if _RISKMODEL_DIR not in sys.path:
    sys.path.insert(0, _RISKMODEL_DIR)

from riskmodel.covariance import PortfolioCovEstimator  # noqa: E402
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
        instruments=instruments, fields=[expr],
        start_time=start_time, end_time=end_time, freq="day",
    )
    raw = raw.reset_index()
    assert {"instrument", "datetime"}.issubset(set(raw.columns))
    ret_col = raw.columns[-1]
    return raw.pivot(index="datetime", columns="instrument", values=ret_col)


def load_benchmark_returns(
    benchmark_code: str,
    start_time: str,
    end_time: str,
) -> pd.Series:
    """Qlib D.features 加载基准日收益 Series（Beta 对冲用 CSI500 SH000905）。"""
    expr = "Ref($close, -1)/$close - 1"
    raw = D.features(
        instruments=[benchmark_code], fields=[expr],
        start_time=start_time, end_time=end_time, freq="day",
    )
    col = raw.columns[0]
    series = raw[col]
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index().set_index("datetime")[col]
    return series.sort_index()


def build_position_cache(
    positions: pd.DataFrame,
) -> dict[pd.Timestamp, dict[str, float]]:
    """从持仓 DataFrame 重建调仓日位置缓存。"""
    dates = sorted(positions["datetime"].unique())
    cache: dict[pd.Timestamp, dict[str, float]] = {}
    for i, date in enumerate(dates):
        d = pd.Timestamp(date)
        day_pos = positions[positions["datetime"] == d]
        day_weights = dict(zip(
            day_pos["instrument"].astype(str),
            day_pos["weight"].astype(float),
        ))
        if not day_weights:
            continue
        if i == 0:
            cache[d] = day_weights
            continue
        prev_d = pd.Timestamp(dates[i - 1])
        prev_pos = positions[positions["datetime"] == prev_d]
        prev_weights = dict(zip(
            prev_pos["instrument"].astype(str),
            prev_pos["weight"].astype(float),
        ))
        if day_weights != prev_weights:
            cache[d] = day_weights
    return cache


def generate_hedge_overlay(
    position_cache: dict[pd.Timestamp, dict[str, float]],
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    futures_provider: FuturesDataProvider,
    hedge_config: dict,
    portfolio_notional: float = 100_000_000,
) -> pd.DataFrame:
    """为所有调仓日生成期货对冲覆盖层。"""
    hedge_symbol = hedge_config.get("hedge_symbol", "IC")
    manager = HedgeManager(
        futures_provider=futures_provider,
        hedge_symbol=hedge_symbol,
        hedge_mode=hedge_config.get("hedge_mode", "full"),
        hedge_ratio_target=hedge_config.get("hedge_ratio_target", 1.0),
        futures_beta=hedge_config.get("futures_beta", 1.0),
        min_contracts=hedge_config.get("min_contracts", 0.5),
        max_hedge_ratio=hedge_config.get("max_hedge_ratio", 1.5),
    )
    stock_betas = estimate_stock_betas(
        stock_returns=stock_returns, benchmark_returns=benchmark_returns,
        window=hedge_config.get("beta_window", 252),
        min_periods=hedge_config.get("beta_min_periods", 60),
    ).shift(1)
    if stock_betas.empty:
        return pd.DataFrame()

    all_dates = pd.DatetimeIndex(sorted(stock_returns.index))
    rebalance_dates = sorted(position_cache.keys())
    rebalance_set = set(rebalance_dates)
    records: list[dict] = []
    last_hedge: Optional[dict] = None

    for date in all_dates:
        d = pd.Timestamp(date)
        if d in rebalance_set:
            weights = position_cache[d]
            if not weights:
                records.append(_make_skip(d, hedge_symbol, "skip_no_data"))
                last_hedge = None
                continue
            beta_date = _nearest_date(stock_betas.index, d)
            if beta_date is None:
                records.append(_make_skip(d, hedge_symbol, "skip_no_data"))
                last_hedge = None
                continue
            date_betas = stock_betas.loc[beta_date]
            if date_betas.isna().all():
                records.append(_make_skip(d, hedge_symbol, "skip_no_data"))
                last_hedge = None
                continue
            ls_betas = compute_long_short_betas(weights, date_betas)
            result = manager.compute_hedge(
                date=d, net_beta=ls_betas["net_beta"],
                portfolio_value=portfolio_notional,
                long_exposure=ls_betas["long_exposure"],
                short_exposure=ls_betas["short_exposure"],
            )
            if result is None:
                records.append(_make_skip(d, hedge_symbol, "skip_no_data"))
                last_hedge = None
                continue
            if abs(result.contracts) < hedge_config.get("min_contracts", 0.5):
                records.append(_make_skip(d, hedge_symbol, "skip_threshold"))
                last_hedge = None
                continue
            row = {
                "datetime": d, "instrument": hedge_symbol,
                "contracts": result.contracts,
                "futures_symbol": result.futures_symbol,
                "futures_price": result.futures_price,
                "net_beta": result.net_beta,
                "hedge_notional": result.hedge_notional,
                "required_margin": result.required_margin,
                "action": "rebalance",
            }
            records.append(row)
            last_hedge = row
        else:
            if last_hedge is not None:
                records.append({**last_hedge, "datetime": d, "action": "carry"})
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)


def _nearest_date(index: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    earlier = index[index <= target]
    return earlier[-1] if len(earlier) > 0 else None


def _make_skip(date: pd.Timestamp, symbol: str, action: str) -> dict:
    return {
        "datetime": date, "instrument": symbol,
        "contracts": 0.0, "futures_symbol": symbol,
        "futures_price": 0.0, "net_beta": 0.0,
        "hedge_notional": 0.0, "required_margin": 0.0, "action": action,
    }


# ── 风险优化持仓（复用 generate_positions_risk.py 逻辑）───────

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
    """生成日频风险优化持仓。"""
    positions_eq = generate_positions(predictions, topk, bottomk, alpha_col)
    if positions_eq.empty:
        return positions_eq

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
        return positions_eq

    all_dates = pd.DatetimeIndex(sorted(stock_returns.index))
    risk_weights: dict[pd.Timestamp, dict[str, float]] = {}

    for rb in rebalance_dates:
        stock_set = list(candidate_stocks[rb].keys())
        long_stocks = [s for s in stock_set if candidate_stocks[rb][s] > 0]
        short_stocks = [s for s in stock_set if candidate_stocks[rb][s] < 0]

        end_idx = all_dates.searchsorted(rb)
        if isinstance(end_idx, np.ndarray):
            end_idx = end_idx[0] if len(end_idx) > 0 else 0
        start_idx = max(0, end_idx - lookback)
        if end_idx - start_idx < min_periods:
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        window_dates = all_dates[start_idx:end_idx]
        window_returns = stock_returns.loc[window_dates]
        valid = window_returns.dropna(axis=1, thresh=min_periods).columns
        all_candidates = long_stocks + short_stocks
        usable = [s for s in all_candidates if s in valid]
        if len(usable) < 10:
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        try:
            cov = cov_estimator.estimate(window_returns[usable])
        except Exception:
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        alpha_scores = {}
        pred_day = predictions[predictions["datetime"] == rb]
        for _, row in pred_day.iterrows():
            inst = str(row["instrument"])
            if inst in usable and pd.notna(row.get(alpha_col)):
                alpha_scores[inst] = float(row[alpha_col])
        if not alpha_scores:
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        opt_weights = optimizer.optimize(
            alpha_scores=pd.Series(alpha_scores),
            cov_matrix=cov,
            topk=len(long_stocks),
            bottomk=len(short_stocks),
        )
        if not opt_weights:
            risk_weights[rb] = {s: candidate_stocks[rb][s] for s in stock_set}
            continue

        has_long = any(w > 0 for w in opt_weights.values())
        has_short = any(w < 0 for w in opt_weights.values())
        if not has_long or not has_short:
            for s, w in candidate_stocks[rb].items():
                if s not in opt_weights:
                    opt_weights[s] = w
        risk_weights[rb] = opt_weights

    # 展开到日频
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
    parser = argparse.ArgumentParser(description="Layer 4: 风险优化 + Beta 对冲")
    parser.add_argument("--predictions", default=None, help="预测信号 CSV")
    parser.add_argument("--config", default="configs/strategies/risk_based.yaml", help="策略配置")
    parser.add_argument("--common-config", default="configs/_common.yaml", help="全局配置")
    parser.add_argument("--output", default=None, help="股票持仓输出路径")
    parser.add_argument("--hedge-output", default=None, help="对冲结果输出路径")
    parser.add_argument("--method", default=None, help="优化方法: inv | rp | gmv | mvo")
    parser.add_argument("--lamb", type=float, default=None, help="MVO 风险厌恶系数")
    parser.add_argument("--cov-method", default=None, help="协方差方法: shrink | structured | poet")
    parser.add_argument("--futures-data", default=None, help="期货数据 CSV（提供则启用对冲）")
    parser.add_argument("--no-hedge", action="store_true", help="禁用 Beta 对冲")
    parser.add_argument("--hedge-symbol", default="IC", help="对冲合约品种代码")
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
        with open(root / "configs" / "strategies" / "top_bottom_neutral.yaml", "r", encoding="utf-8") as f:
            strategy_cfg = yaml.safe_load(f) or {}
        kwargs = strategy_cfg.get("kwargs", {})
        opt_cfg = {}

    topk = kwargs.get("topk", 50)
    bottomk = kwargs.get("bottomk", 50)
    test_start = common_cfg["test_start"]
    test_end = common_cfg["test_end"]
    train_start = common_cfg.get("train_start", test_start)

    method = args.method or opt_cfg.get("method", "inv")
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
    stock_returns = load_stock_returns(all_instruments, train_start, test_end)
    if stock_returns.empty:
        logger.warning("stock returns empty, falling back to equal weight")
        positions = generate_positions(predictions, topk, bottomk)
        write_csv(positions, args.output or str(root / "outputs" / "positions" / "positions_risk.csv"))
        return

    # 1. 风险优化持仓
    method_kwargs = {}
    if method == "mvo":
        method_kwargs["lamb"] = lamb
    optimizer = LongShortRiskOptimizer(method=method, method_kwargs=method_kwargs)
    cov_estimator = PortfolioCovEstimator(method=cov_method)

    positions = generate_risk_positions(
        predictions=predictions, topk=topk, bottomk=bottomk,
        stock_returns=stock_returns, optimizer=optimizer,
        cov_estimator=cov_estimator, lookback=lookback, min_periods=min_periods,
    )

    pos_out = args.output or str(root / "outputs" / "positions" / "positions_risk.csv")
    Path(pos_out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(positions, pos_out)
    logger.info("risk positions exported: %s", pos_out)

    # 2. Beta 对冲
    hedge_enabled = not args.no_hedge
    futures_data_path = args.futures_data or str(root / "outputs" / "data" / "futures_csi500.csv")
    if not hedge_enabled:
        logger.info("hedging disabled")
    else:
        futures_provider = CsvFuturesDataProvider(futures_data_path)
        logger.info("loading benchmark returns via Qlib (CSI500 SH000905)...")
        benchmark_returns = load_benchmark_returns("SH000905", train_start, test_end)

        if benchmark_returns.empty:
            logger.warning("benchmark returns empty, skipping hedge")
        else:
            position_cache = build_position_cache(positions)
            if position_cache:
                hedge_df = generate_hedge_overlay(
                    position_cache=position_cache,
                    stock_returns=stock_returns,
                    benchmark_returns=benchmark_returns,
                    futures_provider=futures_provider,
                    hedge_config={
                        "hedge_symbol": args.hedge_symbol,
                        "hedge_mode": "full",
                        "hedge_ratio_target": 1.0,
                        "futures_beta": 1.0,
                        "min_contracts": 0.5,
                        "max_hedge_ratio": 1.5,
                        "beta_window": 252,
                        "beta_min_periods": 60,
                    },
                )
                hedge_out = args.hedge_output or str(root / "outputs" / "positions" / "hedge_risk.csv")
                if not hedge_df.empty:
                    write_csv(hedge_df, hedge_out)
                    logger.info("hedge overlay exported: %s", hedge_out)

    # 摘要
    if not positions.empty:
        rebalance_dates = sorted(positions["datetime"].unique())
        print("\n" + "=" * 56)
        print("  Risk + Hedge Positions Summary")
        print("=" * 56)
        print(f"  Optimizer:  {method} (cov: {cov_method})")
        print(f"  Beta Hedge: {'on' if hedge_enabled else 'off'}")
        print(f"  Positions:  {len(positions)} rows, {positions['instrument'].nunique()} instruments")
        print(f"  Date range: {str(rebalance_dates[0])[:10]} ~ {str(rebalance_dates[-1])[:10]}")
        first_day = positions[positions["datetime"] == rebalance_dates[0]]
        long_w = first_day[first_day["weight"] > 0]["weight"]
        short_w = first_day[first_day["weight"] < 0]["weight"]
        print(f"  First day:  {len(long_w)} long (max w={long_w.max():.4f}), {len(short_w)} short (max |w|={abs(short_w.min()):.4f})")
        print("=" * 56)

    logger.info("Layer 4 (risk + hedge) complete")


if __name__ == "__main__":
    main()
