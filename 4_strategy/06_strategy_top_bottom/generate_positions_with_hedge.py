#!/usr/bin/env python3
"""Layer 4 带 Beta 对冲的策略入口（编排层）。

不修改 generate_positions.py。调用纯股票持仓生成 + portfolio_beta +
hedge_manager，输出两份 CSV：

- outputs/positions/positions_lgbm.csv  (股票持仓)
- outputs/positions/hedge_lgbm.csv      (期货对冲覆盖层)

Usage:
    # 仅股票持仓（等价于 generate_positions.py）
    python 4_strategy/06_strategy_top_bottom/generate_positions_with_hedge.py --no-hedge

    # 带 Beta 对冲（需期货 CSV）
    python 4_strategy/06_strategy_top_bottom/generate_positions_with_hedge.py \\
        --futures-data outputs/data/futures_csi500.csv
"""

from __future__ import annotations

import argparse
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

logger = get_module_logger(__name__)


# ── Qlib 防御式数据加载 ──────────────────────────────────────────

def load_stock_returns(
    instruments: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """用 Qlib D.features 加载个股日收益矩阵 (date x instrument)。

    防御式处理：不假设 MultiIndex 顺序、不硬编码列名。
    """
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
        "loaded stock returns: %d dates x %d instruments (%s ~ %s)",
        len(result),
        len(result.columns),
        str(result.index[0])[:10] if len(result) > 0 else "empty",
        str(result.index[-1])[:10] if len(result) > 0 else "empty",
    )
    return result


def load_benchmark_returns(
    benchmark_code: str,
    start_time: str,
    end_time: str,
) -> pd.Series:
    """用 Qlib D.features 加载基准日收益 Series。

    对冲 Beta 基准: CSI500 SH000905（策略池基准）。
    回测展示基准:   CSI300 SH000300（独立，仅绩效对比）。
    """
    expr = "Ref($close, -1)/$close - 1"
    raw = D.features(
        instruments=[benchmark_code],
        fields=[expr],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    col = raw.columns[0]
    series = raw[col]
    if isinstance(series.index, pd.MultiIndex):
        series = series.reset_index().set_index("datetime")[col]
    result = series.sort_index()
    logger.info(
        "loaded benchmark returns [%s]: %d dates (%s ~ %s)",
        benchmark_code,
        len(result),
        str(result.index[0])[:10] if len(result) > 0 else "empty",
        str(result.index[-1])[:10] if len(result) > 0 else "empty",
    )
    return result


# ── position_cache 重建 ──────────────────────────────────────────

def build_position_cache(
    positions: pd.DataFrame,
) -> dict[pd.Timestamp, dict[str, float]]:
    """从持仓 DataFrame 重建调仓日位置缓存。

    通过比较相邻日的持仓变化检测调仓日。
    """
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

    logger.info(
        "position_cache rebuilt: %d rebalance dates out of %d trading days",
        len(cache), len(dates),
    )
    return cache


# ── Hedge overlay 生成 ───────────────────────────────────────────

def generate_hedge_overlay(
    position_cache: dict[pd.Timestamp, dict[str, float]],
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    futures_provider: FuturesDataProvider,
    hedge_config: dict,
    portfolio_notional: float = 100_000_000,
) -> pd.DataFrame:
    """为所有调仓日生成期货对冲覆盖层。

    columns: datetime, instrument, contracts, futures_symbol,
             futures_price, net_beta, hedge_notional, required_margin, action
    action in {rebalance, carry, skip_no_data, skip_threshold}
    """
    hedge_symbol = hedge_config.get("hedge_symbol", "IC")
    hedge_mode = hedge_config.get("hedge_mode", "full")
    hedge_ratio_target = hedge_config.get("hedge_ratio_target", 1.0)
    futures_beta = hedge_config.get("futures_beta", 1.0)
    min_contracts = hedge_config.get("min_contracts", 0.5)
    max_hedge_ratio = hedge_config.get("max_hedge_ratio", 1.5)
    beta_window = hedge_config.get("beta_window", 252)
    beta_min_periods = hedge_config.get("beta_min_periods", 60)

    manager = HedgeManager(
        futures_provider=futures_provider,
        hedge_symbol=hedge_symbol,
        hedge_mode=hedge_mode,
        hedge_ratio_target=hedge_ratio_target,
        futures_beta=futures_beta,
        min_contracts=min_contracts,
        max_hedge_ratio=max_hedge_ratio,
    )

    # Beta 估计 + shift(1) 防未来函数
    # 调仓日 t 使用截至 t-1 的 Beta 估计，不会包含当日收益信息
    stock_betas = estimate_stock_betas(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        window=beta_window,
        min_periods=beta_min_periods,
    ).shift(1)

    if stock_betas.empty:
        logger.warning("stock betas empty, no hedge generated")
        return pd.DataFrame()

    all_dates = sorted(stock_returns.index)
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
            net_beta = ls_betas["net_beta"]

            result = manager.compute_hedge(
                date=d,
                net_beta=net_beta,
                portfolio_value=portfolio_notional,
                long_exposure=ls_betas["long_exposure"],
                short_exposure=ls_betas["short_exposure"],
            )

            if result is None:
                records.append(_make_skip(d, hedge_symbol, "skip_no_data"))
                last_hedge = None
                continue

            if abs(result.contracts) < min_contracts:
                records.append(_make_skip(d, hedge_symbol, "skip_threshold"))
                last_hedge = None
                continue

            row = {
                "datetime": d,
                "instrument": hedge_symbol,
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
                row = {**last_hedge, "datetime": d, "action": "carry"}
                records.append(row)

    if not records:
        return pd.DataFrame()

    hedge_df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
    logger.info(
        "hedge overlay: %d rows, actions: %s",
        len(hedge_df),
        hedge_df["action"].value_counts().to_dict(),
    )
    return hedge_df


def _nearest_date(
    index: pd.DatetimeIndex,
    target: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """找 index 中 <= target 的最近日期。"""
    earlier = index[index <= target]
    if len(earlier) == 0:
        return None
    return earlier[-1]


def _make_skip(date: pd.Timestamp, symbol: str, action: str) -> dict:
    return {
        "datetime": date,
        "instrument": symbol,
        "contracts": 0.0,
        "futures_symbol": symbol,
        "futures_price": 0.0,
        "net_beta": 0.0,
        "hedge_notional": 0.0,
        "required_margin": 0.0,
        "action": action,
    }


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 4: 策略（可选 Beta 对冲）")
    parser.add_argument("--predictions", default=None, help="预测信号 CSV 路径")
    parser.add_argument(
        "--config", default="configs/strategies/top_bottom_hedged.yaml",
        help="策略配置"
    )
    parser.add_argument("--common-config", default="configs/_common.yaml", help="全局配置")
    parser.add_argument("--output", default=None, help="股票持仓输出路径")
    parser.add_argument("--hedge-output", default=None, help="对冲结果输出路径")
    parser.add_argument(
        "--futures-data", default=None,
        help="期货数据 CSV 路径（提供即启用对冲）"
    )
    parser.add_argument("--no-hedge", action="store_true", help="强制禁用对冲")
    parser.add_argument("--hedge-symbol", default="IC", help="对冲合约品种代码")
    parser.add_argument(
        "--hedge-mode", default="full", choices=["full", "partial"],
        help="对冲模式"
    )
    parser.add_argument(
        "--hedge-ratio", type=float, default=1.0,
        help="partial 模式对冲比例"
    )
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    qlib.init(
        provider_uri=common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
        region=common_cfg.get("region", "cn"),
    )

    # 加载策略配置
    strategy_path = root / args.config
    if strategy_path.exists():
        with open(strategy_path, "r", encoding="utf-8") as f:
            strategy_cfg = yaml.safe_load(f) or {}
        kwargs = strategy_cfg.get("kwargs", {})
        hedge_cfg = strategy_cfg.get("hedging", {})
    else:
        # 向后兼容：fallback 到旧配置
        old_cfg = root / "configs" / "strategies" / "top_bottom_neutral.yaml"
        with open(old_cfg, "r", encoding="utf-8") as f:
            strategy_cfg = yaml.safe_load(f) or {}
        kwargs = strategy_cfg.get("kwargs", {})
        hedge_cfg = {}

    topk = kwargs.get("topk", 50)
    bottomk = kwargs.get("bottomk", 50)
    test_start = common_cfg["test_start"]
    test_end = common_cfg["test_end"]
    train_start = common_cfg.get("train_start", test_start)

    # 加载预测
    pred_path = args.predictions or str(
        root / "outputs" / "predictions" / "predictions_lgbm.csv"
    )
    logger.info("loading predictions from %s...", pred_path)
    predictions = read_csv(pred_path)

    # 1. 纯股票持仓
    positions_df = generate_positions(predictions, topk, bottomk)

    pos_out = args.output or str(root / "outputs" / "positions" / "positions_lgbm.csv")
    Path(pos_out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(positions_df, pos_out)
    logger.info("positions exported: %s", pos_out)

    # 2. 对冲
    hedge_enabled = not args.no_hedge
    futures_data_path = args.futures_data or hedge_cfg.get(
        "futures_data_path",
        str(root / "outputs" / "data" / "futures_csi500.csv"),
    )

    if not hedge_enabled:
        logger.info("hedging disabled, stopping here")
        logger.info("Layer 4 (with hedge) complete")
        return

    futures_provider = CsvFuturesDataProvider(futures_data_path)
    all_instruments = sorted(positions_df["instrument"].unique())

    # Qlib 防御式加载收益
    logger.info("loading stock returns via Qlib...")
    stock_returns = load_stock_returns(all_instruments, train_start, test_end)

    logger.info("loading benchmark returns via Qlib (CSI500 SH000905)...")
    benchmark_returns = load_benchmark_returns("SH000905", train_start, test_end)

    if stock_returns.empty or len(stock_returns.columns) == 0:
        logger.warning("stock returns empty, skipping hedge")
        logger.info("Layer 4 (with hedge) complete -- no hedge generated")
        return

    # CLI 参数覆盖 YAML 配置
    hedge_cfg["hedge_symbol"] = args.hedge_symbol
    hedge_cfg["hedge_mode"] = args.hedge_mode
    hedge_cfg["hedge_ratio_target"] = args.hedge_ratio
    portfolio_notional = hedge_cfg.get("portfolio_notional", 100_000_000)

    position_cache = build_position_cache(positions_df)
    if not position_cache:
        logger.warning("position_cache empty, skipping hedge")
        logger.info("Layer 4 (with hedge) complete -- no hedge generated")
        return

    hedge_df = generate_hedge_overlay(
        position_cache=position_cache,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        futures_provider=futures_provider,
        hedge_config=hedge_cfg,
        portfolio_notional=portfolio_notional,
    )

    hedge_out = args.hedge_output or str(
        root / "outputs" / "positions" / "hedge_lgbm.csv"
    )
    Path(hedge_out).parent.mkdir(parents=True, exist_ok=True)
    if not hedge_df.empty:
        write_csv(hedge_df, hedge_out)
        logger.info("hedge overlay exported: %s", hedge_out)
        rebalance = hedge_df[hedge_df["action"] == "rebalance"]
        if len(rebalance) > 0:
            avg_c = rebalance["contracts"].abs().mean()
            avg_b = rebalance["net_beta"].abs().mean()
            logger.info(
                "hedge summary: %d rebalance hedges, avg |contracts|=%.1f, avg |net_beta|=%.4f",
                len(rebalance), avg_c, avg_b,
            )
    else:
        logger.info("hedge overlay empty, no CSV written")

    logger.info("Layer 4 (with hedge) complete")


if __name__ == "__main__":
    main()
