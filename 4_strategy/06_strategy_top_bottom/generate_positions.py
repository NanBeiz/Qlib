#!/usr/bin/env python3
"""Layer 4: 策略 —— 将信号转换为目标持仓。

读取预测信号 CSV，在月末调仓日做多 topk、做空 bottomk，
生成日频持仓 CSV。

Usage:
    python 4_strategy/06_strategy_top_bottom/generate_positions.py \\
        --predictions outputs/predictions/predictions_lgbm.csv \\
        --output outputs/positions/positions_lgbm.csv

输出:
    outputs/positions/positions_lgbm.csv  (datetime, instrument, weight)
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

logger = get_module_logger(__name__)


def get_month_end_dates(
    trading_calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """使用 Qlib Calendar API 获取月末交易日列表。"""
    start = trading_calendar[0]
    end = trading_calendar[-1]

    # 逐月获取每个月的最后一个交易日
    dates = []
    current = pd.Timestamp(year=start.year, month=start.month, day=1)
    while current <= end:
        year = current.year
        month = current.month
        if month == 12:
            next_month = pd.Timestamp(year=year + 1, month=1, day=1)
        else:
            next_month = pd.Timestamp(year=year, month=month + 1, day=1)

        try:
            cal = D.calendar(start_time=current, end_time=next_month, freq="day")
            if len(cal) > 0:
                last_day = cal[-1]
                if last_day in trading_calendar:
                    dates.append(last_day)
        except Exception:
            pass

        # 移到下个月
        current = next_month

    return pd.DatetimeIndex(sorted(set(dates)))


def generate_positions(
    predictions: pd.DataFrame,
    topk: int,
    bottomk: int,
    alpha_col: str = "ALPHA",
) -> pd.DataFrame:
    """生成日频持仓。

    Parameters
    ----------
    predictions : pd.DataFrame
        columns: datetime, instrument, ALPHA
    topk : int
        做多数量。
    bottomk : int
        做空数量。
    alpha_col : str
        信号列名。

    Returns
    -------
    pd.DataFrame
        columns: datetime, instrument, weight
    """
    trading_days = pd.DatetimeIndex(sorted(predictions["datetime"].unique()))
    rebalance_dates = get_month_end_dates(trading_days)
    logger.info(
        "trading days: %d, rebalance dates: %d",
        len(trading_days), len(rebalance_dates),
    )

    # 构建 alpha 查找表: date → {instrument: alpha}
    alpha_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for dt, grp in predictions.groupby("datetime", sort=False):
        day_dict = {}
        for _, row in grp.iterrows():
            if pd.notna(row[alpha_col]):
                day_dict[str(row["instrument"])] = float(row[alpha_col])
        if day_dict:
            alpha_lookup[pd.Timestamp(dt)] = day_dict

    # 调仓日选股
    rebalance_set = set(rebalance_dates)

    # 分配调仓日到每个交易日
    rebalance_map = pd.Series(rebalance_dates, index=rebalance_dates)
    rebalance_map = rebalance_map.reindex(trading_days, method="ffill")

    all_positions: list[dict] = []
    last_weights: dict[str, float] = {}
    last_rebalance_date: Optional[pd.Timestamp] = None
    position_cache: dict[pd.Timestamp, dict[str, float]] = {}

    for day in trading_days:
        rb_date = rebalance_map.get(day)
        if rb_date is None or pd.isna(rb_date):
            continue

        if day in rebalance_set:
            # 调仓日：重新选股
            rb_date_key = pd.Timestamp(rb_date) if not isinstance(rb_date, pd.Timestamp) else rb_date
            if rb_date_key not in position_cache:
                day_alpha = alpha_lookup.get(rb_date_key, {})
                if not day_alpha:
                    continue

                sorted_stocks = sorted(day_alpha.items(), key=lambda x: x[1], reverse=True)
                long_stocks = [s for s, _ in sorted_stocks[:topk]]
                short_candidates = sorted_stocks[-bottomk:]
                long_set = set(long_stocks)
                short_stocks = [s for s, _ in short_candidates if s not in long_set]

                weights: dict[str, float] = {}
                for s in long_stocks:
                    weights[s] = 1.0 / topk
                for s in short_stocks:
                    weights[s] = -1.0 / bottomk
                position_cache[rb_date_key] = weights

            last_weights = position_cache.get(rb_date_key, {})
            last_rebalance_date = rb_date_key
        else:
            # 非调仓日：沿用上期持仓（不漂移，直接复制权重）
            pass  # last_weights stays

        for inst, w in last_weights.items():
            all_positions.append({
                "datetime": day,
                "instrument": inst,
                "weight": w,
            })

    if not all_positions:
        logger.warning("no positions generated")
        return pd.DataFrame(columns=["datetime", "instrument", "weight"])

    positions = pd.DataFrame(all_positions).sort_values(["datetime", "instrument"]).reset_index(drop=True)
    logger.info(
        "positions: %d rows, %d instruments, date range %s ~ %s",
        len(positions),
        positions["instrument"].nunique(),
        positions["datetime"].min().date(),
        positions["datetime"].max().date(),
    )

    # 验证：每个调仓日权重和 ≈ 0
    for rb_date, weights in position_cache.items():
        total_w = sum(weights.values())
        abs_w = sum(abs(w) for w in weights.values())
        n_long = sum(1 for w in weights.values() if w > 0)
        n_short = sum(1 for w in weights.values() if w < 0)
        logger.debug(
            "  %s: long=%d, short=%d, sum_w=%.6f, |w|_sum=%.4f",
            rb_date.date(), n_long, n_short, total_w, abs_w,
        )

    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 4: 策略")
    parser.add_argument(
        "--predictions", default=None, help="预测信号 CSV 路径"
    )
    parser.add_argument(
        "--config", default="configs/strategies/top_bottom_neutral.yaml", help="策略配置"
    )
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置"
    )
    parser.add_argument(
        "--output", default=None, help="输出路径"
    )
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    qlib.init(provider_uri=common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
              region=common_cfg.get("region", "cn"))

    # 加载策略配置
    strategy_path = root / args.config
    with open(strategy_path, "r", encoding="utf-8") as f:
        strategy_cfg = yaml.safe_load(f) or {}
    kwargs = strategy_cfg.get("kwargs", {})
    topk = kwargs.get("topk", 50)
    bottomk = kwargs.get("bottomk", 50)

    # 加载预测
    pred_path = args.predictions or str(root / "outputs" / "predictions" / "predictions_lgbm.csv")
    logger.info("loading predictions from %s...", pred_path)
    predictions = read_csv(pred_path)

    # 生成持仓
    positions = generate_positions(predictions, topk, bottomk)

    # 输出
    out_path = args.output or str(root / "outputs" / "positions" / "positions_lgbm.csv")
    write_csv(positions, out_path)
    logger.info("positions exported: %s", out_path)
    logger.info("Layer 4 complete")


if __name__ == "__main__":
    main()
