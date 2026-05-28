#!/usr/bin/env python3
"""Layer 1: 数据准备 —— 生成 point-in-time 可交易股票池。

使用 Qlib D.list_instruments + D.features 拉取 CSI500 成分股，
过滤 ST 和零成交，输出 (datetime, instrument) 格式的 universe CSV。

Usage:
    python 1_data/01_data_processing/prepare_data.py --config configs/_common.yaml

输出:
    outputs/data/universe_csi500.csv   (datetime, instrument)
    outputs/data/universe_stats.csv    (每日成分股数量，审计用)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import qlib
from qlib.data import D
from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_csv, write_csv

logger = get_module_logger(__name__)


def build_instrument_interval(
    instruments: str, start_time: str, end_time: str
) -> pd.DataFrame:
    """获取 CSI500 成分股的时间区间列表。

    Parameters
    ----------
    instruments : str
        股票池标识，如 "csi500"。
    start_time : str
        起始日期。
    end_time : str
        结束日期。

    Returns
    -------
    pd.DataFrame
        columns: instrument, start_time, end_time
    """
    inst_obj = D.instruments(market=instruments)
    inst_dict = D.list_instruments(
        instruments=inst_obj,
        start_time=start_time,
        end_time=end_time,
        as_list=False,
    )

    rows = []
    for inst, periods in inst_dict.items():
        for s, e in periods:
            rows.append(
                {
                    "instrument": inst,
                    "start_time": pd.Timestamp(s),
                    "end_time": pd.Timestamp(e),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["instrument", "start_time", "end_time"])
    return pd.DataFrame(rows).sort_values(["instrument", "start_time"]).reset_index(drop=True)


def expand_to_daily(
    intervals: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """将成分股时间区间展开为日频。

    Parameters
    ----------
    intervals : pd.DataFrame
        columns: instrument, start_time, end_time。
    trading_calendar : pd.DatetimeIndex
        交易日历。

    Returns
    -------
    pd.DataFrame
        columns: datetime, instrument
    """
    rows = []
    for _, rec in intervals.iterrows():
        mask = (trading_calendar >= rec["start_time"]) & (trading_calendar <= rec["end_time"])
        for d in trading_calendar[mask]:
            rows.append({"datetime": d, "instrument": rec["instrument"]})
    return pd.DataFrame(rows).sort_values(["datetime", "instrument"]).reset_index(drop=True)


def fetch_filter_fields(
    universe_daily: pd.DataFrame,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """拉取 $volume 和 $is_st 字段用于过滤。

    Parameters
    ----------
    universe_daily : pd.DataFrame
        columns: datetime, instrument
    start_time : str
        起始日期。
    end_time : str
        结束日期。

    Returns
    -------
    pd.DataFrame
        含 volume, is_st 列的日频数据。
    """
    instruments = sorted(universe_daily["instrument"].unique())
    logger.info("fetching filter fields for %d instruments...", len(instruments))

    fields = ["$volume", "$is_st"]
    df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )

    if df.index.nlevels == 2:
        df = df.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    df = df.rename(columns={c: c.lstrip("$") for c in df.columns if isinstance(c, str)})

    if "instrument" not in df.columns:
        col_map = {c: "instrument" for c in df.columns if "inst" in str(c).lower()}
        if col_map:
            df = df.rename(columns=col_map)

    df["instrument"] = df["instrument"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def apply_filters(
    universe_daily: pd.DataFrame, filter_data: pd.DataFrame
) -> pd.DataFrame:
    """应用 ST 和成交量过滤。

    Parameters
    ----------
    universe_daily : pd.DataFrame
        columns: datetime, instrument
    filter_data : pd.DataFrame
        含 volume, is_st 列。

    Returns
    -------
    pd.DataFrame
        过滤后的 universe。
    """
    merged = universe_daily.merge(filter_data, on=["datetime", "instrument"], how="left")

    n_before = len(merged)

    # 过滤 ST
    if "is_st" in merged.columns:
        merged = merged[merged["is_st"].fillna(0) == 0]

    # 过滤零成交量
    if "volume" in merged.columns:
        merged = merged[merged["volume"].fillna(0) > 0]

    n_after = len(merged)
    logger.info("filter: %d → %d rows (removed %d)", n_before, n_after, n_before - n_after)

    return merged[["datetime", "instrument"]].reset_index(drop=True)


def compute_stats(universe: pd.DataFrame) -> pd.DataFrame:
    """计算每日成分股数量统计。"""
    stats = universe.groupby("datetime").size().reset_index(name="count")
    stats["datetime"] = stats["datetime"].dt.strftime("%Y-%m-%d")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 1: 数据准备")
    parser.add_argument(
        "--config", default="configs/_common.yaml", help="全局配置文件路径"
    )
    args = parser.parse_args()

    root = get_project_root()
    cfg = load_config(args.config if args.config.startswith("configs") else args.config)

    provider_uri = cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data")
    region = cfg.get("region", "cn")
    market = cfg.get("market", "csi500")

    # 缓存配置
    qlib_kwargs = {}
    if cfg.get("cache_enabled"):
        qlib_kwargs["expression_cache"] = "DiskExpressionCache"
        qlib_kwargs["dataset_cache"] = "DiskDatasetCache"
        cache_dir = cfg.get("cache_dir")
        if cache_dir:
            qlib_kwargs["local_cache_path"] = str(root / cache_dir)

    qlib.init(provider_uri=provider_uri, region=region, **qlib_kwargs)

    train_start = cfg["train_start"]
    test_end = cfg["test_end"]

    # 1. 获取 CSI500 成分股区间
    logger.info("Step 1/4: loading CSI500 instrument intervals...")
    intervals = build_instrument_interval(market, train_start, test_end)
    logger.info("  %d instruments, %d interval records", intervals["instrument"].nunique(), len(intervals))

    # 2. 展开为日频
    logger.info("Step 2/4: expanding to daily frequency...")
    trading_calendar = D.calendar(start_time=train_start, end_time=test_end, freq="day")
    universe_daily = expand_to_daily(intervals, trading_calendar)
    logger.info("  %d daily records", len(universe_daily))

    # 3. 拉取过滤字段
    logger.info("Step 3/4: fetching filter fields...")
    filter_data = fetch_filter_fields(universe_daily, train_start, test_end)
    logger.info("  %d filter records", len(filter_data))

    # 4. 应用过滤
    logger.info("Step 4/4: applying filters...")
    universe = apply_filters(universe_daily, filter_data)

    # 输出
    out_dir = root / "outputs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_path = out_dir / "universe_csi500.csv"
    write_csv(universe, universe_path)
    logger.info("universe exported: %s (%d rows)", universe_path, len(universe))

    stats = compute_stats(universe)
    stats_path = out_dir / "universe_stats.csv"
    write_csv(stats, stats_path)
    logger.info("stats exported: %s", stats_path)
    logger.info(
        "date range: %s ~ %s, avg daily stocks: %.1f",
        universe["datetime"].min().date(),
        universe["datetime"].max().date(),
        stats["count"].mean(),
    )


if __name__ == "__main__":
    main()
