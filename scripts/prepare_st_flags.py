#!/usr/bin/env python3
"""准备 ST 标记数据。

从本地 CSV 读取 ST 历史记录，生成 PIT 正确的 $is_st 字段 CSV，
供 Layer 1 数据准备层使用。

Usage:
    python scripts/prepare_st_flags.py --source data/st_history.csv

输出格式: outputs/data/st_flags.csv
列: datetime, instrument, is_st (0=正常, 1=ST)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_csi500_mn.utils.io import write_csv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_st_history(source_path: str) -> pd.DataFrame:
    """加载 ST 历史数据。

    期望 CSV 列: instrument, start_date, end_date
    start_date: ST 标记开始日期（含）
    end_date: ST 标记结束日期（含），NaT 表示至今仍为 ST
    """
    df = pd.read_csv(source_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def expand_st_daily(
    st_history: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """将 ST 期间展开为日频数据。

    对每只股票的每个 ST 期间，生成期间内所有交易日的 is_st=1 记录。
    """
    rows = []
    for _, rec in st_history.iterrows():
        instrument = str(rec["instrument"])
        start = rec["start_date"]
        end = rec["end_date"] if pd.notna(rec["end_date"]) else trading_calendar[-1]

        mask = (trading_calendar >= start) & (trading_calendar <= end)
        for d in trading_calendar[mask]:
            rows.append({"datetime": d, "instrument": instrument, "is_st": 1})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 ST 标记数据")
    parser.add_argument(
        "--source", default="data/st_history.csv", help="ST 历史 CSV 路径"
    )
    parser.add_argument(
        "--trading-calendar",
        default=None,
        help="交易日历 CSV（datetime 列），不提供则用默认 2010-2026",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径，默认 outputs/data/st_flags.csv",
    )
    args = parser.parse_args()

    root = _project_root()
    source_path = root / args.source

    if not source_path.exists():
        print(f"[SKIP] ST 历史文件不存在: {source_path}")
        print("  请先从 Tushare / akshare 下载 ST 历史数据后重试。")
        print("  在此之前，Layer 1 的 is_st 过滤将使用 Qlib 内置字段（如果可用）。")
        return

    st_history = load_st_history(str(source_path))

    if args.trading_calendar:
        cal_df = pd.read_csv(args.trading_calendar)
        trading_calendar = pd.DatetimeIndex(sorted(pd.to_datetime(cal_df["datetime"].unique())))
    else:
        trading_calendar = pd.bdate_range("2010-01-01", "2026-12-31")

    st_daily = expand_st_daily(st_history, trading_calendar)
    st_daily = st_daily.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    output_path = args.output or str(root / "outputs" / "data" / "st_flags.csv")
    write_csv(st_daily, output_path)
    print(f"ST 标记已导出: {output_path}")
    print(f"  记录数: {len(st_daily)}")
    print(f"  涉及股票数: {st_daily['instrument'].nunique()}")
    print(f"  日期范围: {st_daily['datetime'].min().date()} ~ {st_daily['datetime'].max().date()}")


if __name__ == "__main__":
    main()
