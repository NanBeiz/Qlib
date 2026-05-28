#!/usr/bin/env python3
"""准备行业分类 + 市值数据。

从本地 CSV 读取申万一级行业分类（28 类）和总市值数据，
生成 one-hot 虚拟变量 + log 市值 CSV，供 IndustryMcapNeutralize processor 使用。

Usage:
    python scripts/prepare_industry_data.py \\
        --industry-csv data/shenwan_industry.csv \\
        --mcap-csv data/market_cap.csv

输出格式: outputs/data/industry_dummies.csv
列: datetime, instrument, IND_<code> (28 列), LOG_MCAP
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_csi500_mn.utils.io import write_csv


# 申万一级行业编码（28 类）
SW_INDUSTRY_CODES = [
    "801010", "801020", "801030", "801040", "801050",
    "801080", "801110", "801120", "801130", "801140",
    "801150", "801160", "801170", "801180", "801200",
    "801210", "801230", "801710", "801720", "801730",
    "801740", "801750", "801760", "801770", "801780",
    "801790", "801880", "801890",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_industry_classification(path: str) -> pd.DataFrame:
    """加载行业分类数据。

    期望 CSV 列: instrument, industry_code
    industry_code 为申万一级行业代码（如 801010）。
    """
    df = pd.read_csv(path)
    df["instrument"] = df["instrument"].astype(str)
    df["industry_code"] = df["industry_code"].astype(str)
    return df


def load_market_cap(path: str) -> pd.DataFrame:
    """加载市值数据。

    期望 CSV 列: datetime, instrument, mcap（总市值，单位：元）
    """
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str)
    df["mcap"] = pd.to_numeric(df["mcap"], errors="coerce")
    return df


def build_industry_dummies(
    industry_df: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """为每个交易日生成行业 one-hot 虚拟变量。

    假设行业分类在时间上不变（或使用最新的分类作为快照）。
    """
    rows = []
    for d in trading_calendar:
        day_df = industry_df.copy()
        day_df["datetime"] = d
        rows.append(day_df)
    result = pd.concat(rows, ignore_index=True)

    # One-hot 编码
    dummies = pd.get_dummies(result["industry_code"], prefix="IND").astype(float)
    result = pd.concat(
        [result[["datetime", "instrument"]], dummies],
        axis=1,
    )

    # 确保所有 28 个行业列都存在
    expected_cols = [f"IND_{code}" for code in SW_INDUSTRY_CODES]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0.0

    return result[["datetime", "instrument"] + expected_cols]


def merge_mcap(
    industry_dummies: pd.DataFrame,
    mcap_df: pd.DataFrame,
) -> pd.DataFrame:
    """合并市值数据，计算 LOG_MCAP。"""
    merged = industry_dummies.merge(
        mcap_df[["datetime", "instrument", "mcap"]],
        on=["datetime", "instrument"],
        how="left",
    )
    merged["LOG_MCAP"] = np.log(merged["mcap"].replace(0, np.nan))
    merged.loc[merged["LOG_MCAP"].isna(), "LOG_MCAP"] = merged["LOG_MCAP"].median()
    return merged.drop(columns=["mcap"])


def main() -> None:
    parser = argparse.ArgumentParser(description="准备行业分类和市值数据")
    parser.add_argument(
        "--industry-csv", default="data/shenwan_industry.csv", help="行业分类 CSV 路径"
    )
    parser.add_argument(
        "--mcap-csv", default="data/market_cap.csv", help="市值数据 CSV 路径"
    )
    parser.add_argument(
        "--trading-calendar",
        default=None,
        help="交易日历 CSV（datetime 列），不提供则用默认 2010-2026",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径，默认 outputs/data/industry_dummies.csv",
    )
    args = parser.parse_args()

    root = _project_root()
    industry_path = root / args.industry_csv

    if not industry_path.exists():
        print(f"[SKIP] 行业分类文件不存在: {industry_path}")
        print("  请先从数据源下载申万行业分类后重试。")
        print("  在此之前，IndustryMcapNeutralize 将降级为仅截面去均值（X 只含截距列）。")
        return

    industry_df = load_industry_classification(str(industry_path))

    if args.trading_calendar:
        cal_df = pd.read_csv(args.trading_calendar)
        trading_calendar = pd.DatetimeIndex(sorted(pd.to_datetime(cal_df["datetime"].unique())))
    else:
        trading_calendar = pd.bdate_range("2010-01-01", "2026-12-31")

    # 构建行业虚拟变量
    industry_dummies = build_industry_dummies(industry_df, trading_calendar)

    # 合并市值
    mcap_path = root / args.mcap_csv
    if mcap_path.exists():
        mcap_df = load_market_cap(str(mcap_path))
        result = merge_mcap(industry_dummies, mcap_df)
    else:
        print(f"[WARN] 市值文件不存在: {mcap_path}，LOG_MCAP 将使用默认值 0")
        result = industry_dummies.copy()
        result["LOG_MCAP"] = 0.0

    output_path = args.output or str(root / "outputs" / "data" / "industry_dummies.csv")
    write_csv(result, output_path)
    print(f"行业虚拟变量已导出: {output_path}")
    print(f"  记录数: {len(result)}")
    print(f"  行业列数: {len([c for c in result.columns if c.startswith('IND_')])}")
    print(f"  日期范围: {result['datetime'].min().date()} ~ {result['datetime'].max().date()}")


if __name__ == "__main__":
    main()
