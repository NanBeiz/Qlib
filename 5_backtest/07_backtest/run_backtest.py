#!/usr/bin/env python3
"""Layer 5a: 回测 —— 基于持仓和成本模型计算 NAV。

使用 Qlib SimulatorExecutor + Exchange 组件，
从日频持仓 CSV 构建回测，输出 NAV 和 Qlib 标准报告。

Usage:
    python 5_backtest/07_backtest/run_backtest.py

输出:
    outputs/backtest/nav_lgbm.csv           (NAV 序列)
    outputs/backtest/report_normal.csv       (Qlib 标准报告)
    outputs/backtest/report_long_short.csv   (多空拆分)
    outputs/backtest/trades_lgbm.csv         (调仓换手明细)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.data import D
from qlib.log import get_module_logger
from qlib.contrib.evaluate import risk_analysis

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_csv, write_csv

logger = get_module_logger(__name__)


def load_forward_returns(
    instruments: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """用 Qlib 拉取前向收益率（与 label 公式一致）。

    Returns
    -------
    pd.DataFrame
        columns: datetime, instrument, forward_return
    """
    expr = "Ref($close, -2) / Ref($close, -1) - 1"
    df = D.features(
        instruments=instruments,
        fields=[expr],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    if df.index.nlevels == 2:
        df = df.reset_index()

    # 重命名最后一列为 forward_return
    col = df.columns[-1]
    df = df.rename(columns={col: "forward_return"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str)
    return df[["datetime", "instrument", "forward_return"]]


def load_benchmark_returns(
    benchmark: str,
    start_time: str,
    end_time: str,
) -> pd.Series:
    """用 Qlib 拉取基准收益率。"""
    try:
        df = D.features(
            instruments=[benchmark],
            fields=["$close"],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )
        if df.index.nlevels == 2:
            df = df.reset_index()
        close_col = [c for c in df.columns if "close" in str(c).lower()][0]
        bench_ret = df.set_index("datetime")[close_col].pct_change().dropna()
        bench_ret.name = "bench"
        return bench_ret
    except Exception as e:
        logger.warning("could not load benchmark %s: %s", benchmark, e)
        return pd.Series(dtype=float)


def get_rebalance_dates_from_positions(positions: pd.DataFrame) -> set[pd.Timestamp]:
    """从持仓数据中识别调仓日（相邻两天持仓变化的日子）。"""
    dates = sorted(positions["datetime"].unique())
    if len(dates) < 2:
        return set(dates)

    rebalance = set()
    for i, d in enumerate(dates):
        if i == 0:
            rebalance.add(pd.Timestamp(d))
            continue
        prev = positions[positions["datetime"] == dates[i - 1]]
        curr = positions[positions["datetime"] == d]
        prev_set = set(zip(prev["instrument"], prev["weight"]))
        curr_set = set(zip(curr["instrument"], curr["weight"]))
        if prev_set != curr_set:
            rebalance.add(pd.Timestamp(d))
    return rebalance


def run_backtest(
    positions: pd.DataFrame,
    forward_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    cost_cfg: dict,
) -> dict:
    """执行日频回测。

    Parameters
    ----------
    positions : pd.DataFrame
        columns: datetime, instrument, weight
    forward_returns : pd.DataFrame
        columns: datetime, instrument, forward_return
    benchmark_returns : pd.Series
        index: datetime, values: daily benchmark return
    cost_cfg : dict
        open_cost, close_cost, min_cost, limit_threshold

    Returns
    -------
    dict
        nav_df, report_normal_df, report_long_short_df, trades_df
    """
    # 构建查找表
    ret_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for dt, grp in forward_returns.groupby("datetime", sort=False):
        d = pd.Timestamp(dt)
        ret_lookup[d] = {}
        for _, row in grp.iterrows():
            if pd.notna(row["forward_return"]):
                ret_lookup[d][str(row["instrument"])] = float(row["forward_return"])

    pos_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for dt, grp in positions.groupby("datetime", sort=False):
        d = pd.Timestamp(dt)
        pos_lookup[d] = {}
        for _, row in grp.iterrows():
            pos_lookup[d][str(row["instrument"])] = float(row["weight"])

    trading_days = pd.DatetimeIndex(sorted(
        set(pd.Timestamp(d) for d in positions["datetime"].unique())
    ))
    rebalance_dates = get_rebalance_dates_from_positions(positions)

    open_cost = cost_cfg.get("open_cost", 0.0005)
    close_cost = cost_cfg.get("close_cost", 0.0015)
    min_cost = cost_cfg.get("min_cost", 5)

    logger.info(
        "backtest: %d trading days, %d rebalance dates, open_cost=%.4f, close_cost=%.4f",
        len(trading_days), len(rebalance_dates), open_cost, close_cost,
    )

    nav = 1.0
    nav_rows: list[dict] = []
    trade_rows: list[dict] = []
    current_weights: dict[str, float] = {}

    for i, day in enumerate(trading_days):
        day_weights = pos_lookup.get(day, {})
        day_ret = ret_lookup.get(day, {})

        # 组合收益（前一日持仓 × 当日收益）
        port_ret = 0.0
        long_ret = 0.0
        short_ret = 0.0
        active_stocks = 0

        if i > 0 and current_weights:
            prev_day = trading_days[i - 1]
            prev_ret = ret_lookup.get(prev_day, {})
            for inst, w in current_weights.items():
                r = prev_ret.get(inst, 0.0)
                if not np.isnan(r):
                    port_ret += w * r
                    if w > 0:
                        long_ret += w * r
                    else:
                        short_ret += w * r
                    active_stocks += 1

        nav *= 1.0 + port_ret

        # 交易成本
        cost = 0.0
        turnover = 0.0
        is_rebalance = day in rebalance_dates

        if is_rebalance and day_weights:
            # 计算换手
            if current_weights and i > 0:
                # 漂移后的权重
                prev_day = trading_days[i - 1]
                prev_ret = ret_lookup.get(prev_day, {})
                drifted = {}
                for inst, w in current_weights.items():
                    r = prev_ret.get(inst, 0.0)
                    drifted[inst] = w * (1.0 + r) if not np.isnan(r) else w
                denom = 1.0 + port_ret
                if abs(denom) > 1e-10:
                    drifted = {k: v / denom for k, v in drifted.items()}

                # 买方换手 = sum(new_weight - old_weight for new > old)
                buy = 0.0
                sell = 0.0
                all_stocks = set(list(drifted.keys()) + list(day_weights.keys()))
                for inst in all_stocks:
                    old_w = drifted.get(inst, 0.0)
                    new_w = day_weights.get(inst, 0.0)
                    diff = new_w - old_w
                    if diff > 0:
                        buy += diff
                    else:
                        sell += abs(diff)
            else:
                buy = sum(w for w in day_weights.values() if w > 0)
                sell = sum(abs(w) for w in day_weights.values() if w < 0)

            turnover = (buy + sell) / 2.0
            # 成本 = 开仓费 × 总交易额 + 平仓费 × 卖出 + 最小费用
            cost = open_cost * (buy + sell) + close_cost * sell
            cost = max(cost, min_cost / 1e8)
            nav *= 1.0 - cost

            # 记录换手明细
            if turnover > 0:
                trade_rows.append({
                    "datetime": day,
                    "turnover": turnover,
                    "cost": cost,
                    "n_long": sum(1 for w in day_weights.values() if w > 0),
                    "n_short": sum(1 for w in day_weights.values() if w < 0),
                })

            current_weights = {k: v for k, v in day_weights.items()}
        elif i > 0 and current_weights:
            # 漂移
            prev_day = trading_days[i - 1]
            prev_ret = ret_lookup.get(prev_day, {})
            new_w = {}
            for inst, w in current_weights.items():
                r = prev_ret.get(inst, 0.0)
                new_w[inst] = w * (1.0 + r) if not np.isnan(r) else w
            denom = 1.0 + port_ret
            if abs(denom) > 1e-10:
                current_weights = {k: v / denom for k, v in new_w.items()}
            else:
                current_weights = new_w

        bench_ret = benchmark_returns.get(day, 0.0) if len(benchmark_returns) > 0 else 0.0
        nav_rows.append({
            "datetime": day,
            "NAV": nav,
            "return": port_ret,
            "bench": bench_ret,
            "long_return": long_ret,
            "short_return": short_ret,
            "cost": cost,
            "turnover": turnover,
            "excess": port_ret - bench_ret,
        })

    nav_df = pd.DataFrame(nav_rows)

    # Qlib report_normal 格式
    report_normal = nav_df[["datetime", "return", "cost", "bench", "turnover"]].copy()
    report_normal = report_normal.set_index("datetime")
    report_normal.index.name = "datetime"

    # 多空拆分
    report_ls = nav_df[["datetime", "long_return", "short_return"]].copy()
    report_ls["long_short"] = report_ls["long_return"] + report_ls["short_return"]
    report_ls = report_ls.set_index("datetime")
    report_ls.index.name = "datetime"

    trades_df = pd.DataFrame(trade_rows)

    return {
        "nav_df": nav_df,
        "report_normal": report_normal,
        "report_long_short": report_ls,
        "trades_df": trades_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 5a: 回测")
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置"
    )
    parser.add_argument(
        "--positions", default=None, help="持仓 CSV 路径"
    )
    parser.add_argument(
        "--output-dir", default=None, help="输出目录"
    )
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    provider_uri = common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data")
    region = common_cfg.get("region", "cn")
    qlib.init(provider_uri=provider_uri, region=region)

    test_start = common_cfg["test_start"]
    test_end = common_cfg["test_end"]
    benchmark = common_cfg.get("benchmark", "SH000300")
    cost_cfg = common_cfg.get("cost", {})

    # 加载持仓
    pos_path = args.positions or str(root / "outputs" / "positions" / "positions_lgbm.csv")
    logger.info("loading positions from %s...", pos_path)
    positions = read_csv(pos_path)
    positions = positions[
        (positions["datetime"] >= pd.Timestamp(test_start)) &
        (positions["datetime"] <= pd.Timestamp(test_end))
    ]
    logger.info("positions: %d rows, %d instruments", len(positions), positions["instrument"].nunique())

    # 拉取前向收益
    instruments = sorted(positions["instrument"].unique())
    logger.info("fetching forward returns for %d instruments...", len(instruments))
    forward_returns = load_forward_returns(instruments, test_start, test_end)
    logger.info("forward returns: %d rows", len(forward_returns))

    # 拉取基准收益
    logger.info("loading benchmark %s returns...", benchmark)
    bench_returns = load_benchmark_returns(benchmark, test_start, test_end)

    # 执行回测
    results = run_backtest(positions, forward_returns, bench_returns, cost_cfg)

    # 输出
    out_dir = Path(args.output_dir or str(root / "outputs" / "backtest"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # NAV CSV
    nav_path = out_dir / "nav_lgbm.csv"
    write_csv(results["nav_df"], nav_path)
    logger.info("NAV exported: %s", nav_path)

    # report_normal
    report_path = out_dir / "report_normal.csv"
    results["report_normal"].to_csv(report_path, float_format="%.8f")
    logger.info("report_normal exported: %s", report_path)

    # report_long_short
    ls_path = out_dir / "report_long_short.csv"
    results["report_long_short"].to_csv(ls_path, float_format="%.8f")
    logger.info("report_long_short exported: %s", ls_path)

    # trades
    if not results["trades_df"].empty:
        trades_path = out_dir / "trades_lgbm.csv"
        write_csv(results["trades_df"], trades_path)
        logger.info("trades exported: %s", trades_path)

    # Qlib risk_analysis
    try:
        report = results["report_normal"]
        analysis = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="day")
        analysis.to_csv(out_dir / "qlib_risk_analysis.csv", float_format="%.8f")
        logger.info("qlib risk_analysis exported: %s", out_dir / "qlib_risk_analysis.csv")
    except Exception as e:
        logger.warning("qlib risk_analysis failed: %s", e)

    # 终端摘要
    nav_df = results["nav_df"]
    total_ret = nav_df["NAV"].iloc[-1] / nav_df["NAV"].iloc[0] - 1
    n_years = (nav_df["datetime"].iloc[-1] - nav_df["datetime"].iloc[0]).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.5)) - 1
    ann_vol = nav_df["return"].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    peak = nav_df["NAV"].expanding().max()
    drawdown = (nav_df["NAV"] - peak) / peak
    max_dd = drawdown.min()

    print("\n" + "=" * 56)
    print("  Backtest Summary")
    print("=" * 56)
    print(f"  Total Return:      {total_ret*100:+.2f}%")
    print(f"  Annual Return:     {ann_ret*100:+.2f}%")
    print(f"  Annual Volatility: {ann_vol*100:+.2f}%")
    print(f"  Sharpe Ratio:      {sharpe:.4f}")
    print(f"  Max Drawdown:      {max_dd*100:+.2f}%")
    print(f"  Calmar Ratio:      {ann_ret/abs(max_dd) if max_dd != 0 else 0:.4f}")
    print("=" * 56)

    logger.info("Layer 5a complete")


if __name__ == "__main__":
    main()
