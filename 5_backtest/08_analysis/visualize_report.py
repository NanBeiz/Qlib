#!/usr/bin/env python3
"""Layer 5b 扩展: 使用 Qlib 原生报告函数生成可视化。

基于 report_normal.csv 生成 Qlib 标准交互式 Plotly 报告。

Usage:
    python 5_backtest/08_analysis/visualize_report.py

输出:
    outputs/analysis/report.html            (Qlib 综合报告)
    outputs/analysis/risk_analysis.html     (风险分析图)
    outputs/analysis/nav_curve.png          (NAV 曲线 + 回撤)
    outputs/analysis/monthly_heatmap.png    (月度收益热力图)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from qlib.contrib.report.analysis_position import report_graph, risk_analysis_graph
from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_csv

logger = get_module_logger(__name__)


def load_report(report_path: str) -> pd.DataFrame:
    """加载并标准化 Qlib report_normal 格式。

    Returns
    -------
    pd.DataFrame
        index=datetime, columns=[return, cost, bench, turnover]
    """
    df = pd.read_csv(report_path, parse_dates=True, index_col=0)
    df.index.name = "date"
    # 确保必要的列: return, cost, bench, turnover
    for col in ["return", "cost", "bench", "turnover"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def plot_nav_curve(report: pd.DataFrame, output_path: str) -> None:
    """使用 matplotlib 生成 NAV 曲线 + 回撤双面板图。"""
    ret = report["return"].values
    nav = (1 + ret).cumprod()

    # 累计收益
    cum_ret = nav - 1

    # 回撤
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    dates = report.index
    ax1.plot(dates, cum_ret * 100, color="#1f77b4", linewidth=1.0)
    ax1.fill_between(dates, 0, cum_ret * 100, where=cum_ret >= 0,
                      color="#1f77b4", alpha=0.15)
    ax1.fill_between(dates, 0, cum_ret * 100, where=cum_ret < 0,
                      color="#d62728", alpha=0.15)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.set_title("CSI500 Market-Neutral Strategy — NAV Curve", fontsize=14)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dates, 0, drawdown * 100, color="#d62728", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("NAV curve saved: %s", output_path)


def plot_monthly_heatmap(report: pd.DataFrame, output_path: str) -> None:
    """月度收益热力图。"""
    ret = report["return"].copy()
    ret.index = pd.to_datetime(ret.index)

    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = pd.MultiIndex.from_arrays(
        [monthly.index.year, monthly.index.month],
        names=["Year", "Month"],
    )
    monthly = monthly.unstack()
    # 去掉缺失的年份（不完整的首尾年）
    monthly = monthly.dropna(how="all", axis=0)

    if monthly.empty:
        logger.warning("not enough data for monthly heatmap")
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(monthly) * 0.6)))
    im = ax.imshow(monthly.values * 100, aspect="auto", cmap="RdYlGn",
                   vmin=-15, vmax=15)

    # 标注每个格子的值
    for i in range(len(monthly)):
        for j in range(len(monthly.columns)):
            v = monthly.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:+.1f}%", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < 0.1 else "white",
                        fontweight="bold")

    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(monthly.columns)])
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index)
    ax.set_title("Monthly Returns Heatmap (%)", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Monthly heatmap saved: %s", output_path)


def generate_qlib_html_reports(report: pd.DataFrame, out_dir: Path) -> None:
    """使用 Qlib report_graph 和 risk_analysis_graph 生成 HTML 报告。"""
    try:
        graphs = report_graph(report, show_notebook=False)
        if graphs:
            for i, fig in enumerate(graphs):
                fig.write_html(str(out_dir / f"report_{i}.html"))
            logger.info("Qlib report_graph saved: %s", out_dir / "report_*.html")
    except Exception as e:
        logger.warning("report_graph failed: %s", e)

    try:
        # 加载多空拆分
        ls_path = out_dir.parent / "backtest" / "report_long_short.csv"
        if ls_path.exists():
            report_ls = pd.read_csv(ls_path, parse_dates=True, index_col=0)
            report_ls.index.name = "date"
            if "long_return" in report_ls.columns:
                report_ls = report_ls.rename(columns={
                    "long_return": "long", "short_return": "short",
                })
            if "long_short" not in report_ls.columns:
                report_ls["long_short"] = report_ls["long"] + report_ls["short"]
        else:
            report_ls = None

        # 重新用 Qlib risk_analysis 生成标准 analysis_df
        from qlib.contrib.evaluate import risk_analysis
        excess_wo = risk_analysis(report["return"] - report["bench"], freq="day")
        excess_w = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="day")
        analysis_df = pd.concat({
            "excess_return_without_cost": excess_wo,
            "excess_return_with_cost": excess_w,
        })

        graphs = risk_analysis_graph(
            analysis_df=analysis_df,
            report_normal_df=report,
            report_long_short_df=report_ls,
            show_notebook=False,
        )
        if graphs:
            for i, fig in enumerate(graphs):
                fig.write_html(str(out_dir / f"risk_analysis_{i}.html"))
            logger.info("risk_analysis_graph saved: %s", out_dir / "risk_analysis_*.html")
    except Exception as e:
        logger.warning("risk_analysis_graph failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Qlib 回测可视化")
    parser.add_argument(
        "--report-path", default=None, help="report_normal.csv 路径"
    )
    parser.add_argument(
        "--output-dir", default="outputs/analysis", help="输出目录"
    )
    args = parser.parse_args()

    root = get_project_root()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.report_path or str(root / "outputs" / "backtest" / "report_normal.csv")
    logger.info("loading report from %s...", report_path)
    report = load_report(report_path)
    logger.info("report: %d rows, date range %s ~ %s",
                 len(report), report.index[0], report.index[-1])

    # 1. NAV 曲线 + 回撤
    plot_nav_curve(report, str(out_dir / "nav_curve.png"))

    # 2. 月度热力图
    if len(report) > 60:
        plot_monthly_heatmap(report, str(out_dir / "monthly_heatmap.png"))

    # 3. Qlib 原生 HTML 报告
    generate_qlib_html_reports(report, out_dir)

    logger.info("visualization complete")


if __name__ == "__main__":
    main()
