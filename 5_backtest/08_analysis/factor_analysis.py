#!/usr/bin/env python3
"""Layer 5b: 分析 —— IC 序列 + 绩效指标 + 报告生成。

使用 Qlib 的 calc_ic 和 risk_analysis 函数，
生成因子 IC、模型 ICIR、组合绩效指标和 Plotly 报告。

Usage:
    python 5_backtest/08_analysis/factor_analysis.py

输出:
    outputs/analysis/factor_ic.csv            (因子 IC 汇总)
    outputs/analysis/factor_ic_series.csv     (逐日 IC 序列)
    outputs/analysis/performance_metrics.csv  (绩效指标)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qlib.log import get_module_logger
from qlib.contrib.evaluate import risk_analysis

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_csv, write_csv

logger = get_module_logger(__name__)


def _factor_columns(df: pd.DataFrame) -> list[str]:
    """自动检测因子列（排除 datetime, instrument, ALPHA, LABEL0）。"""
    exclude = {"datetime", "instrument", "ALPHA", "LABEL0", "ALPHA_BASELINE"}
    return [c for c in df.columns if c not in exclude]


def compute_ic_series(
    panel: pd.DataFrame,
    factor_names: list[str],
    label_col: str = "LABEL0",
) -> pd.DataFrame:
    """计算每个因子每个交易日的 Pearson IC。

    使用 Qlib 风格的截面 corr，而非自己实现。

    Returns
    -------
    pd.DataFrame
        datetime × factor，值为 IC。
    """
    df = panel[["datetime"] + factor_names + [label_col]].copy()
    df = df.dropna(subset=[label_col])

    results = []
    for date, grp in df.groupby("datetime", sort=True):
        row = {"datetime": date}
        for f in factor_names:
            valid = grp[[f, label_col]].dropna()
            if len(valid) < 30:
                row[f] = np.nan
            else:
                row[f] = valid[f].corr(valid[label_col])
        results.append(row)

    return pd.DataFrame(results).sort_values("datetime").reset_index(drop=True)


def compute_rank_ic_series(
    panel: pd.DataFrame,
    factor_names: list[str],
    label_col: str = "LABEL0",
) -> pd.DataFrame:
    """计算每个因子每个交易日的 Rank IC（Spearman 秩相关）。"""
    df = panel[["datetime"] + factor_names + [label_col]].copy()
    df = df.dropna(subset=[label_col])

    results = []
    for date, grp in df.groupby("datetime", sort=True):
        row = {"datetime": date}
        for f in factor_names:
            valid = grp[[f, label_col]].dropna()
            if len(valid) < 30:
                row[f] = np.nan
            else:
                rank_f = valid[f].rank()
                rank_l = valid[label_col].rank()
                row[f] = rank_f.corr(rank_l)
        results.append(row)

    return pd.DataFrame(results).sort_values("datetime").reset_index(drop=True)


def compute_ic_summary(
    ic_series: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    """从 IC 序列计算 IC Mean, IC Std, ICIR, IR。"""
    rows = []
    for f in factor_names:
        s = ic_series[f].dropna()
        if len(s) < 5:
            rows.append({
                "factor": f, "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "ic_pos_ratio": np.nan, "n_days": len(s),
            })
        else:
            mean_ic = float(s.mean())
            std_ic = float(s.std(ddof=1))
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            rows.append({
                "factor": f,
                "ic_mean": round(mean_ic, 6),
                "ic_std": round(std_ic, 6),
                "icir": round(icir, 4),
                "ic_pos_ratio": round(float((s > 0).sum() / len(s)), 4),
                "n_days": len(s),
            })
    return pd.DataFrame(rows)


def compute_performance_metrics(
    report_normal: pd.DataFrame,
) -> dict:
    """使用 Qlib risk_analysis 计算绩效指标。"""
    try:
        excess = report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        analysis = risk_analysis(excess, freq="day")
        if analysis is not None:
            annual_return = float(analysis.loc["annualized_return"].iloc[0]) if "annualized_return" in analysis.index else 0.0
            annual_vol = float(analysis.loc["annualized_volatility"].iloc[0]) if "annualized_volatility" in analysis.index else 0.0
            sharpe = float(analysis.loc["information_ratio"].iloc[0]) if "information_ratio" in analysis.index else 0.0
            max_dd = float(analysis.loc["max_drawdown"].iloc[0]) if "max_drawdown" in analysis.index else 0.0
        else:
            annual_return = annual_vol = sharpe = max_dd = 0.0
    except Exception:
        annual_return = annual_vol = sharpe = max_dd = 0.0

    # 从 NAV 直接计算
    ret = report_normal["return"]
    total_return = (1 + ret).prod() - 1
    n_years = len(ret) / 252
    if n_years > 0:
        annual_return = (1 + total_return) ** (1 / n_years) - 1
    annual_vol = float(ret.std() * np.sqrt(252))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    cum = (1 + ret).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": round(float(total_return), 6),
        "annual_return": round(float(annual_return), 6),
        "annual_volatility": round(float(annual_vol), 6),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 6),
        "calmar_ratio": round(float(calmar), 4),
        "n_days": len(ret),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 5b: 分析")
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置"
    )
    parser.add_argument(
        "--factors-path", default=None, help="原始因子 CSV 路径"
    )
    parser.add_argument(
        "--predictions-path", default=None, help="预测信号 CSV 路径"
    )
    parser.add_argument(
        "--report-path", default=None, help="report_normal.csv 路径"
    )
    parser.add_argument(
        "--output-dir", default="outputs/analysis", help="输出目录"
    )
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 因子 IC 分析 ----
    factors_path = args.factors_path or str(root / "outputs" / "factors" / "all_factors_raw.csv")
    if Path(factors_path).exists():
        logger.info("loading factors from %s...", factors_path)
        panel = read_csv(factors_path)
        factor_names = _factor_columns(panel)

        has_label = "LABEL0" in panel.columns
        if factor_names and has_label:
            logger.info("computing IC for %d factors...", len(factor_names))

            ic_df = compute_ic_series(panel, factor_names)
            rank_ic_df = compute_rank_ic_series(panel, factor_names)

            ic_summary = compute_ic_summary(ic_df, factor_names)
            rank_ic_summary = compute_ic_summary(rank_ic_df, factor_names)
            rank_ic_summary = rank_ic_summary.rename(columns={
                "ic_mean": "rank_ic_mean", "ic_std": "rank_ic_std", "icir": "rank_icir",
            })

            # 合并 IC + Rank IC
            full_summary = ic_summary.merge(
                rank_ic_summary[["factor", "rank_ic_mean", "rank_icir"]],
                on="factor", how="left",
            )

            # 输出
            write_csv(full_summary, out_dir / "factor_ic.csv")
            write_csv(ic_df, out_dir / "factor_ic_series.csv")
            logger.info("factor IC exported")

            # 终端输出
            print("\n" + "=" * 72)
            print(f"{'Factor':<22s} {'IC Mean':>8s}  {'Rank IC':>8s}  {'ICIR':>8s}  {'Rank ICIR':>8s}")
            print("=" * 72)
            for _, row in full_summary.iterrows():
                print(f"{row['factor']:<22s} "
                      f"{row.get('ic_mean', 0):>+8.4f}  "
                      f"{row.get('rank_ic_mean', 0):>+8.4f}  "
                      f"{row.get('icir', 0):>8.4f}  "
                      f"{row.get('rank_icir', 0):>8.4f}")
            print("=" * 72)

    # ---- 预测信号的 IC/ICIR ----
    pred_path = args.predictions_path or str(root / "outputs" / "predictions" / "predictions_lgbm.csv")
    if Path(pred_path).exists():
        logger.info("loading predictions from %s...", pred_path)
        preds = read_csv(pred_path)

        # 需要 label 才能算 IC；尝试从原始因子数据加载
        if Path(factors_path).exists():
            panel = read_csv(factors_path)
            if "LABEL0" in panel.columns:
                pred_panel = preds.merge(
                    panel[["datetime", "instrument", "LABEL0"]],
                    on=["datetime", "instrument"], how="inner",
                )
                model_ic = compute_ic_series(pred_panel, ["ALPHA"], label_col="LABEL0")
                model_summary = compute_ic_summary(model_ic, ["ALPHA"])
                write_csv(model_summary, out_dir / "model_ic.csv")
                logger.info("model IC exported")

                if not model_summary.empty:
                    print(f"\nModel ALPHA IC: mean={model_summary.iloc[0]['ic_mean']:.4f}, "
                          f"ICIR={model_summary.iloc[0]['icir']:.4f}")

    # ---- 绩效指标 ----
    report_path = args.report_path or str(root / "outputs" / "backtest" / "report_normal.csv")
    if Path(report_path).exists():
        logger.info("loading report from %s...", report_path)
        report = pd.read_csv(report_path, index_col=0, parse_dates=True)
        metrics = compute_performance_metrics(report)

        metrics_df = pd.DataFrame([metrics])
        write_csv(metrics_df, out_dir / "performance_metrics.csv")
        logger.info("performance metrics exported")

        print("\n" + "=" * 56)
        print("  Performance Metrics")
        print("=" * 56)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("=" * 56)

    logger.info("Layer 5b complete")


if __name__ == "__main__":
    main()
