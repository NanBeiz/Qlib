"""组合 Beta 估计模块。

流程：
1. estimate_stock_betas()  — 用个股日收益对基准日收益做滚动 OLS，估计每只股票的 Beta
2. compute_portfolio_beta() — 将持仓权重与个股 Beta 加权求和，得到组合 Beta

基准默认使用 CSI500 指数收益率。若无期货数据，可用现货指数替代。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 个股 Beta 估计 ──────────────────────────────────────────────

def estimate_stock_betas(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """用滚动 OLS 估计每只股票对基准的 Beta。

    Beta = Cov(r_i, r_bench) / Var(r_bench)，用滚动窗口计算。

    Parameters
    ----------
    stock_returns : pd.DataFrame
        个股日收益率，index=date, columns=instrument。
    benchmark_returns : pd.Series
        基准日收益率，index=date。
    window : int
        滚动窗口交易日数，默认 252（约 1 年）。
    min_periods : int
        最少有效数据点，默认 60（约 1 季度）。

    Returns
    -------
    pd.DataFrame
        个股 Beta，index=date, columns=instrument。
        不足 min_periods 的日期为 NaN。
    """
    if stock_returns.empty or benchmark_returns.empty:
        logger.warning("empty input for beta estimation")
        return pd.DataFrame()

    # 对齐日期
    common_dates = stock_returns.index.intersection(benchmark_returns.index)
    if len(common_dates) < min_periods:
        logger.warning(
            "insufficient common dates for beta estimation: %d < %d",
            len(common_dates),
            min_periods,
        )
        return pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)

    stock_aligned = stock_returns.loc[common_dates]
    bench_aligned = benchmark_returns.loc[common_dates]

    betas = pd.DataFrame(
        np.nan,
        index=stock_aligned.index,
        columns=stock_aligned.columns,
    )

    bench_vals = bench_aligned.values

    for i in range(min_periods - 1, len(common_dates)):
        start = i - window + 1 if i >= window else 0
        w_bench = bench_vals[start : i + 1]
        bench_var = np.var(w_bench)
        if bench_var == 0:
            continue

        w_stocks = stock_aligned.iloc[start : i + 1].values
        # 只使用有效值
        for j, col in enumerate(stock_aligned.columns):
            col_vals = w_stocks[:, j]
            mask = ~np.isnan(col_vals)
            if mask.sum() < min_periods:
                continue
            cov = np.cov(col_vals[mask], w_bench[mask], ddof=1)[0, 1]
            betas.iloc[i, j] = cov / bench_var

    logger.info(
        "estimated stock betas: window=%d, stocks=%d, dates=%d",
        window,
        len(stock_aligned.columns),
        len(common_dates),
    )

    return betas


# ── 组合 Beta 计算 ───────────────────────────────────────────────

def compute_portfolio_beta(
    positions: dict[str, float],
    stock_betas: pd.Series,
) -> float:
    """计算组合加权 Beta。

    Parameters
    ----------
    positions : dict[str, float]
        {instrument: weight}，多空持仓权重（long 为正，short 为负）。
    stock_betas : pd.Series
        index=instrument, values=beta，某截面日的个股 Beta。

    Returns
    -------
    float
        组合净 Beta。正数表示组合整体与市场同向。
    """
    if not positions or stock_betas.empty:
        return 0.0

    portfolio_beta = 0.0
    matched = 0
    for instrument, weight in positions.items():
        if instrument not in stock_betas.index:
            continue
        b = stock_betas.at[instrument]
        if pd.isna(b):
            continue
        portfolio_beta += weight * float(b)
        matched += 1

    total_stocks = len(positions)
    if matched < total_stocks:
        logger.debug(
            "beta matched %d/%d stocks for portfolio beta calculation",
            matched,
            total_stocks,
        )

    return portfolio_beta


# ── 多空分解 Beta ────────────────────────────────────────────────

def compute_long_short_betas(
    positions: dict[str, float],
    stock_betas: pd.Series,
) -> dict[str, float]:
    """分别计算多头腿和空头腿的 Beta。

    Returns
    -------
    dict with keys: 'long_beta', 'short_beta', 'net_beta', 'long_exposure', 'short_exposure'
    """
    long_beta = 0.0
    short_beta = 0.0
    long_exposure = 0.0
    short_exposure = 0.0

    for instrument, weight in positions.items():
        if instrument not in stock_betas.index:
            continue
        b = stock_betas.at[instrument]
        if pd.isna(b):
            continue

        if weight > 0:
            long_beta += weight * float(b)
            long_exposure += weight
        else:
            short_beta += abs(weight) * float(b)
            short_exposure += abs(weight)

    return {
        "long_beta": long_beta,
        "short_beta": short_beta,
        "net_beta": long_beta - short_beta,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
    }
