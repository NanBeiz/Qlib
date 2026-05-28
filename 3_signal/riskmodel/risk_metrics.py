"""组合风险指标计算。

所有函数为纯函数，不依赖类状态，输入 positions + cov_matrix → 输出风险指标。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_portfolio_volatility(
    positions: dict[str, float],
    cov_matrix: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """组合年化波动率。

    参数
    ----
    positions : dict
        {instrument: weight}，多空权重。
    cov_matrix : pd.DataFrame
        instrument × instrument 日频协方差矩阵。
    annualize : bool
        是否年化。
    trading_days : int
        年化交易日数。

    返回
    ----
    float
    """
    w = _weights_array(positions, cov_matrix)
    if w is None:
        return 0.0

    S = cov_matrix.loc[w.index, w.index].values
    daily_var = float(w.values @ S @ w.values)
    daily_vol = np.sqrt(max(daily_var, 0.0))
    return daily_vol * np.sqrt(trading_days) if annualize else daily_vol


def compute_risk_decomposition(
    positions: dict[str, float],
    cov_matrix: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """每只股票的边际风险贡献与成分风险贡献。

    成分风险贡献 = w_i × (Σ w)_i / σ_p
    边际风险贡献 = (Σ w)_i / σ_p

    返回
    ----
    pd.DataFrame
        columns: instrument, weight, marginal_risk, risk_contribution, pct_contribution
    """
    w = _weights_array(positions, cov_matrix)
    if w is None or len(w) == 0:
        return pd.DataFrame()

    w_vals = w.values
    S = cov_matrix.loc[w.index, w.index].values
    portfolio_var = float(w_vals @ S @ w_vals)
    portfolio_vol = np.sqrt(max(portfolio_var, 1e-16))

    scale = np.sqrt(trading_days) if annualize else 1.0

    # 边际贡献: ∂σ/∂w_i = (Σ w)_i / σ
    marginal = (S @ w_vals) / portfolio_vol * scale

    # 成分贡献: w_i × ∂σ/∂w_i
    component = w_vals * marginal

    # 百分比
    total_risk = np.sum(component)
    pct = component / total_risk if abs(total_risk) > 1e-16 else np.zeros_like(component)

    return pd.DataFrame({
        "instrument": w.index,
        "weight": w_vals,
        "marginal_risk": marginal,
        "risk_contribution": component,
        "pct_contribution": pct,
    }).sort_values("pct_contribution", ascending=False).reset_index(drop=True)


def compute_var(
    positions: dict[str, float],
    cov_matrix: pd.DataFrame,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """参数法 VaR（假设正态分布）。

    参数
    ----
    confidence : float
        置信水平，默认 0.95。
    horizon : int
        持有期（交易日），默认 1。

    返回
    ----
    float
        VaR（正数，表示损失）。
    """
    daily_vol = compute_portfolio_volatility(
        positions, cov_matrix, annualize=False,
    )
    z_score = abs(norm.ppf(1 - confidence))  # VaR 为正数表示损失
    return float(daily_vol * z_score * np.sqrt(horizon))


def compute_cvar(
    positions: dict[str, float],
    cov_matrix: pd.DataFrame,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """参数法 CVaR / Expected Shortfall（假设正态分布）。

    返回
    ----
    float
        CVaR（正数，表示平均超额损失）。
    """
    daily_vol = compute_portfolio_volatility(
        positions, cov_matrix, annualize=False,
    )
    z_score = norm.ppf(1 - confidence)
    # CVaR = σ × φ(z) / (1 - α)  (for zero mean)
    cvar_daily = daily_vol * norm.pdf(z_score) / (1 - confidence)
    return float(cvar_daily * np.sqrt(horizon))


def compute_variance_decomposition(
    positions: dict[str, float],
    factors: np.ndarray,
    factor_cov: np.ndarray,
    specific_vars: np.ndarray,
    instruments: Optional[List[str]] = None,
) -> dict:
    """因子风险 vs 特质风险分解。

    参数
    ----
    positions : dict
        {instrument: weight}。
    factors : np.ndarray
        instrument × k 因子载荷矩阵。
    factor_cov : np.ndarray
        k × k 因子协方差矩阵。
    specific_vars : np.ndarray
        每只股票的特质方差，长度 = 股票数。
    instruments : list[str], optional
        与 factors/specific_vars 顺序对应的 instrument 列表。
        若为 None，假设 factors 行与 sorted(positions.keys()) 对齐。

    返回
    ----
    dict
    """
    if instruments is None:
        instruments = sorted(positions.keys())

    # 对齐：只取 positions 和 factors/specific_vars 中共有的股票
    aligned_inst = [s for s in instruments if s in positions]
    if not aligned_inst:
        return {
            "total_var": 0.0, "factor_var": 0.0, "specific_var": 0.0,
            "factor_pct": 0.0, "specific_pct": 0.0,
        }

    idx_map = {inst: i for i, inst in enumerate(instruments)}
    w_vals = np.array([float(positions[inst]) for inst in aligned_inst])
    f_idx = np.array([idx_map[inst] for inst in aligned_inst])

    F_sub = factors[f_idx, :]   # m × k
    sv_sub = specific_vars[f_idx]  # m

    factor_var = float(w_vals @ F_sub @ factor_cov @ F_sub.T @ w_vals)
    specific_var = float(np.sum(w_vals**2 * sv_sub))
    total_var = factor_var + specific_var

    total_var_safe = max(total_var, 1e-16)
    return {
        "total_var": total_var,
        "factor_var": factor_var,
        "specific_var": specific_var,
        "factor_pct": factor_var / total_var_safe,
        "specific_pct": specific_var / total_var_safe,
    }


def _weights_array(
    positions: dict[str, float],
    cov_matrix: pd.DataFrame,
) -> pd.Series | None:
    """将持仓字典对齐到协方差矩阵的 instrument 顺序。"""
    inst_list = [k for k in cov_matrix.columns if k in positions]
    if not inst_list:
        return None
    return pd.Series(
        {inst: float(positions[inst]) for inst in inst_list},
        name="weight",
    )
