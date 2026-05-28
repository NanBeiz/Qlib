"""风险模型模块 — 协方差估计 + 组合风险分解。

提供：
- PortfolioCovEstimator: 封装 Qlib ShrinkCov / StructuredCov / POET
- 组合风险指标：波动率、VaR、CVaR、风险贡献
- CLI 入口：build_riskmodel.py
"""

from .covariance import PortfolioCovEstimator
from .risk_metrics import (
    compute_cvar,
    compute_portfolio_volatility,
    compute_risk_decomposition,
    compute_var,
    compute_variance_decomposition,
)

__all__ = [
    "PortfolioCovEstimator",
    "compute_portfolio_volatility",
    "compute_risk_decomposition",
    "compute_var",
    "compute_cvar",
    "compute_variance_decomposition",
]
