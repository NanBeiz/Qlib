"""测试风险模型模块。

覆盖：
- PortfolioCovEstimator 协方差估计（shrink / structured / poet）
- 组合风险指标（波动率、VaR、CVaR、风险贡献）
- 因子-特质风险分解
- 边界情况（空输入、NaN、无效方法）
- import 路径验证
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 将 3_signal/riskmodel/ 加入 sys.path
_IMPL_DIR = str(Path(__file__).resolve().parents[1] / "3_signal" / "riskmodel")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from covariance import PortfolioCovEstimator  # noqa: E402
from risk_metrics import (  # noqa: E402
    compute_cvar,
    compute_portfolio_volatility,
    compute_risk_decomposition,
    compute_var,
    compute_variance_decomposition,
)


# ── 测试夹具 ────────────────────────────────────────────────────

def _make_returns(n_dates=300, n_stocks=50, seed=42):
    """生成合成日收益矩阵。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    data = rng.standard_normal((n_dates, n_stocks)) * 0.02
    return pd.DataFrame(data, index=dates, columns=[f"S{i:04d}" for i in range(n_stocks)])


def _make_positions(n_stocks=50, n_long=25, n_short=25):
    """生成等权多空持仓。"""
    weights = {}
    for i in range(n_long):
        weights[f"S{i:04d}"] = 1.0 / n_long
    for i in range(n_long, n_long + n_short):
        weights[f"S{i:04d}"] = -1.0 / n_short
    return weights


def _make_cov_matrix(n_stocks=50, seed=42):
    """生成正定协方差矩阵。"""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_stocks, n_stocks)) * 0.02
    cov = A @ A.T + np.eye(n_stocks) * 0.001
    stocks = [f"S{i:04d}" for i in range(n_stocks)]
    return pd.DataFrame(cov, index=stocks, columns=stocks)


# ── PortfolioCovEstimator 测试 ───────────────────────────────────

class TestPortfolioCovEstimator:
    """协方差估计器测试。"""

    def test_shrink_estimator_shape(self):
        """收缩估计器返回正确的协方差矩阵形状。"""
        returns = _make_returns(n_dates=200, n_stocks=20)
        est = PortfolioCovEstimator(method="shrink")
        cov = est.estimate(returns)
        assert cov.shape == (20, 20)
        assert list(cov.index) == list(returns.columns)
        # 对角线应为正
        assert (np.diag(cov.values) > 0).all()

    def test_shrink_estimator_symmetry(self):
        """协方差矩阵应对称。"""
        returns = _make_returns(n_dates=200, n_stocks=15)
        est = PortfolioCovEstimator(method="shrink")
        cov = est.estimate(returns)
        assert np.allclose(cov.values, cov.values.T, atol=1e-10)

    def test_structured_estimator(self):
        """PCA 因子模型估计器。"""
        returns = _make_returns(n_dates=200, n_stocks=20)
        est = PortfolioCovEstimator(method="structured", estimator_kwargs={
            "factor_model": "pca", "num_factors": 5,
        })
        cov = est.estimate(returns)
        assert cov.shape == (20, 20)

    def test_structured_decomposition(self):
        """structured 方法支持因子分解。"""
        returns = _make_returns(n_dates=200, n_stocks=20)
        est = PortfolioCovEstimator(method="structured", estimator_kwargs={
            "factor_model": "pca", "num_factors": 5,
        })
        result = est.estimate_with_decomposition(returns)
        assert "cov_matrix" in result
        assert result["factors"] is not None
        assert result["factor_cov"] is not None
        assert result["specific_vars"] is not None
        assert result["factors"].shape == (20, 5)
        assert result["factor_cov"].shape == (5, 5)
        assert len(result["specific_vars"]) == 20

    def test_poet_estimator(self):
        """POET 估计器。"""
        returns = _make_returns(n_dates=200, n_stocks=20)
        est = PortfolioCovEstimator(method="poet", estimator_kwargs={
            "num_factors": 3, "thresh": 1.0, "thresh_method": "soft",
        })
        cov = est.estimate(returns)
        assert cov.shape == (20, 20)

    def test_invalid_method_raises(self):
        """无效方法应抛出 ValueError。"""
        with pytest.raises(ValueError, match="unknown method"):
            PortfolioCovEstimator(method="invalid_method")

    def test_custom_estimator_kwargs(self):
        """自定义估计器参数。"""
        returns = _make_returns(n_dates=200, n_stocks=20)
        est = PortfolioCovEstimator(
            method="shrink",
            estimator_kwargs={"alpha": 0.5, "target": "const_var"},
        )
        cov = est.estimate(returns)
        assert cov.shape == (20, 20)

    def test_empty_returns(self):
        """空输入返回空 DataFrame。"""
        est = PortfolioCovEstimator(method="shrink")
        cov = est.estimate(pd.DataFrame())
        assert cov.empty

    def test_estimator_caching(self):
        """多次调用复用同一估计器实例，返回形状一致。"""
        returns = _make_returns(n_dates=100, n_stocks=10)
        est = PortfolioCovEstimator(method="shrink")
        cov1 = est.estimate(returns)
        cov2 = est.estimate(returns)
        assert cov1.shape == cov2.shape
        assert list(cov1.index) == list(cov2.index)

    def test_shrink_without_decomposition(self):
        """shrink 方法不支持因子分解，应优雅降级。"""
        returns = _make_returns(n_dates=200, n_stocks=15)
        est = PortfolioCovEstimator(method="shrink")
        result = est.estimate_with_decomposition(returns)
        assert result["cov_matrix"] is not None
        # shrink doesn't support decomposition, these should be None
        assert result["factors"] is None
        assert result["factor_cov"] is None
        assert result["specific_vars"] is None


# ── 风险指标测试 ────────────────────────────────────────────────

class TestRiskMetrics:
    """风险指标纯函数测试。"""

    def test_portfolio_volatility(self):
        """组合波动率 > 0。"""
        positions = _make_positions(n_stocks=30, n_long=15, n_short=15)
        cov = _make_cov_matrix(n_stocks=30)
        vol = compute_portfolio_volatility(positions, cov)
        assert vol > 0

    def test_portfolio_volatility_annualize(self):
        """年化波动率 = 日波动率 × sqrt(252)。"""
        positions = _make_positions(n_stocks=20, n_long=10, n_short=10)
        cov = _make_cov_matrix(n_stocks=20)
        daily = compute_portfolio_volatility(positions, cov, annualize=False)
        annual = compute_portfolio_volatility(positions, cov, annualize=True)
        assert abs(annual - daily * np.sqrt(252)) < 1e-10

    def test_volatility_zero_for_empty_positions(self):
        """空持仓波动率为 0。"""
        cov = _make_cov_matrix(n_stocks=10)
        vol = compute_portfolio_volatility({}, cov)
        assert vol == 0.0

    def test_var_positive(self):
        """VaR 应为正数。"""
        positions = _make_positions(n_stocks=30, n_long=15, n_short=15)
        cov = _make_cov_matrix(n_stocks=30)
        var = compute_var(positions, cov, confidence=0.95)
        assert var > 0

    def test_var_increases_with_confidence(self):
        """置信度越高 VaR 越大。"""
        positions = _make_positions(n_stocks=30, n_long=15, n_short=15)
        cov = _make_cov_matrix(n_stocks=30)
        var_95 = compute_var(positions, cov, confidence=0.95)
        var_99 = compute_var(positions, cov, confidence=0.99)
        assert var_99 > var_95

    def test_cvar_greater_than_var(self):
        """CVaR >= VaR（同置信度）。"""
        positions = _make_positions(n_stocks=30, n_long=15, n_short=15)
        cov = _make_cov_matrix(n_stocks=30)
        var = compute_var(positions, cov, confidence=0.95)
        cvar = compute_cvar(positions, cov, confidence=0.95)
        assert cvar > var

    def test_risk_decomposition_shape(self):
        """风险分解返回正确列数。"""
        positions = _make_positions(n_stocks=30, n_long=15, n_short=15)
        cov = _make_cov_matrix(n_stocks=30)
        decomp = compute_risk_decomposition(positions, cov)
        assert len(decomp) > 0
        expected_cols = {"instrument", "weight", "marginal_risk", "risk_contribution", "pct_contribution"}
        assert expected_cols.issubset(set(decomp.columns))

    def test_risk_decomposition_pct_sums_to_one(self):
        """风险贡献百分比和为 1。"""
        positions = _make_positions(n_stocks=20, n_long=10, n_short=10)
        cov = _make_cov_matrix(n_stocks=20)
        decomp = compute_risk_decomposition(positions, cov)
        assert abs(decomp["pct_contribution"].sum() - 1.0) < 1e-8

    def test_risk_decomposition_matches_volatility(self):
        """风险贡献总和 = 组合波动率。"""
        positions = _make_positions(n_stocks=20, n_long=10, n_short=10)
        cov = _make_cov_matrix(n_stocks=20)
        vol = compute_portfolio_volatility(positions, cov)
        decomp = compute_risk_decomposition(positions, cov)
        total_contrib = decomp["risk_contribution"].sum()
        assert abs(total_contrib - vol) < 1e-8

    def test_variance_decomposition(self):
        """因子/特质风险分解数值合理。"""
        rng = np.random.default_rng(42)
        n_stocks, n_factors = 30, 5
        n_long, n_short = 15, 15
        instruments = [f"S{i:04d}" for i in range(n_stocks)]

        factors = rng.standard_normal((n_stocks, n_factors))
        factor_cov = rng.standard_normal((n_factors, n_factors))
        factor_cov = factor_cov @ factor_cov.T  # make PSD
        specific_vars = np.abs(rng.standard_normal(n_stocks)) * 0.001

        positions = {}
        for i in range(n_long):
            positions[f"S{i:04d}"] = 1.0 / n_long
        for i in range(n_long, n_long + n_short):
            positions[f"S{i:04d}"] = -1.0 / n_short

        result = compute_variance_decomposition(
            positions, factors, factor_cov, specific_vars,
            instruments=instruments,
        )

        assert result["total_var"] > 0
        assert 0 <= result["factor_pct"] <= 1
        assert 0 <= result["specific_pct"] <= 1
        assert abs(result["factor_pct"] + result["specific_pct"] - 1.0) < 1e-8

    def test_var_horizon_scaling(self):
        """VaR 随 horizon 增大（sqrt 缩放）。"""
        positions = _make_positions(n_stocks=20, n_long=10, n_short=10)
        cov = _make_cov_matrix(n_stocks=20)
        var1 = compute_var(positions, cov, horizon=1)
        var5 = compute_var(positions, cov, horizon=5)
        assert abs(var5 - var1 * np.sqrt(5)) < 1e-6

    def test_volatility_missing_instruments(self):
        """协方差矩阵中有持仓里不存在的股票时，应正确跳过。"""
        positions = {"S0000": 0.5, "S0001": -0.5, "MISSING": 0.3}
        cov = _make_cov_matrix(n_stocks=10)
        vol = compute_portfolio_volatility(positions, cov)
        assert vol > 0

    def test_long_only_portfolio(self):
        """纯多头组合风险指标正常。"""
        positions = {f"S{i:04d}": 1.0 / 30 for i in range(30)}
        cov = _make_cov_matrix(n_stocks=30)
        vol = compute_portfolio_volatility(positions, cov)
        assert vol > 0
        var = compute_var(positions, cov)
        assert var > 0


# ── 导入路径测试 ────────────────────────────────────────────────

class TestRiskModelImports:
    """验证模块可正常导入。"""

    def test_import_from_package(self):
        """通过 riskmodel 包导入（需 3_signal/ 在 path 上）。"""
        _parent = str(Path(__file__).resolve().parents[1] / "3_signal")
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from riskmodel import (
            PortfolioCovEstimator,
            compute_cvar,
            compute_portfolio_volatility,
            compute_risk_decomposition,
            compute_var,
            compute_variance_decomposition,
        )
        assert PortfolioCovEstimator is not None
        assert compute_portfolio_volatility is not None

    def test_direct_import_covariance(self):
        """直接导入 covariance 模块。"""
        from covariance import PortfolioCovEstimator
        est = PortfolioCovEstimator(method="shrink")
        assert est.method == "shrink"

    def test_direct_import_risk_metrics(self):
        """直接导入 risk_metrics 模块。"""
        from risk_metrics import compute_portfolio_volatility
        assert callable(compute_portfolio_volatility)
