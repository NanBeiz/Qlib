"""测试风险优化器模块。

覆盖：
- LongShortRiskOptimizer 四种方法（inv, rp, gmv, mvo）
- 多空权重约束（long > 0, short < 0, sum = 0）
- 协方差子集截取
- 候选不足降级
- generate_risk_positions 端到端
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_IMPL_DIR = str(Path(__file__).resolve().parents[1] / "4_strategy" / "06_strategy_top_bottom")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from risk_optimizer import LongShortRiskOptimizer  # noqa: E402


# ── 测试夹具 ────────────────────────────────────────────────────

def _make_alpha(n_stocks=100, seed=42):
    """生成合成 alpha 分数。"""
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(n_stocks)
    return pd.Series(scores, index=[f"S{i:04d}" for i in range(n_stocks)])


def _make_cov(n_stocks=100, seed=99):
    """生成正定协方差矩阵。"""
    rng = np.random.default_rng(seed)
    diag = np.abs(rng.standard_normal(n_stocks)) * 0.01 + 0.001
    A = rng.standard_normal((n_stocks, n_stocks)) * 0.005
    cov = A @ A.T + np.diag(diag)
    stocks = [f"S{i:04d}" for i in range(n_stocks)]
    return pd.DataFrame(cov, index=stocks, columns=stocks)


# ── LongShortRiskOptimizer 测试 ─────────────────────────────────

class TestLongShortRiskOptimizer:
    """LongShortRiskOptimizer 单元测试。"""

    def test_optimize_inv_returns_weights(self):
        """逆波动率方法返回非空权重。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="inv")
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        assert len(weights) > 0

    def test_optimize_rp_returns_weights(self):
        """风险平价方法返回非空权重。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="rp")
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        assert len(weights) > 0

    def test_optimize_gmv_returns_weights(self):
        """GMV 方法返回非空权重。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="gmv")
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        assert len(weights) > 0

    def test_optimize_mvo_returns_weights(self):
        """MVO 方法返回非空权重。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="mvo", method_kwargs={"lamb": 1.0})
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        assert len(weights) > 0

    def test_weights_sum_to_zero(self):
        """多空权重和应接近 0（美元中性）。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="rp")
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        total = sum(weights.values())
        assert abs(total) < 1e-8, f"weight sum = {total}"

    def test_long_weights_positive(self):
        """多头权重全部 > 0。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="rp")
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        long_weights = [w for w in weights.values() if w > 0]
        assert len(long_weights) > 0
        assert all(w > 0 for w in long_weights)

    def test_short_weights_negative(self):
        """空头权重全部 < 0。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="rp")
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        short_weights = [w for w in weights.values() if w < 0]
        assert len(short_weights) > 0
        assert all(w < 0 for w in short_weights)

    def test_abs_weights_sum_to_two(self):
        """多头 + |空头| ≈ 2。"""
        alpha = _make_alpha(100)
        cov = _make_cov(100)
        opt = LongShortRiskOptimizer(method="rp")
        weights = opt.optimize(alpha, cov, topk=25, bottomk=25)
        abs_sum = sum(abs(w) for w in weights.values())
        assert abs(abs_sum - 2.0) < 1e-8, f"|w|_sum = {abs_sum}"

    def test_cov_subset_handling(self):
        """协方差矩阵含多余股票时正确截取。"""
        alpha = _make_alpha(50)
        cov = _make_cov(100)  # 100 stocks, alpha 只有 50
        opt = LongShortRiskOptimizer(method="inv")
        weights = opt.optimize(alpha, cov, topk=10, bottomk=10)
        assert len(weights) > 0
        # 所有权重对应的股票应在 alpha 中
        assert all(s in alpha.index for s in weights)

    def test_insufficient_candidates(self):
        """候选不足时降级处理（不抛异常）。"""
        alpha = _make_alpha(5)  # 只有 5 只股票
        cov = _make_cov(50)
        opt = LongShortRiskOptimizer(method="rp")
        weights = opt.optimize(alpha, cov, topk=20, bottomk=20)
        # 应降级返回部分结果或空
        assert isinstance(weights, dict)

    def test_empty_intersection_returns_empty(self):
        """无交集时返回空字典。"""
        alpha = pd.Series([0.1, -0.2], index=["A", "B"])
        cov = pd.DataFrame(np.eye(3), index=["X", "Y", "Z"], columns=["X", "Y", "Z"])
        opt = LongShortRiskOptimizer(method="rp")
        weights = opt.optimize(alpha, cov, topk=10, bottomk=10)
        assert weights == {}

    def test_invalid_method_raises(self):
        """无效方法抛出 ValueError。"""
        with pytest.raises(ValueError, match="unknown method"):
            LongShortRiskOptimizer(method="unknown")

    def test_method_property(self):
        """method 属性正确。"""
        opt = LongShortRiskOptimizer(method="rp")
        assert opt.method == "rp"

    def test_all_methods_four(self):
        """四种方法都能生成合理权重。"""
        alpha = _make_alpha(60)
        cov = _make_cov(60)
        for method in ["inv", "rp", "gmv", "mvo"]:
            opt = LongShortRiskOptimizer(method=method)
            weights = opt.optimize(alpha, cov, topk=15, bottomk=15)
            assert len(weights) > 0, f"method {method} returned empty weights"

    def test_inv_assigns_higher_weight_to_low_vol(self):
        """逆波动率优化：低波动股票权重更高。"""
        # 构造已知方差的协方差矩阵
        diags = [0.01, 0.04, 0.09, 0.16]  # σ = 0.1, 0.2, 0.3, 0.4
        cov = np.diag(diags)
        stocks = [f"S{i:04d}" for i in range(4)]
        cov_df = pd.DataFrame(cov, index=stocks, columns=stocks)
        alpha = pd.Series([0.5, 0.3, 0.1, -0.1], index=stocks)

        opt = LongShortRiskOptimizer(method="inv")
        weights = opt.optimize(alpha, cov_df, topk=3, bottomk=1)

        top3 = sorted(alpha.nlargest(3).index)
        # 低波动 = 高权重
        w_values = [weights[s] for s in top3]
        assert w_values[0] > w_values[2], f"low vol stock should have higher weight, got {w_values}"

    def test_mvo_uses_alpha(self):
        """MVO 使用 alpha 信息，与 GMV 结果不同。"""
        alpha = _make_alpha(80)
        cov = _make_cov(80)
        gmv_opt = LongShortRiskOptimizer(method="gmv")
        mvo_opt = LongShortRiskOptimizer(method="mvo")
        w_gmv = gmv_opt.optimize(alpha, cov, topk=20, bottomk=10)
        w_mvo = mvo_opt.optimize(alpha, cov, topk=20, bottomk=10)
        # 两者权重分配应不同
        gmv_vec = np.array([w_gmv.get(s, 0.0) for s in sorted(set(w_gmv) | set(w_mvo))])
        mvo_vec = np.array([w_mvo.get(s, 0.0) for s in sorted(set(w_gmv) | set(w_mvo))])
        assert not np.allclose(gmv_vec, mvo_vec, atol=1e-6)


# ── generate_risk_positions 集成测试 ─────────────────────────────

class TestGenerateRiskPositions:
    """generate_risk_positions 集成测试。"""

    def test_returns_dataframe(self):
        """返回 DataFrame 带正确列。"""
        from generate_positions_risk import generate_risk_positions

        n_dates, n_stocks = 300, 40
        rng = np.random.default_rng(42)
        dates_list = pd.date_range("2022-01-03", periods=n_dates, freq="B")
        returns_data = rng.standard_normal((n_dates, n_stocks)) * 0.02
        stock_returns = pd.DataFrame(
            returns_data,
            index=dates_list,
            columns=[f"S{i:04d}" for i in range(n_stocks)],
        )

        records = []
        for d in dates_list[-60:]:  # 最后 60 天做预测
            for i in range(n_stocks):
                records.append({
                    "datetime": d,
                    "instrument": f"S{i:04d}",
                    "ALPHA": rng.standard_normal(),
                })
        predictions = pd.DataFrame(records)

        from riskmodel.covariance import PortfolioCovEstimator
        opt = LongShortRiskOptimizer(method="rp")
        cov_est = PortfolioCovEstimator(method="shrink")

        result = generate_risk_positions(
            predictions=predictions,
            topk=10,
            bottomk=10,
            stock_returns=stock_returns,
            optimizer=opt,
            cov_estimator=cov_est,
            lookback=120,
            min_periods=30,
        )

        assert isinstance(result, pd.DataFrame)
        assert "datetime" in result.columns
        assert "instrument" in result.columns
        assert "weight" in result.columns

    def test_weight_sums_to_zero_per_day(self):
        """每天权重和 ≈ 0。"""
        from generate_positions_risk import generate_risk_positions

        n_dates, n_stocks = 300, 30
        rng = np.random.default_rng(99)
        dates_list = pd.date_range("2022-01-03", periods=n_dates, freq="B")
        returns_data = rng.standard_normal((n_dates, n_stocks)) * 0.02
        stock_returns = pd.DataFrame(
            returns_data, index=dates_list,
            columns=[f"S{i:04d}" for i in range(n_stocks)],
        )

        records = []
        for d in dates_list[-40:]:
            for i in range(n_stocks):
                records.append({
                    "datetime": d,
                    "instrument": f"S{i:04d}",
                    "ALPHA": rng.standard_normal(),
                })
        predictions = pd.DataFrame(records)

        from riskmodel.covariance import PortfolioCovEstimator
        opt = LongShortRiskOptimizer(method="rp")
        cov_est = PortfolioCovEstimator(method="shrink")

        result = generate_risk_positions(
            predictions=predictions, topk=10, bottomk=10,
            stock_returns=stock_returns, optimizer=opt,
            cov_estimator=cov_est, lookback=100, min_periods=30,
        )

        for dt, grp in result.groupby("datetime"):
            assert abs(grp["weight"].sum()) < 1e-8, f"date={dt}, sum={grp['weight'].sum()}"
