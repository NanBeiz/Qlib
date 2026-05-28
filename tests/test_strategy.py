"""测试 TopBottomNeutralStrategy 权重生成。"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_csi500_mn.strategies.top_bottom_neutral import TopBottomNeutralStrategy


class MockPosition:
    """最小化 mock Qlib Position。"""

    def get_stock_list(self):
        return []

    def get_stock_amount(self, code):
        return 0

    def get_cash(self):
        return 1e8

    def get_stock_weight_dict(self, only_stock=True):
        return {}


def _make_scores(n_stocks: int = 100, seed: int = 42) -> pd.Series:
    """生成随机分数序列。"""
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.standard_normal(n_stocks),
        index=[f"STOCK_{i:04d}" for i in range(n_stocks)],
        name="score",
    )


class TestTopBottomNeutralStrategy:
    """TopBottomNeutralStrategy 单元测试。"""

    @patch("quant_csi500_mn.strategies.top_bottom_neutral.D")
    def test_rebalance_day_weights(self, mock_D):
        """月末日：验证权重市场中性 (sum≈0, |w|_sum=2)。"""
        # Mock D.calendar 返回当月末日期
        trade_date = pd.Timestamp("2024-01-31")
        mock_D.calendar.return_value = [pd.Timestamp("2024-01-02"), trade_date]

        scores = _make_scores(100)
        strategy = TopBottomNeutralStrategy(
            topk=20,
            bottomk=20,
            signal=pd.Series(dtype=float),
        )

        weights = strategy.generate_target_weight_position(
            score=scores,
            current=MockPosition(),
            trade_start_time=trade_date,
            trade_end_time=trade_date,
        )

        assert weights is not None, "Expected weights on rebalance day"

        total_weight = sum(weights.values())
        assert abs(total_weight) < 1e-10, f"Weight sum={total_weight}, expected ~0"

        abs_sum = sum(abs(w) for w in weights.values())
        assert abs(abs_sum - 2.0) < 1e-10, f"Abs weight sum={abs_sum}, expected 2.0"

        n_long = sum(1 for w in weights.values() if w > 0)
        n_short = sum(1 for w in weights.values() if w < 0)
        assert n_long == 20, f"Long count={n_long}, expected 20"
        assert n_short == 20, f"Short count={n_short}, expected 20"

        # 验证等权重
        long_weights = [w for w in weights.values() if w > 0]
        short_weights = [w for w in weights.values() if w < 0]
        for w in long_weights:
            assert abs(w - 1.0 / 20) < 1e-10
        for w in short_weights:
            assert abs(w + 1.0 / 20) < 1e-10

    @patch("quant_csi500_mn.strategies.top_bottom_neutral.D")
    def test_non_rebalance_day_returns_none(self, mock_D):
        """非月末日：验证返回 None（保持持仓）。"""
        # Mock D.calendar: 月末日为 2024-01-31，当前为 2024-01-15
        mock_D.calendar.return_value = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-31"),
        ]

        scores = _make_scores(50)
        strategy = TopBottomNeutralStrategy(topk=10, bottomk=10, signal=pd.Series(dtype=float))

        weights = strategy.generate_target_weight_position(
            score=scores,
            current=MockPosition(),
            trade_start_time=pd.Timestamp("2024-01-15"),
            trade_end_time=pd.Timestamp("2024-01-15"),
        )

        assert weights is None, "Expected None on non-rebalance day"

    @patch("quant_csi500_mn.strategies.top_bottom_neutral.D")
    def test_nan_scores_are_excluded(self, mock_D):
        """验证 NaN score 被排除，不影响选股数量。"""
        trade_date = pd.Timestamp("2024-01-31")
        mock_D.calendar.return_value = [pd.Timestamp("2024-01-02"), trade_date]

        scores = _make_scores(100)
        # 故意把最高分的前 5 只股票设为 NaN
        top5 = scores.nlargest(5).index
        scores[top5] = np.nan

        strategy = TopBottomNeutralStrategy(topk=20, bottomk=20, signal=pd.Series(dtype=float))
        weights = strategy.generate_target_weight_position(
            score=scores,
            current=MockPosition(),
            trade_start_time=trade_date,
            trade_end_time=trade_date,
        )

        # NaN 股票不应出现在权重中
        for stock in top5:
            assert stock not in weights, f"NaN stock {stock} should not be in weights"

        assert len(weights) == 40  # 20 long + 20 short

    @patch("quant_csi500_mn.strategies.top_bottom_neutral.D")
    def test_insufficient_scores_returns_none(self, mock_D):
        """当有效 score 不足时返回 None。"""
        trade_date = pd.Timestamp("2024-01-31")
        mock_D.calendar.return_value = [pd.Timestamp("2024-01-02"), trade_date]

        # 只有 10 只股票，但 topk=20
        scores = _make_scores(10)
        strategy = TopBottomNeutralStrategy(topk=20, bottomk=20, signal=pd.Series(dtype=float))

        weights = strategy.generate_target_weight_position(
            score=scores,
            current=MockPosition(),
            trade_start_time=trade_date,
            trade_end_time=trade_date,
        )

        assert weights is None, "Expected None when insufficient scores"
