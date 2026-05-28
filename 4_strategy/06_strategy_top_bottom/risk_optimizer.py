"""多空风险优化器。

将多空组合拆为两条腿分别调用 Qlib PortfolioOptimizer：
- 多头腿：topk 股票，w >= 0, sum(w) = 1
- 空头腿：bottomk 股票，w >= 0, sum(w) = 1 → 取负

支持四种方法：inv（逆波动率）、rp（风险平价）、gmv（全局最小方差）、mvo（均值-方差）。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from qlib.contrib.strategy.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

# Qlib 优化器支持的方法
_METHODS = {"inv", "rp", "gmv", "mvo"}

# 默认参数
_DEFAULT_METHOD_KWARGS = {
    "mvo": {"lamb": 1.0},
}


class LongShortRiskOptimizer:
    """多空风险优化器。

    参数
    ----
    method : str
        "inv" | "rp" | "gmv" | "mvo"
    method_kwargs : dict, optional
        传给 Qlib PortfolioOptimizer 的额外参数。
    """

    def __init__(
        self,
        method: str = "rp",
        method_kwargs: Optional[dict] = None,
    ):
        if method not in _METHODS:
            raise ValueError(f"unknown method: {method}, expected one of {sorted(_METHODS)}")
        self._method = method

        if method_kwargs is None:
            method_kwargs = _DEFAULT_METHOD_KWARGS.get(method, {}).copy()
        self._method_kwargs = method_kwargs

    @property
    def method(self) -> str:
        return self._method

    def optimize(
        self,
        alpha_scores: pd.Series,
        cov_matrix: pd.DataFrame,
        topk: int = 50,
        bottomk: int = 50,
    ) -> dict[str, float]:
        """多空风险优化。

        参数
        ----
        alpha_scores : pd.Series
            index=instrument, values=alpha score，越大越好。
        cov_matrix : pd.DataFrame
            instrument × instrument 协方差矩阵。
        topk : int
            多头候选数量。
        bottomk : int
            空头候选数量。

        返回
        ----
        dict[str, float]
            {instrument: weight}，多头为正，空头为负。
        """
        # 过滤共有股票
        common = sorted(set(alpha_scores.index) & set(cov_matrix.index) & set(cov_matrix.columns))
        if len(common) < max(topk, bottomk):
            logger.warning(
                "insufficient common instruments: %d < max(topk=%d, bottomk=%d)",
                len(common), topk, bottomk,
            )
            if len(common) < 2:
                return {}

        alpha = alpha_scores[common].dropna().sort_values(ascending=False)
        if len(alpha) < max(topk, bottomk):
            logger.warning("insufficient valid alpha scores: %d", len(alpha))

        # 多头候选
        long_n = min(topk, len(alpha))
        long_stocks = list(alpha.index[:long_n])

        # 空头候选
        short_n = min(bottomk, len(alpha))
        short_stocks = list(alpha.index[-short_n:])

        # 去掉多头和空头的重叠
        long_set = set(long_stocks)
        short_stocks = [s for s in short_stocks if s not in long_set]

        if len(long_stocks) == 0 and len(short_stocks) == 0:
            return {}

        weights: dict[str, float] = {}

        # 只保留 cov_matrix 中存在的股票
        cov_instruments = set(cov_matrix.index) & set(cov_matrix.columns)
        long_stocks = [s for s in long_stocks if s in cov_instruments]
        short_stocks = [s for s in short_stocks if s in cov_instruments]

        # --- 多头腿 ---
        if long_stocks:
            long_cov = cov_matrix.loc[long_stocks, long_stocks]
            long_w = self._optimize_leg(
                cov_sub=long_cov,
                alpha_sub=alpha[long_stocks] if self._method == "mvo" else None,
            )
            for inst, w in zip(long_stocks, long_w):
                if w > 1e-8:
                    weights[inst] = float(w)

        # --- 空头腿 ---
        if short_stocks:
            short_cov = cov_matrix.loc[short_stocks, short_stocks]
            # mvo 时用 -alpha（越差越应该做空，对优化器来说就是更高"收益"）
            short_alpha = None
            if self._method == "mvo":
                short_alpha = -alpha[short_stocks]
            short_w = self._optimize_leg(
                cov_sub=short_cov,
                alpha_sub=short_alpha,
            )
            for inst, w in zip(short_stocks, short_w):
                if w > 1e-8:
                    weights[inst] = -float(w)

        logger.info(
            "risk optimization [%s]: long=%d stocks, short=%d stocks, total_abs_weight=%.2f",
            self._method,
            len(long_stocks),
            len(short_stocks),
            sum(abs(v) for v in weights.values()),
        )
        return weights

    def _optimize_leg(
        self,
        cov_sub: pd.DataFrame,
        alpha_sub: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """优化单条腿，含 fallback 链：method → inv → 等权。"""
        # 1. 用指定方法
        w = self._try_optimize(cov_sub, self._method, alpha_sub)
        if w is not None:
            return w

        # 2. fallback: 逆波动率（稳定，不需要 scipy 优化）
        if self._method != "inv":
            logger.info("falling back to inv for %d stocks", len(cov_sub))
            w = self._try_optimize(cov_sub, "inv", None)
            if w is not None:
                return w

        # 3. 最终 fallback: 等权
        n = len(cov_sub)
        logger.warning("all optimization failed, falling back to equal weight (%d stocks)", n)
        return np.ones(n) / n

    @staticmethod
    def _try_optimize(
        cov_sub: pd.DataFrame,
        method: str,
        alpha_sub: Optional[pd.Series] = None,
    ) -> Optional[np.ndarray]:
        """尝试单次优化，失败返回 None。"""
        optimizer = PortfolioOptimizer(method=method)
        r = None
        if alpha_sub is not None and method == "mvo":
            r = pd.Series(alpha_sub.values, index=cov_sub.index)
        try:
            w = optimizer(cov_sub, r=r)
            if isinstance(w, pd.Series):
                w = w.values
            return w
        except Exception:
            return None
