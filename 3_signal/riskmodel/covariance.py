"""协方差矩阵估计器封装。

复用 Qlib 内置 RiskModel（ShrinkCov / StructuredCov / POET），
提供项目级默认参数和 DataFrame 输入/输出接口。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from qlib.model.riskmodel import (
    POETCovEstimator,
    RiskModel,
    ShrinkCovEstimator,
    StructuredCovEstimator,
)

logger = logging.getLogger(__name__)

_METHOD_MAP = {
    "shrink": ShrinkCovEstimator,
    "structured": StructuredCovEstimator,
    "poet": POETCovEstimator,
}

_DEFAULT_ESTIMATOR_KWARGS = {
    "shrink": {"alpha": "lw", "target": "const_corr"},
    "structured": {"factor_model": "pca", "num_factors": 10},
    "poet": {"num_factors": 3, "thresh": 1.0, "thresh_method": "soft"},
}


class PortfolioCovEstimator:
    """组合协方差矩阵估计器。

    封装 Qlib RiskModel，支持三种方法：
    - shrink: Ledoit-Wolf 收缩估计（默认，最稳健）
    - structured: PCA/FA 因子模型
    - poet: POET 阈值估计

    参数
    ----
    method : str
        "shrink" | "structured" | "poet"
    estimator_kwargs : dict, optional
        传给底层 Qlib 估计器的参数。
        若为 None 则使用项目默认值。
    nan_option : str
        Qlib RiskModel 的 nan 处理选项（"fill" / "mask" / "ignore"）。
    """

    def __init__(
        self,
        method: str = "shrink",
        estimator_kwargs: Optional[dict] = None,
        nan_option: str = "fill",
    ):
        if method not in _METHOD_MAP:
            raise ValueError(
                f"unknown method: {method}, expected one of {list(_METHOD_MAP.keys())}"
            )
        self._method = method
        self._estimator_cls = _METHOD_MAP[method]

        if estimator_kwargs is None:
            estimator_kwargs = _DEFAULT_ESTIMATOR_KWARGS[method]
        self._estimator_kwargs = estimator_kwargs
        self._nan_option = nan_option

        self._estimator: Optional[RiskModel] = None

    @property
    def method(self) -> str:
        return self._method

    def _get_estimator(self) -> RiskModel:
        if self._estimator is None:
            self._estimator = self._estimator_cls(
                nan_option=self._nan_option,
                **self._estimator_kwargs,
            )
        return self._estimator

    def estimate(
        self,
        stock_returns: pd.DataFrame,
        is_price: bool = False,
    ) -> pd.DataFrame:
        """估计协方差矩阵。

        参数
        ----
        stock_returns : pd.DataFrame
            date × instrument，日收益率（或日价格，若 is_price=True）。
        is_price : bool
            输入是否为价格（True 时自动计算收益率）。

        返回
        ----
        pd.DataFrame
            instrument × instrument 协方差矩阵（日频）。
        """
        if stock_returns.empty:
            logger.warning("empty stock returns, returning empty covariance")
            return pd.DataFrame()

        estimator = self._get_estimator()
        cov = estimator.predict(stock_returns, is_price=is_price)

        if isinstance(cov, pd.DataFrame):
            result = cov
        else:
            result = pd.DataFrame(
                cov,
                index=stock_returns.columns,
                columns=stock_returns.columns,
            )

        logger.info(
            "covariance estimated [%s]: %d × %d, mean diag var=%.6f",
            self._method,
            len(result),
            len(result.columns),
            np.mean(np.diag(result.values)) if len(result) > 0 else 0,
        )
        return result

    def estimate_with_decomposition(
        self,
        stock_returns: pd.DataFrame,
        is_price: bool = False,
    ) -> dict:
        """估计协方差矩阵并返回因子分解。

        仅 StructuredCovEstimator 和 POETCovEstimator 支持分解。
        ShrinkCovEstimator 不支持时返回空字典。

        返回
        ----
        dict:
            cov_matrix : pd.DataFrame     instrument × instrument 协方差矩阵
            factors : np.ndarray or None  instrument × k 因子载荷
            factor_cov : np.ndarray or None  k × k 因子协方差
            specific_vars : np.ndarray or None  每只股票的特质方差
        """
        estimator = self._get_estimator()

        try:
            F, cov_b, var_u = estimator.predict(
                stock_returns,
                is_price=is_price,
                return_decomposed_components=True,
            )
        except (AssertionError, TypeError):
            logger.warning(
                "estimator [%s] does not support decomposition, returning cov only",
                self._method,
            )
            cov = self.estimate(stock_returns, is_price=is_price)
            return {
                "cov_matrix": cov,
                "factors": None,
                "factor_cov": None,
                "specific_vars": None,
            }

        # 同时获取完整协方差矩阵
        cov = self.estimate(stock_returns, is_price=is_price)

        return {
            "cov_matrix": cov,
            "factors": F,
            "factor_cov": cov_b,
            "specific_vars": var_u,
        }
