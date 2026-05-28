"""截面行业+对数市值中性化 Processor。

继承 qlib.data.dataset.processor.Processor，对每个交易日截面，
将 feature 列对 [行业虚拟变量, log(mcap), 截距] 做 OLS 回归取残差。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.data.dataset.processor import Processor
from qlib.log import get_module_logger

logger = get_module_logger(__name__)


class IndustryMcapNeutralize(Processor):
    """截面行业 + 对数市值中性化处理器。

    对 fields_group 指定的每组 feature 列，在每个交易日截面上，
    用 OLS 回归剔除行业和市值的影响，保留残差作为中性化后的因子值。

    Parameters
    ----------
    fields_group : str
        待中性化的字段组名，默认 "feature"。
    industry_prefix : str
        行业虚拟变量列名前缀，默认 "IND_"。
    mcap_col : str
        对数市值列名，默认 "LOG_MCAP"。
    """

    def __init__(
        self,
        fields_group: str = "feature",
        industry_prefix: str = "IND_",
        mcap_col: str = "LOG_MCAP",
    ):
        self.fields_group = fields_group
        self.industry_prefix = industry_prefix
        self.mcap_col = mcap_col

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行截面中性化（原地修改）。

        Parameters
        ----------
        df : pd.DataFrame
            含 MultiIndex 列（level 0 = field group, level 1 = field name）
            和扁平控制变量列（IND_XXX, LOG_MCAP）的数据，索引为 (datetime, instrument)。

        Returns
        -------
        pd.DataFrame
            原地修改后的 DataFrame。
        """
        feature_cols = self._get_feature_columns(df)
        if feature_cols.empty:
            logger.debug("no feature columns found for group=%s", self.fields_group)
            return df

        control_cols = self._get_control_columns(df)
        logger.info(
            "neutralizing %d features with %d control variables",
            len(feature_cols),
            len(control_cols),
        )

        for dt, group_idx in df.groupby(level="datetime", sort=False).groups.items():
            self._neutralize_slice(df, group_idx, feature_cols, control_cols)

        return df

    def is_for_infer(self) -> bool:
        """在 train / valid / test 所有阶段都运行。"""
        return True

    def readonly(self) -> bool:
        """会原地修改数据。"""
        return False

    # ---------- 内部方法 ----------

    def _get_feature_columns(self, df: pd.DataFrame) -> pd.Index:
        """提取指定 fields_group 下的 feature 列名。"""
        if not isinstance(df.columns, pd.MultiIndex):
            return pd.Index([])
        try:
            return df.columns.get_level_values(-1)[
                df.columns.get_level_values(0) == self.fields_group
            ]
        except KeyError:
            return pd.Index([])

    def _get_control_columns(self, df: pd.DataFrame) -> list[str]:
        """提取控制变量列名（行业虚拟变量 + 市值）。"""
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = df.columns.get_level_values(-1)
        else:
            flat_cols = df.columns
        cols = [c for c in flat_cols if c.startswith(self.industry_prefix)]
        if self.mcap_col in flat_cols:
            cols.append(self.mcap_col)
        return cols

    @staticmethod
    def _get_column_values(df: pd.DataFrame, col_name: str) -> pd.Series:
        """获取列值，兼容 MultiIndex 和扁平列。

        对 MultiIndex 列，匹配最后一层的列名。
        """
        if isinstance(df.columns, pd.MultiIndex):
            mask = df.columns.get_level_values(-1) == col_name
            if mask.any():
                key = df.columns[mask][0]
                return df[key]
            raise KeyError(f"Column {col_name} not found in MultiIndex columns")
        return df[col_name]

    def _neutralize_slice(
        self,
        df: pd.DataFrame,
        group_idx: pd.Index,
        feature_cols: pd.Index,
        control_cols: list[str],
    ) -> None:
        """对单个截面执行 OLS 中性化。"""
        group = df.loc[group_idx]
        n = len(group)

        # 构建设计矩阵 X: [行业虚拟变量, log(mcap), 截距]
        X_parts = []
        for col in control_cols:
            col_data = self._get_column_values(group, col)
            X_parts.append(col_data.values.astype(np.float64))
        X = np.column_stack(X_parts) if X_parts else np.empty((n, 0))
        X = np.column_stack([X, np.ones(n, dtype=np.float64)])

        for col in feature_cols:
            full_col = (self.fields_group, col)
            y = group[full_col].values.astype(np.float64)
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

            residual = np.full(n, np.nan, dtype=np.float64)
            if mask.sum() >= X.shape[1] + 1:
                beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
                residual[mask] = y[mask] - X[mask] @ beta
            df.loc[group_idx, full_col] = residual
