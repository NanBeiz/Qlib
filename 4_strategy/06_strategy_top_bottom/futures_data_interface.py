"""期货数据抽象接口。

当前为占位实现，等期货数据就绪后替换为实际数据源。
所有方法签名保持稳定，下游代码无需修改。

支持的数据源：
1. CsvFuturesDataProvider  — 从 CSV 文件读取（当前占位实现）
2. 未来可扩展：WindDataProvider, TushareDataProvider 等
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── 期货合约元数据 ──────────────────────────────────────────────

@dataclass(frozen=True)
class FuturesContractSpec:
    """期货合约基本规格。

    Attributes
    ----------
    symbol : str
        合约代码，如 "IC"、"IF"。
    multiplier : float
        合约乘数，如 IC 为 200 元/点。
    margin_ratio : float
        保证金比例，如 0.12（12%）。
    """

    symbol: str
    multiplier: float
    margin_ratio: float


# 中金所主要股指期货合约规格
CONTRACT_SPECS: dict[str, FuturesContractSpec] = {
    "IC": FuturesContractSpec(symbol="IC", multiplier=200.0, margin_ratio=0.12),
    "IF": FuturesContractSpec(symbol="IF", multiplier=300.0, margin_ratio=0.12),
    "IH": FuturesContractSpec(symbol="IH", multiplier=300.0, margin_ratio=0.12),
    "IM": FuturesContractSpec(symbol="IM", multiplier=200.0, margin_ratio=0.12),
}


# ── 抽象接口 ────────────────────────────────────────────────────

class FuturesDataProvider(ABC):
    """期货数据提供者抽象基类。

    子类必须实现 get_price() 和 get_contract_multiplier()。
    """

    @abstractmethod
    def get_price(
        self,
        symbol: str,
        date: pd.Timestamp,
    ) -> Optional[float]:
        """获取指定期货合约在某交易日的收盘价（或主力连续价格）。

        Parameters
        ----------
        symbol : str
            合约代码，如 "IC"。
        date : pd.Timestamp
            交易日。

        Returns
        -------
        float or None
            该交易日收盘价。若数据缺失返回 None。
        """
        ...

    @abstractmethod
    def get_contract_multiplier(self, symbol: str) -> float:
        """获取合约乘数。

        Parameters
        ----------
        symbol : str
            合约代码。

        Returns
        -------
        float
        """
        ...

    def get_contract_spec(self, symbol: str) -> FuturesContractSpec:
        """获取完整合约规格，默认查 CONTRACT_SPECS 表。

        子类可重写以支持动态规格。
        """
        if symbol in CONTRACT_SPECS:
            return CONTRACT_SPECS[symbol]
        raise KeyError(f"unknown contract symbol: {symbol}")


# ── CSV 占位实现 ────────────────────────────────────────────────

class CsvFuturesDataProvider(FuturesDataProvider):
    """基于 CSV 文件的期货数据提供者。

    用于没有实时期货数据源时的本地回测占位。
    等实际数据就绪后，替换为 Wind/Tushare 等提供者即可。

    CSV 格式要求（主力连续合约）:

    .. code-block:: text

        date,IC,IF,IH,IM
        2022-01-04,7105.0,4912.4,3250.6,7851.2
        2022-01-05,7132.8,4900.0,3238.4,7812.0

    Parameters
    ----------
    csv_path : str or Path
        CSV 文件路径。
    """

    def __init__(self, csv_path: str | Path):
        self._csv_path = Path(csv_path)
        self._prices: Optional[pd.DataFrame] = None

    def _ensure_loaded(self) -> None:
        """惰性加载 CSV，仅在首次查询时读取。"""
        if self._prices is not None:
            return

        if not self._csv_path.exists():
            logger.warning(
                "futures CSV not found: %s — hedge will be skipped. "
                "Expected columns: [date, IC, IF, IH, IM]",
                self._csv_path,
            )
            self._prices = pd.DataFrame()
            return

        df = pd.read_csv(self._csv_path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        self._prices = df
        logger.info(
            "loaded futures data: %d rows, symbols=%s",
            len(df),
            list(df.columns),
        )

    def get_price(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        self._ensure_loaded()
        if self._prices.empty or symbol not in self._prices.columns:
            return None

        # 日期精确匹配
        if date in self._prices.index:
            return float(self._prices.loc[date, symbol])

        # 向前查找最近交易日（容错
        earlier = self._prices.index[self._prices.index <= date]
        if len(earlier) == 0:
            return None
        return float(self._prices.loc[earlier[-1], symbol])

    def get_contract_multiplier(self, symbol: str) -> float:
        return self.get_contract_spec(symbol).multiplier

    def get_price_series(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """获取区间内完整价格序列，用于 Beta 估计中的指数替代。"""
        self._ensure_loaded()
        if self._prices.empty or symbol not in self._prices.columns:
            return pd.Series(dtype=float)
        return self._prices.loc[start:end, symbol]


# ── Mock 数据提供者（纯回测占位） ────────────────────────────────

class MockFuturesDataProvider(FuturesDataProvider):
    """Mock 期货数据提供者，返回固定价格。

    仅用于单元测试和需要期货数据但暂缺时的占位。
    实际部署时必须替换为真实数据源。

    Parameters
    ----------
    mock_price : float
        返回的固定价格。
    mock_multiplier : float
        返回的固定乘数。
    """

    def __init__(self, mock_price: float = 6000.0, mock_multiplier: float = 200.0):
        self._price = mock_price
        self._multiplier = mock_multiplier

    def get_price(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        return self._price

    def get_contract_multiplier(self, symbol: str) -> float:
        return self._multiplier
