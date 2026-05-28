"""对冲管理器。

根据组合 Beta 暴露计算最优期货对冲手数，支持以下对冲目标：

- full      : 完全对冲，目标组合 Beta = 0
- partial   : 部分对冲，保留指定比例的 Beta 暴露
- target    : 将组合 Beta 调整到目标值

核心公式：

    hedge_notional = net_beta × portfolio_value × (1 - target_ratio)
    contracts = hedge_notional / (futures_price × multiplier)

若期货 Beta ≠ 1（期货对基准的 Beta 不完全等于 1），引入期货 Beta 调整：

    contracts = hedge_notional / (futures_price × multiplier × futures_beta)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from futures_data_interface import FuturesDataProvider

logger = logging.getLogger(__name__)


# ── 对冲结果数据类 ───────────────────────────────────────────────

@dataclass
class HedgeResult:
    """单次对冲计算结果。

    Attributes
    ----------
    date : pd.Timestamp
        调仓日。
    net_beta : float
        对冲前组合净 Beta 暴露。
    hedge_ratio : float
        对冲比例（0~1），1 = 完全对冲。
    contracts : float
        需要卖出的期货合约数量（正 = 做空期货对冲多头 Beta）。
    hedge_notional : float
        对冲所需名义金额（元）。
    futures_price : float
        所用期货价格。
    futures_symbol : str
        所用期货合约代码。
    required_margin : float
        所需保证金（元）。
    notes : str
        备注信息。
    """

    date: pd.Timestamp
    net_beta: float
    hedge_ratio: float
    contracts: float
    hedge_notional: float
    futures_price: float
    futures_symbol: str
    required_margin: float
    notes: str = ""


# ── 对冲管理器 ──────────────────────────────────────────────────

class HedgeManager:
    """期货 Beta 对冲管理器。

    在组合构建后调用，根据组合净 Beta 计算应卖出的期货手数。

    Parameters
    ----------
    futures_provider : FuturesDataProvider
        期货数据提供者。
    hedge_symbol : str
        用于对冲的期货合约代码，默认 "IC"（中证 500 股指期货）。
    hedge_mode : str
        对冲模式：
        - "full": 完全对冲，目标 Beta = 0
        - "partial": 部分对冲，保留 hedge_ratio 比例的 Beta
    hedge_ratio_target : float
        对冲比例（仅 hedge_mode="partial" 时生效）。
        0 = 不对冲，1 = 完全对冲，0.5 = 对冲一半 Beta。
    futures_beta : float
        期货对基准指数的 Beta。通常 IC 对 CSI500 的 Beta ≈ 1，
        但若存在基差风险可调整此参数。
    min_contracts : float
        最小对冲手数阈值，低于此值不执行对冲（避免过度交易）。
    max_hedge_ratio : float
        单次对冲比例上限，防止极端仓位。
    """

    def __init__(
        self,
        futures_provider: FuturesDataProvider,
        hedge_symbol: str = "IC",
        hedge_mode: str = "full",
        hedge_ratio_target: float = 1.0,
        futures_beta: float = 1.0,
        min_contracts: float = 0.5,
        max_hedge_ratio: float = 1.5,
    ):
        self._provider = futures_provider
        self._hedge_symbol = hedge_symbol
        self._hedge_mode = hedge_mode
        self._hedge_ratio_target = hedge_ratio_target
        self._futures_beta = futures_beta
        self._min_contracts = min_contracts
        self._max_hedge_ratio = max_hedge_ratio

        self._contract_spec = futures_provider.get_contract_spec(hedge_symbol)

        logger.info(
            "HedgeManager initialized: symbol=%s, mode=%s, target=%.2f, "
            "multiplier=%.0f, margin=%.1f%%",
            hedge_symbol,
            hedge_mode,
            hedge_ratio_target,
            self._contract_spec.multiplier,
            self._contract_spec.margin_ratio * 100,
        )

    # ── 主计算入口 ───────────────────────────────────────────

    def compute_hedge(
        self,
        date: pd.Timestamp,
        net_beta: float,
        portfolio_value: float,
        long_exposure: float = 1.0,
        short_exposure: float = 1.0,
    ) -> Optional[HedgeResult]:
        """计算给定调仓日的期货对冲手数。

        Parameters
        ----------
        date : pd.Timestamp
            调仓日。
        net_beta : float
            对冲前组合净 Beta 暴露。
        portfolio_value : float
            组合总资产价值（含多空两条腿的市值，通常取净资产的绝对值之和）。
            简化情景下可用净资产，但推荐使用 long_leg_value + abs(short_leg_value)。
        long_exposure : float
            多头总权重（如 1.0 = 100%）。
        short_exposure : float
            空头总权重绝对值（如 1.0 = 100%）。

        Returns
        -------
        HedgeResult or None
            None 表示无法执行对冲（数据缺失或无需对冲）。
        """
        # 获取期货价格
        futures_price = self._provider.get_price(self._hedge_symbol, date)
        if futures_price is None or futures_price <= 0:
            logger.warning(
                "futures price unavailable for %s on %s, skipping hedge",
                self._hedge_symbol,
                date.date(),
            )
            return None

        # 确定对冲比例
        hedge_ratio = self._resolve_hedge_ratio()

        # 计算需要对冲的 Beta 暴露
        beta_to_hedge = net_beta * hedge_ratio

        if abs(beta_to_hedge) < 1e-8:
            return HedgeResult(
                date=date,
                net_beta=net_beta,
                hedge_ratio=hedge_ratio,
                contracts=0.0,
                hedge_notional=0.0,
                futures_price=futures_price,
                futures_symbol=self._hedge_symbol,
                required_margin=0.0,
                notes="net beta near zero, no hedge needed",
            )

        # 对冲名义金额 = Beta 暴露 × 组合价值
        gross_exposure = long_exposure + short_exposure
        hedge_notional = beta_to_hedge * portfolio_value

        # 合约数量 = 对冲名义金额 / (期货价格 × 乘数 × 期货 Beta)
        multiplier = self._contract_spec.multiplier
        contract_value = futures_price * multiplier * self._futures_beta
        raw_contracts = hedge_notional / contract_value

        # 整数手数（向下取整，保守），保留方向
        # raw_contracts > 0 → 做空期货（负手数）对冲多头 Beta
        # raw_contracts < 0 → 做多期货（正手数）对冲空头 Beta
        abs_contracts = self._round_contracts(raw_contracts)
        contracts = -abs_contracts if raw_contracts >= 0 else abs_contracts

        if abs(contracts) < self._min_contracts:
            return HedgeResult(
                date=date,
                net_beta=net_beta,
                hedge_ratio=hedge_ratio,
                contracts=0.0,
                hedge_notional=hedge_notional,
                futures_price=futures_price,
                futures_symbol=self._hedge_symbol,
                required_margin=0.0,
                notes=f"contracts ({raw_contracts:.3f}) < min ({self._min_contracts})",
            )

        # 保证金计算
        margin_required = abs(contracts) * contract_value * self._contract_spec.margin_ratio

        result = HedgeResult(
            date=date,
            net_beta=net_beta,
            hedge_ratio=hedge_ratio,
            contracts=contracts,  # 正=做多期货, 负=做空期货
            hedge_notional=hedge_notional,
            futures_price=futures_price,
            futures_symbol=self._hedge_symbol,
            required_margin=margin_required,
            notes="ok",
        )

        logger.info(
            "hedge %s: net_beta=%.4f, hedge_notional=%.0f, price=%.1f, "
            "contracts=%.1f, margin=%.0f",
            date.date(),
            net_beta,
            hedge_notional,
            futures_price,
            contracts,
            margin_required,
        )

        return result

    # ── 内部方法 ───────────────────────────────────────────

    def _resolve_hedge_ratio(self) -> float:
        """解析最终使用的对冲比例。"""
        if self._hedge_mode == "full":
            return 1.0
        if self._hedge_mode == "partial":
            return max(0.0, min(self._hedge_ratio_target, self._max_hedge_ratio))
        logger.warning("unknown hedge_mode: %s, defaulting to full", self._hedge_mode)
        return 1.0

    @staticmethod
    def _round_contracts(raw: float) -> float:
        """向下取整，确保不超对冲。"""
        return float(int(abs(raw)))

    # ── 批量计算 ───────────────────────────────────────────

    def compute_hedge_timeseries(
        self,
        dates: list[pd.Timestamp],
        net_betas: list[float],
        portfolio_values: list[float],
        long_exposures: Optional[list[float]] = None,
        short_exposures: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """对一系列调仓日批量计算对冲。

        Returns
        -------
        pd.DataFrame
            每行的 HedgeResult 字段。
        """
        records = []
        n = len(dates)
        for i in range(n):
            long_exp = long_exposures[i] if long_exposures else 1.0
            short_exp = short_exposures[i] if short_exposures else 1.0
            result = self.compute_hedge(
                date=dates[i],
                net_beta=net_betas[i],
                portfolio_value=portfolio_values[i],
                long_exposure=long_exp,
                short_exposure=short_exp,
            )
            if result is not None:
                records.append({
                    "date": result.date,
                    "net_beta": result.net_beta,
                    "hedge_ratio": result.hedge_ratio,
                    "contracts": result.contracts,
                    "hedge_notional": result.hedge_notional,
                    "futures_price": result.futures_price,
                    "futures_symbol": result.futures_symbol,
                    "required_margin": result.required_margin,
                    "notes": result.notes,
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        return df
