"""Beta hedging components for the CSI500 market-neutral strategy.

This is a thin re-export layer. All implementation lives in
4_strategy/06_strategy_top_bottom/ for file isolation per project convention.
"""

import sys
from pathlib import Path

_impl_dir = Path(__file__).resolve().parents[3] / "4_strategy" / "06_strategy_top_bottom"
if str(_impl_dir) not in sys.path:
    sys.path.insert(0, str(_impl_dir))

from futures_data_interface import (  # noqa: E402, F401
    CONTRACT_SPECS,
    CsvFuturesDataProvider,
    FuturesContractSpec,
    FuturesDataProvider,
    MockFuturesDataProvider,
)
from hedge_manager import HedgeManager, HedgeResult  # noqa: E402, F401
from portfolio_beta import (  # noqa: E402, F401
    compute_long_short_betas,
    compute_portfolio_beta,
    estimate_stock_betas,
)

__all__ = [
    "CONTRACT_SPECS",
    "CsvFuturesDataProvider",
    "FuturesContractSpec",
    "FuturesDataProvider",
    "MockFuturesDataProvider",
    "HedgeManager",
    "HedgeResult",
    "compute_long_short_betas",
    "compute_portfolio_beta",
    "estimate_stock_betas",
]
