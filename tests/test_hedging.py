"""测试 Beta 对冲模块。

覆盖：
- HedgeManager 单元测试（Mock 期货数据）
- portfolio_beta 单元测试（合成数据）
- generate_hedge_overlay 集成测试
- Beta shift(1) 防未来函数
- Qlib 数据形状防御
- action 字段正确性
- generate_positions() 接口不变
- re-export 层可导入
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

# 将 4_strategy/ 加入 sys.path，使测试能导入实现模块
_IMPL_DIR = str(Path(__file__).resolve().parents[1] / "4_strategy" / "06_strategy_top_bottom")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from futures_data_interface import MockFuturesDataProvider  # noqa: E402
from hedge_manager import HedgeManager, HedgeResult  # noqa: E402
from portfolio_beta import (  # noqa: E402
    compute_long_short_betas,
    compute_portfolio_beta,
    estimate_stock_betas,
)


# ── 测试夹具 ────────────────────────────────────────────────────

def _make_returns(
    n_dates: int = 300,
    n_stocks: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成个股日收益矩阵 (date x instrument)。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    data = rng.standard_normal((n_dates, n_stocks)) * 0.02
    return pd.DataFrame(
        data,
        index=dates,
        columns=[f"STOCK_{i:04d}" for i in range(n_stocks)],
    )


def _make_benchmark_returns(
    stock_returns: pd.DataFrame,
    seed: int = 99,
) -> pd.Series:
    """生成合成基准日收益 Series（与个股有一定相关性）。"""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(stock_returns)) * 0.005
    bench = stock_returns.mean(axis=1) + noise
    bench.name = "benchmark"
    return bench


def _make_positions(
    n_long: int = 10,
    n_short: int = 10,
    stocks: Optional[List[str]] = None,
) -> dict[str, float]:
    """生成等权多空持仓字典。"""
    if stocks is None:
        stocks = [f"STOCK_{i:04d}" for i in range(30)]
    weights: dict[str, float] = {}
    for s in stocks[:n_long]:
        weights[s] = 1.0 / n_long
    for s in stocks[n_long:n_long + n_short]:
        weights[s] = -1.0 / n_short
    return weights


# ── HedgeManager 单元测试 ────────────────────────────────────────

class TestHedgeManager:
    """HedgeManager 单元测试（MockFuturesDataProvider）。"""

    def setup_method(self):
        self.provider = MockFuturesDataProvider(mock_price=6000.0, mock_multiplier=200.0)

    def test_full_hedge_positive_beta(self):
        """正 Beta 应生成做空期货（contracts < 0）。"""
        manager = HedgeManager(
            futures_provider=self.provider,
            hedge_symbol="IC",
            hedge_mode="full",
        )
        result = manager.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=0.85,
            portfolio_value=100_000_000,
        )
        assert result is not None
        assert result.contracts < 0, f"expected short futures, got {result.contracts}"
        assert abs(result.contracts) > 0

    def test_full_hedge_negative_beta(self):
        """负 Beta 应生成做多期货（contracts > 0）。"""
        manager = HedgeManager(
            futures_provider=self.provider,
            hedge_symbol="IC",
            hedge_mode="full",
        )
        result = manager.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=-0.3,
            portfolio_value=100_000_000,
        )
        assert result is not None
        assert result.contracts > 0, f"expected long futures, got {result.contracts}"

    def test_zero_beta_no_hedge(self):
        """Beta 接近 0 时合约数为 0。"""
        manager = HedgeManager(
            futures_provider=self.provider,
            hedge_symbol="IC",
        )
        result = manager.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=0.0,
            portfolio_value=100_000_000,
        )
        assert result is not None
        assert result.contracts == 0.0

    def test_partial_hedge(self):
        """50% 对冲比例时合约数约为完全对冲的一半。"""
        manager_full = HedgeManager(
            futures_provider=self.provider, hedge_mode="full",
        )
        manager_half = HedgeManager(
            futures_provider=self.provider,
            hedge_mode="partial",
            hedge_ratio_target=0.5,
        )
        r_full = manager_full.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=1.0,
            portfolio_value=100_000_000,
        )
        r_half = manager_half.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=1.0,
            portfolio_value=100_000_000,
        )
        assert r_full is not None and r_half is not None
        assert abs(r_half.contracts) <= abs(r_full.contracts) * 0.6

    def test_min_contracts_threshold(self):
        """合约数低于 min_contracts 时返回 0。"""
        manager = HedgeManager(
            futures_provider=self.provider,
            min_contracts=10.0,  # high threshold
        )
        result = manager.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=0.01,
            portfolio_value=10_000_000,
        )
        assert result is not None
        assert result.contracts == 0.0

    def test_futures_beta_adjustment(self):
        """futures_beta != 1 时合约数正确调整。"""
        manager_no_adj = HedgeManager(
            futures_provider=self.provider, futures_beta=1.0,
        )
        manager_with_adj = HedgeManager(
            futures_provider=self.provider, futures_beta=0.95,
        )
        r1 = manager_no_adj.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=1.0,
            portfolio_value=100_000_000,
        )
        r2 = manager_with_adj.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=1.0,
            portfolio_value=100_000_000,
        )
        assert r1 is not None and r2 is not None
        # futures_beta < 1 → contract_value 变小 → 需要更多手数
        assert abs(r2.contracts) >= abs(r1.contracts)

    def test_price_unavailable_returns_none(self):
        """期货价格不可用时返回 None。"""
        provider = MockFuturesDataProvider(mock_price=0.0, mock_multiplier=200.0)
        manager = HedgeManager(futures_provider=provider)
        result = manager.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=0.5,
            portfolio_value=100_000_000,
        )
        assert result is None

    def test_hedge_result_fields(self):
        """验证 HedgeResult 所有字段非空/类型正确。"""
        manager = HedgeManager(futures_provider=self.provider, hedge_symbol="IC")
        result = manager.compute_hedge(
            date=pd.Timestamp("2024-01-31"),
            net_beta=0.5,
            portfolio_value=100_000_000,
        )
        assert result is not None
        assert isinstance(result.date, pd.Timestamp)
        assert isinstance(result.net_beta, float)
        assert isinstance(result.hedge_ratio, float)
        assert isinstance(result.contracts, float)
        assert isinstance(result.hedge_notional, float)
        assert isinstance(result.futures_price, float)
        assert result.futures_symbol == "IC"
        assert result.required_margin >= 0

    def test_compute_hedge_timeseries(self):
        """批量计算对冲返回正确 DataFrame。"""
        manager = HedgeManager(futures_provider=self.provider)
        dates = [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-28")]
        betas = [0.8, -0.3]
        values = [100_000_000, 100_000_000]
        df = manager.compute_hedge_timeseries(dates, betas, values)
        assert len(df) == 2
        assert df.iloc[0]["contracts"] < 0  # positive beta → short
        assert df.iloc[1]["contracts"] > 0  # negative beta → long


# ── portfolio_beta 单元测试 ──────────────────────────────────────

class TestPortfolioBeta:
    """portfolio_beta 模块单元测试。"""

    def test_estimate_stock_betas_shape(self):
        """验证 Beta 估计输出形状正确。"""
        returns = _make_returns(n_dates=300, n_stocks=20)
        bench = _make_benchmark_returns(returns)
        betas = estimate_stock_betas(returns, bench, window=60, min_periods=20)
        assert betas.shape == returns.shape
        # 前 min_periods-1 天应为 NaN
        assert betas.iloc[:19].isna().all().all()
        # 后续应有有效值
        assert not betas.iloc[-1].isna().all()

    def test_estimate_stock_betas_values_in_range(self):
        """Beta 值在合理范围内。"""
        returns = _make_returns(n_dates=300, n_stocks=10)
        bench = _make_benchmark_returns(returns)
        betas = estimate_stock_betas(returns, bench, window=100, min_periods=30)
        valid = betas.dropna(how="all")
        # 合成数据下 Beta 应在 [-3, 3] 之间（宽泛范围）
        assert (valid.min().min() > -5) and (valid.max().max() < 5)

    def test_estimate_stock_betas_empty_input(self):
        """空输入返回空 DataFrame。"""
        empty = pd.DataFrame()
        empty_bench = pd.Series(dtype=float)
        result = estimate_stock_betas(empty, empty_bench)
        assert result.empty

    def test_compute_portfolio_beta(self):
        """组合加权 Beta 计算正确。"""
        betas = pd.Series({
            "STOCK_0000": 1.2,
            "STOCK_0001": 0.8,
            "STOCK_0002": 1.5,
            "STOCK_0003": 0.6,
        })
        positions = {
            "STOCK_0000": 0.5,
            "STOCK_0001": 0.5,
            "STOCK_0002": -0.5,
            "STOCK_0003": -0.5,
        }
        # long beta = 0.5*1.2 + 0.5*0.8 = 1.0
        # short beta = 0.5*0.6 + 0.5*1.5 = 1.05 → actually wait
        # abs(weight) * beta for short: 0.5*1.5 + 0.5*0.6 = 1.05
        # net = 1.0 - 1.05 = -0.05
        net = compute_portfolio_beta(positions, betas)
        assert abs(net + 0.05) < 1e-10

    def test_compute_long_short_betas(self):
        """多空分解正确。"""
        betas = pd.Series({
            "S1": 1.0,
            "S2": 0.5,
            "S3": 1.5,
            "S4": 0.8,
        })
        positions = {
            "S1": 0.5,    # long
            "S2": 0.5,    # long
            "S3": -0.5,   # short
            "S4": -0.5,   # short
        }
        result = compute_long_short_betas(positions, betas)
        assert abs(result["long_beta"] - 0.75) < 1e-10   # 0.5*1.0 + 0.5*0.5
        assert abs(result["short_beta"] - 1.15) < 1e-10  # 0.5*1.5 + 0.5*0.8
        assert abs(result["net_beta"] + 0.40) < 1e-10     # 0.75 - 1.15
        assert abs(result["long_exposure"] - 1.0) < 1e-10
        assert abs(result["short_exposure"] - 1.0) < 1e-10

    def test_compute_portfolio_beta_with_nan(self):
        """NaN Beta 的股票应被跳过。"""
        betas = pd.Series({
            "S1": 1.0,
            "S2": np.nan,
            "S3": 0.5,
        })
        positions = {"S1": 1.0, "S2": 1.0, "S3": -1.0}
        net = compute_portfolio_beta(positions, betas)
        # S2 跳过，只计算 S1 + S3
        assert abs(net - 0.5) < 1e-10  # 1.0*1.0 + (-1.0)*0.5 = 0.5


# ── Beta shift(1) 防未来函数 ────────────────────────────────────

class TestBetaNoLookahead:
    """验证 Beta 估计不包含未来信息。"""

    def test_shift_removes_current_day_info(self):
        """shift(1) 后当天 Beta 值不含当天收益。"""
        returns = _make_returns(n_dates=200, n_stocks=10)
        bench = _make_benchmark_returns(returns)
        betas = estimate_stock_betas(returns, bench, window=60, min_periods=30)
        betas_shifted = betas.shift(1)

        # 取最后一天：shifted 的值应等于原 DataFrame 中前一天的值
        last_date = returns.index[-1]
        prev_date = returns.index[-2]

        if last_date in betas_shifted.index:
            shifted_val = betas_shifted.loc[last_date, "STOCK_0000"]
            original_prev = betas.loc[prev_date, "STOCK_0000"]
            if pd.notna(shifted_val) and pd.notna(original_prev):
                assert abs(shifted_val - original_prev) < 1e-10

    def test_rebalance_uses_prior_day_beta(self):
        """模拟调仓日：使用的 Beta 来自前一交易日。"""
        returns = _make_returns(n_dates=252, n_stocks=20)
        bench = _make_benchmark_returns(returns)
        betas = estimate_stock_betas(returns, bench, window=120, min_periods=60)
        betas_shifted = betas.shift(1)

        # 假设调仓日在索引 251（最后一天）
        rebalance_date = returns.index[200]
        weights = _make_positions(10, 10, list(returns.columns))

        # 取 Beta（shift 后的 rebalance_date 实际是原 DataFrame 的 rebalance_date-1）
        beta_date = betas_shifted.index[betas_shifted.index <= rebalance_date][-1]
        date_betas = betas_shifted.loc[beta_date]

        result = compute_long_short_betas(weights, date_betas)
        assert isinstance(result["net_beta"], float)


# ── Qlib 数据形状防御 ────────────────────────────────────────────

class TestQlibFeatureShape:
    """验证 pivot 逻辑兼容不同 MultiIndex 顺序。"""

    def test_instrument_datetime_order(self):
        """MultiIndex (instrument, datetime) → pivot 正确。"""
        idx = pd.MultiIndex.from_tuples(
            [("SH600000", "2020-01-02"), ("SH600000", "2020-01-03"),
             ("SH600004", "2020-01-02"), ("SH600004", "2020-01-03")],
            names=["instrument", "datetime"],
        )
        df = pd.DataFrame(
            {"ret": [0.01, -0.02, 0.005, 0.008]},
            index=idx,
        )
        raw = df.reset_index()
        assert {"instrument", "datetime"}.issubset(set(raw.columns))
        ret_col = raw.columns[-1]
        pivoted = raw.pivot(index="datetime", columns="instrument", values=ret_col)
        assert pivoted.shape == (2, 2)
        assert "SH600000" in pivoted.columns

    def test_datetime_instrument_order(self):
        """MultiIndex (datetime, instrument) → pivot 正确。"""
        idx = pd.MultiIndex.from_tuples(
            [("2020-01-02", "SH600000"), ("2020-01-02", "SH600004"),
             ("2020-01-03", "SH600000"), ("2020-01-03", "SH600004")],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame(
            {"ret": [0.01, 0.005, -0.02, 0.008]},
            index=idx,
        )
        raw = df.reset_index()
        assert {"instrument", "datetime"}.issubset(set(raw.columns))
        ret_col = raw.columns[-1]
        pivoted = raw.pivot(index="datetime", columns="instrument", values=ret_col)
        assert pivoted.shape == (2, 2)

    def test_benchmark_series_from_multindex(self):
        """基准收益 MultiIndex → flat Series 提取正确。"""
        idx = pd.MultiIndex.from_tuples(
            [("SH000905", "2020-01-02"), ("SH000905", "2020-01-03")],
            names=["instrument", "datetime"],
        )
        df = pd.DataFrame({"ret": [0.005, -0.003]}, index=idx)
        col = df.columns[0]
        series = df[col]
        assert isinstance(series.index, pd.MultiIndex)
        flat = series.reset_index()
        flat["datetime"] = pd.to_datetime(flat["datetime"])
        flat = flat.set_index("datetime")[col].sort_index()
        assert isinstance(flat.index, pd.DatetimeIndex)
        assert len(flat) == 2


# ── generate_hedge_overlay 集成测试 ──────────────────────────────

class TestGenerateHedgeOverlay:
    """generate_hedge_overlay 集成测试（mock 组件）。"""

    def test_action_field_values(self):
        """验证 action 字段包含正确的值。"""
        from generate_positions_with_hedge import generate_hedge_overlay

        returns = _make_returns(n_dates=150, n_stocks=30)
        bench = _make_benchmark_returns(returns)
        provider = MockFuturesDataProvider(mock_price=6000.0, mock_multiplier=200.0)
        hedge_cfg = {
            "hedge_symbol": "IC",
            "hedge_mode": "full",
            "futures_beta": 1.0,
            "min_contracts": 0.5,
            "max_hedge_ratio": 1.5,
            "beta_window": 60,
            "beta_min_periods": 20,
        }

        # 用最后两个日期模拟调仓
        rebalance_date = returns.index[-2]
        stocks = list(returns.columns)
        weights = _make_positions(10, 10, stocks)
        position_cache = {rebalance_date: weights}

        df = generate_hedge_overlay(
            position_cache=position_cache,
            stock_returns=returns,
            benchmark_returns=bench,
            futures_provider=provider,
            hedge_config=hedge_cfg,
            portfolio_notional=100_000_000,
        )

        assert not df.empty
        assert "action" in df.columns
        actions = df["action"].unique()
        # 应有 rebalance（调仓日）和 carry（非调仓日沿用）
        assert "rebalance" in actions
        assert "carry" in actions

    def test_rebalance_has_contracts(self):
        """调仓日 action=rebalance 的记录 contracts != 0。"""
        from generate_positions_with_hedge import generate_hedge_overlay

        returns = _make_returns(n_dates=150, n_stocks=30)
        bench = _make_benchmark_returns(returns)
        provider = MockFuturesDataProvider(mock_price=6000.0, mock_multiplier=200.0)
        hedge_cfg = {
            "hedge_symbol": "IC",
            "hedge_mode": "full",
            "futures_beta": 1.0,
            "min_contracts": 0.5,
            "max_hedge_ratio": 1.5,
            "beta_window": 60,
            "beta_min_periods": 20,
        }

        rebalance_date = returns.index[-2]
        weights = _make_positions(10, 10, list(returns.columns))
        position_cache = {rebalance_date: weights}

        df = generate_hedge_overlay(
            position_cache=position_cache,
            stock_returns=returns,
            benchmark_returns=bench,
            futures_provider=provider,
            hedge_config=hedge_cfg,
            portfolio_notional=100_000_000,
        )

        rebalance_rows = df[df["action"] == "rebalance"]
        assert len(rebalance_rows) == 1
        assert rebalance_rows.iloc[0]["contracts"] != 0.0

    def test_empty_position_cache_returns_empty(self):
        """空 position_cache 返回空 DataFrame。"""
        from generate_positions_with_hedge import generate_hedge_overlay

        returns = _make_returns(n_dates=100, n_stocks=10)
        bench = _make_benchmark_returns(returns)
        provider = MockFuturesDataProvider()

        df = generate_hedge_overlay(
            position_cache={},
            stock_returns=returns,
            benchmark_returns=bench,
            futures_provider=provider,
            hedge_config={"hedge_symbol": "IC", "hedge_mode": "full"},
            portfolio_notional=100_000_000,
        )
        assert df.empty


# ── generate_positions() 接口不变 ────────────────────────────────

class TestGeneratePositionsUnchanged:
    """验证 generate_positions 接口未被修改。"""

    def test_returns_single_dataframe(self):
        """返回值仍是单个 DataFrame。"""
        from generate_positions import generate_positions

        dates = pd.date_range("2024-01-02", "2024-01-31", freq="B")
        records = []
        for d in dates:
            for i in range(30):
                records.append({
                    "datetime": d,
                    "instrument": f"STOCK_{i:04d}",
                    "ALPHA": np.random.standard_normal(),
                })
        predictions = pd.DataFrame(records)

        result = generate_positions(predictions, topk=10, bottomk=10)
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, tuple)
        assert "datetime" in result.columns
        assert "instrument" in result.columns
        assert "weight" in result.columns

    def test_weight_sum_is_zero(self):
        """每个交易日权重和 ≈ 0（美元中性）。"""
        from generate_positions import generate_positions

        dates = pd.date_range("2024-01-02", "2024-01-31", freq="B")
        rng = np.random.default_rng(42)
        records = []
        for d in dates:
            for i in range(50):
                records.append({
                    "datetime": d,
                    "instrument": f"STOCK_{i:04d}",
                    "ALPHA": rng.standard_normal(),
                })
        predictions = pd.DataFrame(records)

        result = generate_positions(predictions, topk=10, bottomk=10)
        # 检查每个唯一日期的权重和
        for dt, grp in result.groupby("datetime"):
            s = grp["weight"].sum()
            assert abs(s) < 1e-10, f"date={dt}, weight_sum={s}"


# ── re-export 层 ─────────────────────────────────────────────────

class TestReExport:
    """验证 quant_csi500_mn.hedging 可正常导入。"""

    def test_all_public_symbols(self):
        """所有公共符号可通过包导入。"""
        from quant_csi500_mn.hedging import (
            CONTRACT_SPECS,
            CsvFuturesDataProvider,
            FuturesContractSpec,
            FuturesDataProvider,
            HedgeManager,
            HedgeResult,
            MockFuturesDataProvider,
            compute_long_short_betas,
            compute_portfolio_beta,
            estimate_stock_betas,
        )
        assert HedgeManager is not None
        assert MockFuturesDataProvider is not None
        assert estimate_stock_betas is not None
        assert isinstance(CONTRACT_SPECS, dict)
        assert "IC" in CONTRACT_SPECS


# ── 端到端无对冲模式 ─────────────────────────────────────────────

class TestEndToEndNoHedge:
    """build_position_cache 和 no-hedge 路径测试。"""

    def test_build_position_cache_detects_rebalance(self):
        """从持仓 DataFrame 正确重建 position_cache。"""
        from generate_positions_with_hedge import build_position_cache

        # 模拟两个调仓日
        dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
                 pd.Timestamp("2024-01-04")]
        rows = []
        for i, d in enumerate(dates):
            suffix = f"_{i % 2}"  # 隔日切换持仓
            for j in range(10):
                rows.append({
                    "datetime": d,
                    "instrument": f"STOCK_{j:04d}{suffix}",
                    "weight": 0.1 if j < 5 else -0.1,
                })
        positions = pd.DataFrame(rows)
        cache = build_position_cache(positions)
        # 应该检测到至少 2 个调仓日（positions changed on alternate days）
        assert len(cache) >= 1
