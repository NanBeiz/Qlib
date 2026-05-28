"""测试 IndustryMcapNeutralize 处理器。"""

import numpy as np
import pandas as pd
import pytest

from quant_csi500_mn.processors.neutralize import IndustryMcapNeutralize


def _make_test_df(
    dates: list[pd.Timestamp],
    instruments: list[str],
    industry_map: dict[str, str],
    mcap_map: dict[str, float],
    factor_bias: dict[str, float],
) -> pd.DataFrame:
    """构造含 MultiIndex 列的测试 DataFrame。

    因子值 = industry_bias + 随机噪声。
    """
    rows = []
    for d in dates:
        for sym in instruments:
            ind = industry_map.get(sym, "OTHER")
            noise = np.random.randn() * 0.01
            rows.append(
                {
                    ("datetime", ""): d,
                    ("instrument", ""): sym,
                    ("feature", "factor1"): factor_bias.get(ind, 0.0) + noise,
                    "LOG_MCAP": np.log(mcap_map.get(sym, 1e10)),
                }
            )

    df = pd.DataFrame(rows)

    # 添加行业虚拟变量
    for ind_name in set(industry_map.values()):
        col_name = f"IND_{ind_name}"
        df[col_name] = 0.0

    for d in dates:
        for sym in instruments:
            ind = industry_map.get(sym, "OTHER")
            mask = (df[("datetime", "")] == d) & (df[("instrument", "")] == sym)
            df.loc[mask, f"IND_{ind}"] = 1.0

    df = df.set_index([("datetime", ""), ("instrument", "")])
    df.index.names = ["datetime", "instrument"]
    df.columns = pd.MultiIndex.from_tuples(
        [(c[0], c[1]) if isinstance(c, tuple) else ("_", c) for c in df.columns]
    )
    return df


class TestIndustryMcapNeutralize:
    """IndustryMcapNeutralize 单元测试。"""

    def test_removes_industry_bias(self):
        """构造有行业偏差的因子，验证中性化后行业回归系数 < 1e-6。"""
        dates = pd.date_range("2024-01-03", "2024-01-05")
        instruments = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E", "STOCK_F"]
        industry_map = {
            "STOCK_A": "TECH", "STOCK_B": "TECH", "STOCK_C": "TECH",
            "STOCK_D": "FIN",  "STOCK_E": "FIN",  "STOCK_F": "FIN",
        }
        mcap_map = {s: 1e10 for s in instruments}
        # 故意设行业偏差：TECH=2, FIN=5
        factor_bias = {"TECH": 2.0, "FIN": 5.0}

        df = _make_test_df(dates, instruments, industry_map, mcap_map, factor_bias)

        processor = IndustryMcapNeutralize(fields_group="feature")
        result = processor(df)

        # 中性化后按日截面回归，行业系数应接近 0
        for d in dates:
            day_df = result.loc[d]
            y = day_df[("feature", "factor1")].values.astype(np.float64)
            ind_tech = day_df[("_", "IND_TECH")].values.astype(np.float64)
            ind_fin = day_df[("_", "IND_FIN")].values.astype(np.float64)
            X = np.column_stack([np.ones(6), ind_tech, ind_fin])
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            # 截距后的行业系数应 < 1e-6
            assert abs(beta[1]) < 1e-6, f"IND_TECH beta={beta[1]:.2e} not zero at {d.date()}"
            assert abs(beta[2]) < 1e-6, f"IND_FIN beta={beta[2]:.2e} not zero at {d.date()}"

    def test_preserves_non_feature_columns(self):
        """验证非 feature 列（如 LOG_MCAP、IND_XXX）保持原值不变。"""
        dates = pd.date_range("2024-01-03", "2024-01-04")
        instruments = ["STOCK_A", "STOCK_B"]
        industry_map = {"STOCK_A": "TECH", "STOCK_B": "FIN"}
        mcap_map = {"STOCK_A": 1e10, "STOCK_B": 2e10}

        df = _make_test_df(dates, instruments, industry_map, mcap_map, {"TECH": 1.0, "FIN": 2.0})
        original_mcap = df[("_", "LOG_MCAP")].copy()

        processor = IndustryMcapNeutralize()
        result = processor(df)

        pd.testing.assert_series_equal(
            result[("_", "LOG_MCAP")], original_mcap, check_names=False
        )

    def test_handles_empty_feature_group(self):
        """当 fields_group 不存在时，不抛异常，原样返回。"""
        df = pd.DataFrame(
            {"A": [1.0, 2.0], "IND_TECH": [1.0, 0.0], "LOG_MCAP": [20.0, 21.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-03"), "S1"), (pd.Timestamp("2024-01-03"), "S2")],
                names=["datetime", "instrument"],
            ),
        )
        processor = IndustryMcapNeutralize(fields_group="nonexistent")
        result = processor(df)
        pd.testing.assert_frame_equal(result, df)
