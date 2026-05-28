"""测试 CSI500AlphaHandler 数据加载。"""

import pytest

from qlib.data.dataset.handler import DataHandlerLP

from quant_csi500_mn.handlers.csi500_handler import CSI500AlphaHandler
from quant_csi500_mn.utils.qlib_init import init_qlib, reset_qlib


@pytest.fixture(autouse=True)
def _qlib_env():
    """每个测试前后初始化/重置 Qlib。"""
    init_qlib()
    yield
    reset_qlib()


from pathlib import Path

_FACTOR_YAML = str(Path(__file__).resolve().parents[1] / "configs" / "factors" / "alpha14.yaml")


class TestCSI500AlphaHandler:
    """CSI500AlphaHandler 集成测试。"""

    def test_feature_count(self):
        """验证 feature 列数 == 14。"""
        handler = CSI500AlphaHandler(
            instruments="csi500",
            start_time="2024-01-01",
            end_time="2024-06-30",
            factor_yaml=_FACTOR_YAML,
        )

        df = handler.fetch(
            selector=slice("2024-01-01", "2024-06-30"),
            col_set="feature",
            data_key=DataHandlerLP.DK_I,
        )

        feature_cols = df.columns.get_level_values(-1)
        assert len(feature_cols) == 14, f"Expected 14 features, got {len(feature_cols)}"
        assert len(handler.factor_names) == 14

    def test_label_exists(self):
        """验证 label 列 LABEL0 存在。"""
        handler = CSI500AlphaHandler(
            instruments="csi500",
            start_time="2024-01-01",
            end_time="2024-06-30",
            factor_yaml=_FACTOR_YAML,
        )

        df = handler.fetch(
            selector=slice("2024-01-01", "2024-06-30"),
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        assert "label" in df.columns.get_level_values(0)
        assert handler.label_name in df.columns.get_level_values(-1)

    def test_minimum_trading_days(self):
        """验证至少 80 个交易日。"""
        handler = CSI500AlphaHandler(
            instruments="csi500",
            start_time="2024-01-01",
            end_time="2024-06-30",
            factor_yaml=_FACTOR_YAML,
        )

        df = handler.fetch(
            selector=slice("2024-01-01", "2024-06-30"),
            col_set="feature",
            data_key=DataHandlerLP.DK_I,
        )

        trading_days = df.index.get_level_values("datetime").unique()
        assert len(trading_days) >= 80, f"Expected >=80 trading days, got {len(trading_days)}"

    def test_learn_data_differs_from_infer(self):
        """验证 learn 数据（经 learn_processors 处理）与 infer 数据不同。"""
        handler = CSI500AlphaHandler(
            instruments="csi500",
            start_time="2024-01-01",
            end_time="2024-06-30",
            factor_yaml=_FACTOR_YAML,
            learn_processors=[
                {
                    "class": "DropnaLabel",
                    "module_path": "qlib.data.dataset.processor",
                    "kwargs": {"fields_group": "label"},
                },
                {
                    "class": "CSRankNorm",
                    "module_path": "qlib.data.dataset.processor",
                    "kwargs": {"fields_group": "label"},
                },
            ],
        )

        infer_df = handler.fetch(
            selector=slice("2024-01-01", "2024-06-30"),
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_I,
        )
        learn_df = handler.fetch(
            selector=slice("2024-01-01", "2024-06-30"),
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        infer_label = infer_df[("label", handler.label_name)]
        learn_label = learn_df[("label", handler.label_name)]

        # learn label 经 DropnaLabel 会丢失 NaN 行，行数 ≤ infer
        assert len(learn_label) <= len(infer_label), (
            f"Expected learn rows ({len(learn_label)}) <= infer rows ({len(infer_label)})"
        )

        # learn label 经 CSRankNorm 映射到 ~N(0,1)，值范围在 [-3, 3]
        learn_clean = learn_label.dropna()
        assert learn_clean.abs().max() < 4.0, (
            f"Expected learn label in [-4, 4], got max={learn_clean.abs().max():.2f}"
        )

        # infer 和 learn 的 label 值不同（因为 learn 经过 transform）
        common_idx = infer_label.dropna().index.intersection(learn_clean.index)
        assert not infer_label.loc[common_idx].equals(learn_label.loc[common_idx]), (
            "Expected learn label to differ from infer label after CSRankNorm"
        )
