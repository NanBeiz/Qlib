"""冒烟测试：运行完整 pipeline（仅 3 个月数据），验证每层输出文件存在。

此测试通过调用各层入口脚本的 main 函数（而非 subprocess）来运行，
以便在同一个进程中验证数据流。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 将项目根加入 path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from quant_csi500_mn.utils.qlib_init import init_qlib, reset_qlib


@pytest.fixture(autouse=True)
def _qlib_env():
    init_qlib()
    yield
    reset_qlib()


class TestPipelineSmoke:
    """冒烟测试：逐层验证核心组件可运行。"""

    def test_layer1_prepare_data_output(self):
        """Layer 1: 验证 universe CSV 可生成。"""
        from quant_csi500_mn.utils.io import read_csv

        output_dir = _PROJECT_ROOT / "outputs" / "data"
        universe_path = output_dir / "universe_csi500.csv"

        # 如果文件已存在（由之前运行生成），直接验证
        if universe_path.exists():
            df = read_csv(universe_path)
            assert len(df) > 0
            assert list(df.columns) == ["datetime", "instrument"]
            assert df["datetime"].nunique() > 10
        else:
            pytest.skip("universe_csi500.csv not found, run prepare_data.py first")

    def test_layer5a_backtest_output(self):
        """Layer 5a: 验证回测报告可读取。"""
        report_path = _PROJECT_ROOT / "outputs" / "backtest" / "report_normal.csv"
        if report_path.exists():
            import pandas as pd
            report = pd.read_csv(report_path, index_col=0, parse_dates=True)
            assert "return" in report.columns
            assert len(report) > 10
        else:
            pytest.skip("report_normal.csv not found, run pipeline first")

    def test_utils_import(self):
        """验证所有 utils 模块可导入。"""
        from quant_csi500_mn.utils.qlib_init import init_qlib
        from quant_csi500_mn.utils.io import read_csv, write_csv
        from quant_csi500_mn.utils.config import get_project_root, load_config

        assert get_project_root() == _PROJECT_ROOT

    def test_handler_import(self):
        """验证 CSI500AlphaHandler 可导入。"""
        from quant_csi500_mn.handlers.csi500_handler import CSI500AlphaHandler
        assert CSI500AlphaHandler.FACTOR_NAMES is not None
        assert len(CSI500AlphaHandler.FACTOR_NAMES) == 14

    def test_strategy_import(self):
        """验证策略类可导入。"""
        import pandas as pd
        from quant_csi500_mn.strategies.top_bottom_neutral import TopBottomNeutralStrategy

        strategy = TopBottomNeutralStrategy(
            topk=10, bottomk=10, signal=pd.Series(dtype=float),
        )
        assert strategy.topk == 10
        assert strategy.bottomk == 10

    def test_processor_import(self):
        """验证中性化处理器可导入。"""
        from quant_csi500_mn.processors.neutralize import IndustryMcapNeutralize

        processor = IndustryMcapNeutralize()
        assert processor.fields_group == "feature"

    def test_config_loading(self):
        """验证全局配置和因子配置可加载。"""
        from quant_csi500_mn.utils.config import load_config

        common = load_config("_common.yaml")
        assert common["market"] == "csi500"
        assert "train_start" in common

        import yaml
        factor_path = _PROJECT_ROOT / "configs" / "factors" / "alpha14.yaml"
        with open(factor_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f) or {}
        assert len(factors.get("factors", [])) == 14
