#!/usr/bin/env python3
"""Layer 2b: 因子工程 —— 构造并序列化 DataHandlerLP 实例。

从 YAML 配置通过 init_instance_by_config 实例化 CSI500AlphaHandler，
调用 fetch() 触发数据加载和 processor 链处理，pickle 持久化。

Usage:
    python 2_factor/03_factor_engineering/build_handler.py --config configs/handlers/csi500_alpha14.yaml

输出:
    outputs/handler/csi500_alpha14_handler.pkl       (DataHandlerLP 实例)
    outputs/handler/features_processed_sample.csv    (前 30 个交易日的特征，调试用)
    
    
YAML 配置
   ↓
init_instance_by_config
   ↓
CSI500AlphaHandler 实例
   ↓
加载 raw feature / label
   ↓
processor 链处理
   ↓
得到 _infer / _learn
   ↓
pickle 持久化


init_instance_by_config(...)
  ↓
调用 CSI500AlphaHandler.__init__
  ↓
DataHandlerLP.__init__
  ↓
setup_data()
  ↓
data_loader.load(...)
  ↓
processor.fit / processor.__call__
  ↓
生成 _data / _infer / _learn
  ↓
fetch() 取出处理后的数据
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

import qlib
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.utils import init_instance_by_config

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import write_pickle, write_csv

logger = get_module_logger(__name__)


def load_handler_config(config_path: str) -> dict:
    """加载 handler YAML 配置。"""
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_sample(
    handler: DataHandlerLP,
    segment: str = "test",
    max_days: int = 30,
) -> pd.DataFrame:
    """提取前 max_days 个交易日的处理后特征样本。

    Parameters
    ----------
    handler : DataHandlerLP
        已初始化的 handler。
    segment : str
        取哪个数据段（train/valid/test）。
    max_days : int
        最多取多少个交易日。

    Returns
    -------
    pd.DataFrame
    """
    # 尝试用 DK_I (infer data) 获取处理后数据
    try:
        df = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    except Exception:
        try:
            df = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_L)
        except Exception:
            logger.warning("cannot fetch data from handler")
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.nlevels == 2:
        dates = df.index.get_level_values("datetime").unique()
        sample_dates = dates[:max_days]
        df = df.loc[df.index.get_level_values("datetime").isin(sample_dates)]
    elif "datetime" in df.columns:
        dates = df["datetime"].unique()
        sample_dates = dates[:max_days]
        df = df[df["datetime"].isin(sample_dates)]

    return df.reset_index() if df.index.nlevels == 2 else df


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 2b: 因子工程")
    parser.add_argument(
        "--config", default="configs/handlers/csi500_alpha14.yaml", help="Handler YAML 配置"
    )
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置"
    )
    args = parser.parse_args()

    root = get_project_root()

    # 加载全局配置
    common_cfg = load_config(args.common_config)
    provider_uri = common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data")

    qlib_kwargs = {}
    if common_cfg.get("cache_enabled"):
        qlib_kwargs["expression_cache"] = "DiskExpressionCache"
        qlib_kwargs["dataset_cache"] = "DiskDatasetCache"

    qlib.init(provider_uri=provider_uri, region=common_cfg.get("region", "cn"), **qlib_kwargs)

    # 加载 handler 配置
    handler_cfg = load_handler_config(args.config)
    logger.info("loaded handler config: class=%s", handler_cfg.get("class"))

    # 覆盖时间区间（使用 _common.yaml 的日期）
    kwargs = handler_cfg.setdefault("kwargs", {})
    kwargs.setdefault("start_time", common_cfg["train_start"])
    kwargs.setdefault("end_time", common_cfg["test_end"])

    # 通过 init_instance_by_config 实例化 handler
    logger.info("instantiating handler via init_instance_by_config...")
    handler: DataHandlerLP = init_instance_by_config(handler_cfg)
    logger.info("handler instantiated: %s", type(handler).__name__)

    # 触发数据加载（fetch 会触发所有 processor 链）
    logger.info("triggering data fetch...")
    _ = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    logger.info("data fetch complete")

    # 输出 handler pickle
    out_dir = root / "outputs" / "handler"
    out_dir.mkdir(parents=True, exist_ok=True)

    handler_path = out_dir / "csi500_alpha14_handler.pkl"
    write_pickle(handler, handler_path)
    logger.info("handler exported: %s", handler_path)

    # 输出样本 CSV（调试用）
    sample_df = extract_sample(handler)
    if not sample_df.empty:
        sample_path = out_dir / "features_processed_sample.csv"
        write_csv(sample_df, sample_path)
        logger.info("sample exported: %s (%d rows)", sample_path, len(sample_df))

    logger.info("Layer 2b complete")


if __name__ == "__main__":
    main()
