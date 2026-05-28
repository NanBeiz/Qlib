#!/usr/bin/env python3
"""Layer 3a: Alpha 合成 —— 构造 DatasetH + 等权 baseline。

从 pickle 加载 handler，构造 DatasetH（含 train/valid/test 分段），
同时产生等权 baseline 信号用于与 LGBM 模型对比。

Usage:
    python 3_signal/04_alpha_score/build_dataset.py

输出:
    outputs/dataset/dataset.pkl              (DatasetH 实例)
    outputs/dataset/baseline_equal_weight.csv (等权 baseline ALPHA 信号)

    
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

import numpy as np
import pandas as pd

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_pickle, write_pickle, write_csv

logger = get_module_logger(__name__)


def compute_equal_weight_baseline(
    handler: DataHandlerLP,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """计算等权 baseline ALPHA 信号。

    取 test 段的 feature DataFrame（已经过 processor 链处理），
    等权平均所有因子列 → rank percentile 映射到 [0, 1]。

    Parameters
    ----------
    handler : DataHandlerLP
        已初始化的 handler。
    start_time : str
        test 段起始日期。
    end_time : str
        test 段结束日期。

    Returns
    -------
    pd.DataFrame
        columns: datetime, instrument, ALPHA_BASELINE
    """
    logger.info("fetching infer data for baseline...")
    df = handler.fetch(
        selector=slice(start_time, end_time),
        col_set="feature",
        data_key=DataHandlerLP.DK_I,
    )

    if df is None or df.empty:
        logger.warning("no infer data available")
        return pd.DataFrame(columns=["datetime", "instrument", "ALPHA_BASELINE"])

    # 提取 feature 列（MultiIndex: level 0 = "feature"）
    if isinstance(df.columns, pd.MultiIndex):
        feature_cols = df.columns.get_level_values(-1)[
            df.columns.get_level_values(0) == "feature"
        ]
        feature_df = df.xs("feature", axis=1, level=0, drop_level=False)
        if isinstance(feature_df.columns, pd.MultiIndex):
            feature_df = feature_df.xs("feature", axis=1, level=0)
    else:
        feature_cols = df.columns
        feature_df = df

    # 等权平均
    alpha_raw = feature_df.mean(axis=1, skipna=True)

    # 截面 rank percentile → [0, 1]
    alpha_df = alpha_raw.to_frame("ALPHA_RAW")
    alpha_df = alpha_df.reset_index()
    alpha_df["datetime"] = pd.to_datetime(alpha_df["datetime"])

    alpha_df["ALPHA_BASELINE"] = alpha_df.groupby("datetime", sort=False)[
        "ALPHA_RAW"
    ].transform(lambda s: s.rank(pct=True))

    logger.info(
        "baseline computed: %d rows, mean=%.4f, std=%.4f",
        len(alpha_df),
        alpha_df["ALPHA_BASELINE"].mean(),
        alpha_df["ALPHA_BASELINE"].std(),
    )

    return alpha_df[["datetime", "instrument", "ALPHA_BASELINE"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3a: Alpha 合成")
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置"
    )
    parser.add_argument(
        "--handler-config", default="configs/handlers/csi500_alpha14.yaml",
        help="Handler YAML 配置路径",
    )
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    # 初始化 qlib
    import qlib
    qlib_kwargs = {}
    if common_cfg.get("cache_enabled"):
        qlib_kwargs["expression_cache"] = "DiskExpressionCache"
        qlib_kwargs["dataset_cache"] = "DiskDatasetCache"
    qlib.init(provider_uri=common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
              region=common_cfg.get("region", "cn"), **qlib_kwargs)

    # 从 YAML 重新实例化 handler（pickle 不会保留 _infer/_data 属性）
    import yaml
    from qlib.utils import init_instance_by_config

    handler_cfg_path = root / args.handler_config
    with open(handler_cfg_path, "r", encoding="utf-8") as f:
        handler_cfg = yaml.safe_load(f) or {}
    handler_cfg.setdefault("kwargs", {})["start_time"] = common_cfg["train_start"]
    handler_cfg["kwargs"]["end_time"] = common_cfg["test_end"]

    logger.info("instantiating handler from %s...", args.handler_config)
    handler = init_instance_by_config(handler_cfg)

    # 构造 DatasetH
    train_start = common_cfg["train_start"]
    train_end = common_cfg["train_end"]
    valid_start = common_cfg["valid_start"]
    valid_end = common_cfg["valid_end"]
    test_start = common_cfg["test_start"]
    test_end = common_cfg["test_end"]

    segments = {
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, test_end),
    }
    logger.info("segments: train=%s~%s, valid=%s~%s, test=%s~%s",
                 train_start, train_end, valid_start, valid_end, test_start, test_end)

    logger.info("creating DatasetH...")
    dataset = DatasetH(
        handler=handler,
        segments=segments,
    )

    # 触发训练段数据准备（预加载 feature + label）
    logger.info("preparing train segment...")
    dataset.prepare(["train"], col_set=["feature", "label"])
    logger.info("dataset prepared")

    # 输出 dataset pickle
    out_dir = root / "outputs" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "dataset.pkl"
    write_pickle(dataset, dataset_path)
    logger.info("dataset exported: %s", dataset_path)

    # 等权 baseline
    baseline_df = compute_equal_weight_baseline(handler, test_start, test_end)
    if not baseline_df.empty:
        baseline_path = out_dir / "baseline_equal_weight.csv"
        write_csv(baseline_df, baseline_path)
        logger.info("baseline exported: %s", baseline_path)

    logger.info("Layer 3a complete")


if __name__ == "__main__":
    main()
