#!/usr/bin/env python3
"""Layer 3c: 预测 —— 生成 test 段预测信号。

加载训练好的 LGBModel，在 test 段预测 → 截面 rank percentile → ALPHA ∈ [0, 1]。

Usage:
    python 3_signal/05_model/predict_lgbm.py

输出:
    outputs/predictions/predictions_lgbm.csv     (LGBM 预测)
    outputs/predictions/predictions_baseline.csv (等权 baseline)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_pickle, read_csv, write_csv

logger = get_module_logger(__name__)


def compute_alpha_from_predictions(
    pred: np.ndarray,
    df_with_index: pd.DataFrame,
) -> pd.DataFrame:
    """将 raw predictions 转为截面 rank percentile ALPHA。

    Parameters
    ----------
    pred : np.ndarray
        模型预测值（1D，与 df_with_index 行数相同）。
    df_with_index : pd.DataFrame
        含 datetime 列或 MultiIndex 的数据。

    Returns
    -------
    pd.DataFrame
        columns: datetime, instrument, ALPHA
    """
    if df_with_index.index.nlevels == 2:
        result = df_with_index.reset_index()[["datetime", "instrument"]].copy()
    elif "datetime" in df_with_index.columns and "instrument" in df_with_index.columns:
        result = df_with_index[["datetime", "instrument"]].copy()
    else:
        result = pd.DataFrame(index=df_with_index.index).reset_index()
        result.columns = ["datetime", "instrument"][: result.shape[1]]

    result["datetime"] = pd.to_datetime(result["datetime"])
    result["instrument"] = result["instrument"].astype(str)
    result["score_raw"] = pred

    # 截面 rank percentile
    result["ALPHA"] = result.groupby("datetime", sort=False)["score_raw"].transform(
        lambda s: s.rank(pct=True)
    )

    return result[["datetime", "instrument", "ALPHA"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3c: 预测")
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="LGBModel pickle 路径，默认 outputs/model/lgbm_model.pkl",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="DatasetH pickle 路径，默认 outputs/dataset/dataset.pkl",
    )
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    import qlib
    qlib.init(provider_uri=common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
              region=common_cfg.get("region", "cn"))

    # 加载模型
    model_path = args.model_path or str(root / "outputs" / "model" / "lgbm_model.pkl")
    logger.info("loading model from %s...", model_path)
    model = read_pickle(model_path)

    # 从 YAML 重新实例化 handler + dataset
    import yaml as _yaml
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH

    handler_cfg_path = root / "configs" / "handlers" / "csi500_alpha14.yaml"
    with open(handler_cfg_path, "r", encoding="utf-8") as f:
        handler_cfg = _yaml.safe_load(f) or {}
    handler_cfg.setdefault("kwargs", {})["start_time"] = common_cfg["train_start"]
    handler_cfg["kwargs"]["end_time"] = common_cfg["test_end"]
    logger.info("instantiating handler from YAML...")
    handler = init_instance_by_config(handler_cfg)

    segments = {
        "train": (common_cfg["train_start"], common_cfg["train_end"]),
        "valid": (common_cfg["valid_start"], common_cfg["valid_end"]),
        "test": (common_cfg["test_start"], common_cfg["test_end"]),
    }
    dataset = DatasetH(handler=handler, segments=segments)
    logger.info("dataset created")

    # 预测 test 段
    logger.info("predicting on test segment...")
    raw_pred = model.predict(dataset, segment="test")

    # raw_pred 是 pd.Series，index 为 (datetime, instrument) MultiIndex
    if isinstance(raw_pred, np.ndarray):
        # numpy array: 从 test_data 获取 index
        test_data = dataset.prepare("test", col_set="feature", data_key=dataset.handler.DK_I)
        predictions = pd.DataFrame({
            "score_raw": raw_pred,
        }, index=test_data.index)
        predictions = predictions.reset_index()
    else:
        # pd.Series with MultiIndex
        predictions = raw_pred.reset_index()
        predictions.columns = ["datetime", "instrument", "score_raw"] if len(predictions.columns) == 3 else ["datetime", "instrument", "score_raw"]

    # 标准化类型
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])
    predictions["instrument"] = predictions["instrument"].astype(str)

    # 截面 rank percentile → ALPHA
    predictions["ALPHA"] = predictions.groupby("datetime", sort=False)["score_raw"].transform(
        lambda s: s.rank(pct=True)
    )
    predictions = predictions[["datetime", "instrument", "ALPHA"]]

    logger.info(
        "predictions: %d rows, ALPHA mean=%.4f, std=%.4f",
        len(predictions),
        predictions["ALPHA"].mean(),
        predictions["ALPHA"].std(),
    )

    # 输出
    out_dir = root / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions_lgbm.csv"
    write_csv(predictions, pred_path)
    logger.info("predictions exported: %s", pred_path)

    # 复制 baseline（如果存在）
    baseline_src = root / "outputs" / "dataset" / "baseline_equal_weight.csv"
    if baseline_src.exists():
        baseline_df = read_csv(baseline_src)
        baseline_df = baseline_df.rename(columns={"ALPHA_BASELINE": "ALPHA"})
        baseline_dst = out_dir / "predictions_baseline.csv"
        write_csv(baseline_df, baseline_dst)
        logger.info("baseline predictions exported: %s", baseline_dst)
    else:
        logger.warning("baseline not found at %s, skipping", baseline_src)

    logger.info("Layer 3c complete")


if __name__ == "__main__":
    main()
