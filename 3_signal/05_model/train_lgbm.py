#!/usr/bin/env python3
"""Layer 3b: 模型训练 —— 训练 LightGBM 模型。

使用 Qlib LGBModel 包装器，从 DatasetH 读取 train/valid 段，
自动处理早停和日志，输出模型和特征重要性。

Usage:
    python 3_signal/05_model/train_lgbm.py

输出:
    outputs/model/lgbm_model.pkl         (LGBModel 实例)
    outputs/model/feature_importance.csv (特征重要性)
    outputs/model/training_log.txt       (训练日志)
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

from qlib.contrib.model.gbdt import LGBModel
from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_pickle, write_pickle, write_csv

logger = get_module_logger(__name__)


def get_default_lgbm_params() -> dict:
    """返回默认 LGBM 超参。"""
    return {
        "loss": "mse",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "early_stopping_rounds": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "min_child_samples": 50,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3b: 训练 LGBM")
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置"
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="DatasetH pickle 路径，默认 outputs/dataset/dataset.pkl",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="learning_rate（覆盖默认值）"
    )
    parser.add_argument(
        "--num-leaves", type=int, default=None, help="num_leaves（覆盖默认值）"
    )
    args = parser.parse_args()

    root = get_project_root()
    common_cfg = load_config(args.common_config)

    import qlib
    qlib.init(provider_uri=common_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
              region=common_cfg.get("region", "cn"))

    # 从 YAML 重新实例化 handler + dataset（DataHandlerLP 不支持 pickle）
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
    dataset.prepare(["train", "valid"], col_set=["feature", "label"])
    logger.info("dataset prepared: train=%s~%s, valid=%s~%s",
                 common_cfg["train_start"], common_cfg["train_end"],
                 common_cfg["valid_start"], common_cfg["valid_end"])

    # 构造模型参数
    params = get_default_lgbm_params()
    if args.lr is not None:
        params["learning_rate"] = args.lr
    if args.num_leaves is not None:
        params["num_leaves"] = args.num_leaves

    logger.info("LGBM params: lr=%.4f, leaves=%d, estimators=%d",
                 params["learning_rate"], params["num_leaves"], params["n_estimators"])

    # 实例化 LGBModel
    model = LGBModel(**params)

    # 训练（Qlib 自动处理 train/valid 切分和早停）
    logger.info("training LGBModel...")

    # 捕获训练日志
    log_buffer = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = log_buffer
        model.fit(dataset)
    finally:
        sys.stdout = old_stdout

    training_log = log_buffer.getvalue()
    logger.info("training complete")

    # 输出
    out_dir = root / "outputs" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 模型 pickle
    model_path = out_dir / "lgbm_model.pkl"
    write_pickle(model, model_path)
    logger.info("model exported: %s", model_path)

    # 特征重要性
    try:
        importance = model.feature_importance
        if importance is not None:
            imp_df = pd.DataFrame(
                {"feature": range(len(importance)), "importance": importance}
            ).sort_values("importance", ascending=False)
            imp_path = out_dir / "feature_importance.csv"
            write_csv(imp_df, imp_path)
            logger.info("feature importance exported: %s", imp_path)
    except Exception as e:
        logger.warning("could not extract feature importance: %s", e)

    # 训练日志
    if training_log.strip():
        log_path = out_dir / "training_log.txt"
        log_path.write_text(training_log, encoding="utf-8")
        logger.info("training log exported: %s", log_path)

    logger.info("Layer 3b complete")


if __name__ == "__main__":
    main()
