#!/usr/bin/env python3
"""Layer 2a: 因子构建 —— 产出原始未处理的因子值。

读取 alpha14.yaml 的 expression 列表，用 D.features 一次性计算所有因子，
与 universe 内连接保证 PIT 正确。

Usage:
    python 2_factor/02_factor_construction/build_factors.py --config configs/factors/alpha14.yaml

输出:
    outputs/factors/all_factors_raw.csv    (宽表)
    outputs/factors/factor_coverage.csv    (非空率、首个非空日期)

alpha14.yaml 因子表达式
        ↓
D.features 批量计算因子
        ↓
得到 instrument × datetime 的因子表
        ↓
和 PIT universe 做 inner join
        ↓
只保留“当天在股票池里”的样本

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

import qlib
from qlib.data import D
from qlib.log import get_module_logger

from quant_csi500_mn.utils.config import get_project_root, load_config
from quant_csi500_mn.utils.io import read_csv, write_csv

logger = get_module_logger(__name__)


def load_factor_definitions(factor_yaml_path: str) -> list[dict]:
    """加载因子定义列表。"""
    root = get_project_root()
    if not Path(factor_yaml_path).is_absolute():
        if factor_yaml_path.startswith("configs/"):
            factor_yaml_path = factor_yaml_path[len("configs/"):]
        path = root / "configs" / factor_yaml_path
    else:
        path = Path(factor_yaml_path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("factors", [])


def compute_raw_factors(
    instruments: list[str],
    expressions: list[str],
    names: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """用 D.features 一次性计算所有因子。

    Parameters
    ----------
    instruments : list[str]
        股票列表。
    expressions : list[str]
        Qlib 表达式列表。
    names : list[str]
        因子名列表，与 expressions 一一对应。
    start_time : str
        起始日期。
    end_time : str
        结束日期。

    Returns
    -------
    pd.DataFrame
        columns: datetime, instrument, name1, name2, ...
    """
    logger.info("computing %d factors for %d instruments...", len(expressions), len(instruments))

    df = D.features(
        instruments=instruments,
        fields=expressions,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )

    if df.index.nlevels == 2:
        df = df.reset_index()

    # 用 QlibDataLoader 风格重命名列：表达式 → 因子名
    rename_map = {}
    for expr, name in zip(expressions, names):
        for col in df.columns:
            if col == expr:
                rename_map[col] = name
                break
    df = df.rename(columns=rename_map)

    # 标准化列名
    col_rename = {}
    for c in df.columns:
        if isinstance(c, str) and c.startswith("$"):
            col_rename[c] = c.lstrip("$")
    df = df.rename(columns=col_rename)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    if "instrument" in df.columns:
        df["instrument"] = df["instrument"].astype(str)

    return df


def compute_coverage(df: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    """计算每个因子的非空率和首个非空日期。

    Parameters
    ----------
    df : pd.DataFrame
        因子宽表。
    factor_names : list[str]
        因子列名列表。

    Returns
    -------
    pd.DataFrame
        columns: factor, coverage, first_valid_date, n_valid, n_total
    """
    rows = []
    for name in factor_names:
        if name not in df.columns:
            rows.append(
                {
                    "factor": name,
                    "coverage": 0.0,
                    "first_valid_date": None,
                    "n_valid": 0,
                    "n_total": len(df),
                }
            )
            continue

        col = df[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        valid = col.notna()
        coverage = float(valid.mean() if hasattr(valid, "mean") else valid.sum() / len(valid))
        first_date = df.loc[valid.values if hasattr(valid, "values") else valid, "datetime"].min() if valid.any() else None

        rows.append(
            {
                "factor": name,
                "coverage": round(coverage, 6),
                "first_valid_date": str(first_date.date()) if first_date is not None else "N/A",
                "n_valid": int(valid.sum()),
                "n_total": len(df),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 2a: 因子构建")
    parser.add_argument(
        "--config", default="configs/factors/alpha14.yaml", help="因子配置文件"
    )
    parser.add_argument(
        "--common-config", default="configs/_common.yaml", help="全局配置文件"
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

    # 加载因子定义
    factor_defs = load_factor_definitions(args.config)
    expressions = [f["expression"] for f in factor_defs]
    names = [f["name"] for f in factor_defs]
    logger.info("loaded %d factors: %s", len(names), ", ".join(names))

    # 加载 universe
    universe_path = root / "outputs" / "data" / "universe_csi500.csv"
    if not universe_path.exists():
        logger.warning("universe not found at %s, loading all csi500 instruments...", universe_path)
        instruments = D.instruments(market=common_cfg.get("market", "csi500"))
        instruments = D.list_instruments(instruments=instruments, as_list=True)
    else:
        universe = read_csv(universe_path)
        instruments = sorted(universe["instrument"].unique())

    # 计算因子
    train_start = common_cfg["train_start"]
    test_end = common_cfg["test_end"]

    raw_factors = compute_raw_factors(instruments, expressions, names, train_start, test_end)

    # 与 universe 内连接
    if universe_path.exists():
        universe = read_csv(universe_path)
        raw_factors = universe.merge(raw_factors, on=["datetime", "instrument"], how="inner")
        logger.info("after inner join with universe: %d rows", len(raw_factors))

    # 输出
    out_dir = root / "outputs" / "factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    factors_path = out_dir / "all_factors_raw.csv"
    write_csv(raw_factors, factors_path)
    logger.info("raw factors exported: %s (%d rows × %d cols)", factors_path, len(raw_factors), len(raw_factors.columns))

    # 覆盖率报告
    coverage = compute_coverage(raw_factors, names)
    coverage_path = out_dir / "factor_coverage.csv"
    write_csv(coverage, coverage_path)
    logger.info("coverage exported: %s", coverage_path)

    print("\n因子覆盖率:")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
