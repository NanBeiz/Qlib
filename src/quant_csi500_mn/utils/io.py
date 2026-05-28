"""CSV/pickle 读写规约，统一层间数据交换格式。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd


def read_csv(path: Union[str, Path], sort: bool = True) -> pd.DataFrame:
    """读取 CSV，标准化 datetime / instrument 列类型并排序。

    Parameters
    ----------
    path : str or Path
        CSV 文件路径。
    sort : bool
        是否按 datetime, instrument 排序。

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    df = pd.read_csv(p)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    if "instrument" in df.columns:
        df["instrument"] = df["instrument"].astype(str)
    if sort and "datetime" in df.columns and "instrument" in df.columns:
        df = df.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    return df


def write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs: Any) -> None:
    """写入 CSV，统一浮点格式。

    Parameters
    ----------
    df : pd.DataFrame
        待写入数据。
    path : str or Path
        输出路径（父目录不存在时自动创建）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, Any] = {"index": False, "float_format": "%.8f"}
    defaults.update(kwargs)
    df.to_csv(p, **defaults)


def read_pickle(path: Union[str, Path]) -> Any:
    """读取 pickle 文件。"""
    return pd.read_pickle(path)


def write_pickle(obj: Any, path: Union[str, Path]) -> None:
    """写入 pickle 文件（父目录不存在时自动创建）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, p)
