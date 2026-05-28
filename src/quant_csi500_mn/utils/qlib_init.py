"""统一 Qlib 初始化，带缓存配置和防重复初始化。"""

from typing import Optional

import qlib
from qlib.config import REG_CN
from qlib.log import get_module_logger

logger = get_module_logger(__name__)
_QLIB_INITIALIZED = False


def init_qlib(
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    region: str = REG_CN,
    cache_dir: Optional[str] = None,
    enable_cache: bool = False,
) -> None:
    """初始化 Qlib，模块级标志防重复初始化。

    Parameters
    ----------
    provider_uri : str
        Qlib 数据目录。
    region : str
        市场区域，默认 REG_CN（中国市场）。
    cache_dir : str, optional
        缓存目录路径。
    enable_cache : bool
        是否启用 DiskExpressionCache 和 DiskDatasetCache。
    """
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED:
        logger.debug("qlib already initialized, skipping")
        return

    kwargs: dict = {}
    if enable_cache:
        kwargs["expression_cache"] = "DiskExpressionCache"
        kwargs["dataset_cache"] = "DiskDatasetCache"
        if cache_dir:
            kwargs["local_cache_path"] = cache_dir

    qlib.init(provider_uri=provider_uri, region=region, **kwargs)
    _QLIB_INITIALIZED = True
    logger.info("qlib initialized: provider_uri=%s, cache=%s", provider_uri, enable_cache)


def reset_qlib() -> None:
    """重置初始化标志（供测试隔离使用）。"""
    global _QLIB_INITIALIZED
    _QLIB_INITIALIZED = False
    logger.debug("qlib init flag reset")
