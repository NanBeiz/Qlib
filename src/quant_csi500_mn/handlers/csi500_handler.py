"""CSI500 14 因子 DataHandlerLP 子类。

继承 qlib.data.dataset.handler.DataHandlerLP，提供因子名称和标签名称的
便捷属性。支持从 factor_yaml 自动构造 QlibDataLoader 配置。
所有参数可序列化，保证 qrun 兼容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger

logger = get_module_logger(__name__)

FACTOR_NAMES = [
    "lowvol", "momentum_12_1", "reversal_20",
    "rev_1", "rev_5", "resid_rev_5",
    "mom_20_5", "mom_60_5",
    "vol_20", "idio_vol_60",
    "amihud_20", "liq_20",
    "close_pos", "ret_amount_corr_20",
]

LABEL_NAME = "LABEL0"
LABEL_EXPRESSION = "Ref($close, -2) / Ref($close, -1) - 1"


class CSI500AlphaHandler(DataHandlerLP):
    """CSI500 14 因子 Alpha Handler。

    封装 QlibDataLoader + infer/learn processor 链，
    为 DatasetH 提供训练/验证/测试数据。

    两种使用方式：
    1. 传入 factor_yaml（字符串路径），自动从 alpha14.yaml 读取
       表达式和名称，构造 QlibDataLoader。
    2. 传入 data_loader（dict），使用自定义 loader 配置。

    Parameters
    ----------
    instruments : str, optional
        股票池标识，如 "csi500"。
    start_time : str, optional
        数据起始日期。
    end_time : str, optional
        数据结束日期。
    factor_yaml : str, optional
        alpha14.yaml 的路径（相对于项目根或绝对路径）。
        提供后自动构造 QlibDataLoader。
    data_loader : dict, optional
        QlibDataLoader 配置字典（显式覆盖 factor_yaml）。
    infer_processors : list, optional
        推理处理器链（train/valid/test 均应用）。
    learn_processors : list, optional
        学习处理器链（仅 train 时在 infer 基础上叠加）。
    shared_processors : list, optional
        共享处理器链。
    process_type : str
        处理类型，默认 "append"（learn = infer + learn_processors）。
    drop_raw : bool
        是否丢弃原始数据，默认 False。
    """

    def __init__(
        self,
        instruments: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        factor_yaml: Optional[str] = None,
        data_loader: Optional[dict] = None,
        infer_processors: Optional[list] = None,
        learn_processors: Optional[list] = None,
        shared_processors: Optional[list] = None,
        process_type: str = DataHandlerLP.PTYPE_A,
        drop_raw: bool = False,
    ):
        infer_processors = infer_processors or []
        learn_processors = learn_processors or []
        shared_processors = shared_processors or []

        # 如果未显式提供 data_loader，从 factor_yaml 自动构造
        if data_loader is None and factor_yaml is not None:
            data_loader = self._build_data_loader_from_yaml(factor_yaml)

        logger.info(
            "initializing CSI500AlphaHandler: instruments=%s, "
            "start=%s, end=%s, infer_procs=%d, learn_procs=%d",
            instruments, start_time, end_time,
            len(infer_processors), len(learn_processors),
        )

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            shared_processors=shared_processors,
            process_type=process_type,
            drop_raw=drop_raw,
        )

        logger.info("CSI500AlphaHandler initialized successfully")

    @property
    def factor_names(self) -> list[str]:
        """返回 14 个因子名称列表。"""
        return list(FACTOR_NAMES)

    @property
    def label_name(self) -> str:
        """返回标签列名。"""
        return LABEL_NAME

    @staticmethod
    def _build_data_loader_from_yaml(factor_yaml: str) -> dict:
        """从 alpha14.yaml 自动构造 QlibDataLoader 配置字典。

        Parameters
        ----------
        factor_yaml : str
            alpha14.yaml 的路径。

        Returns
        -------
        dict
            含 class, module_path, kwargs 的 QlibDataLoader 配置。
        """
        yaml_path = Path(factor_yaml)
        if not yaml_path.is_absolute():
            # 相对于项目根（src/quant_csi500_mn/utils/ 的上三级）
            project_root = Path(__file__).resolve().parents[3]
            yaml_path = project_root / "configs" / factor_yaml
            if not yaml_path.exists():
                # 回退：直接按给定路径尝试
                yaml_path = Path(factor_yaml)

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        factors = cfg.get("factors", [])
        expressions = [f["expression"] for f in factors]
        names = [f["name"] for f in factors]

        logger.info("loaded %d factors from %s", len(factors), yaml_path)

        return {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": (expressions, names),
                    "label": ([LABEL_EXPRESSION], [LABEL_NAME]),
                },
                "freq": "day",
            },
        }
