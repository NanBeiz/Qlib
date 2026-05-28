"""YAML 配置加载 + 路径解析。"""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """返回 quant_csi500_mn/ 项目根目录。

    Returns
    -------
    Path
        项目根目录的绝对路径。
    """
    return Path(__file__).resolve().parents[3]


def load_config(relative_path: str) -> dict[str, Any]:
    """加载 YAML 配置，将 outputs/ 相对路径解析为绝对路径。

    Parameters
    ----------
    relative_path : str
        相对于项目根或 configs/ 的路径，如 "_common.yaml"、"configs/_common.yaml"。

    Returns
    -------
    dict
        解析后的配置字典。
    """
    root = get_project_root()
    # 自动剥离可能的 configs/ 前缀
    if relative_path.startswith("configs/"):
        relative_path = relative_path[len("configs/"):]
    config_path = root / "configs" / relative_path
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_paths(cfg, root)


def _resolve_paths(cfg: Any, root: Path) -> Any:
    """递归解析以 outputs/ 开头的相对路径。"""
    if isinstance(cfg, dict):
        return {k: _resolve_paths(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve_paths(item, root) for item in cfg]
    if isinstance(cfg, str) and cfg.startswith("outputs/"):
        return str(root / cfg)
    return cfg
