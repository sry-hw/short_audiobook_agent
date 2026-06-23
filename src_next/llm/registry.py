"""src_next/llm/registry.py

LLM backend 工厂。

core / pipeline / 测试脚本只通过本模块的 ``create_llm_client`` 创建 client，
不直接 import 具体后端，这样：
* 切换 backend 只改 profile 配置，不改业务代码；
* 新增 backend 只动 registry + 新 client 文件，core 不感知。

懒导入：只有用到某 backend 时才 import 对应模块，避免 import src_next.llm
时把所有后端依赖都拉进来（保持 __init__.py 轻量）。

字段过滤：profile yaml 里的 ``base_url_env`` / ``api_key_env`` 等是文档型字段
（告诉用户该 backend 需要哪些环境变量），不是构造函数参数。本 registry 会
按 backend 白名单过滤 config，避免 ``TypeError: unexpected keyword argument``。
"""

from __future__ import annotations

from typing import Any

from .base import BaseLLMClient, LLMError


# 各 backend 构造函数允许的字段（其他字段会被静默丢弃）
_QWEN_KWARGS = {"base_url", "api_key", "model", "timeout", "bypass_proxy"}
_GEMMA4_KWARGS = {"base_url", "api_key", "model", "timeout", "bypass_proxy"}


def _filter(config: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    """只保留 allowed 集合里的 key。"""
    return {k: v for k, v in config.items() if k in allowed}


def create_llm_client(
    backend: str,
    **config: Any,
) -> BaseLLMClient:
    """根据 backend 名称创建 LLM client。

    Args:
        backend: 后端标识。当前支持：
            - "mock": MockLLMClient（离线占位，无 config）
            - "qwen_http": QwenHTTPClient（OpenAI-compatible HTTP）
            - "gemma4_http": Gemma4HTTPClient
        **config: 传给具体 client 构造函数的参数。文档型字段（如
            ``base_url_env`` / ``api_key_env``）会被静默丢弃——这些字段
            告诉用户该 backend 读哪些环境变量，但 client 自己会从 env 读取。

    Returns:
        BaseLLMClient 实例。

    Raises:
        LLMError: backend 未知，或 client 构造失败（缺 env / 配置不合法）。
    """
    if backend == "mock":
        # MockLLMClient 不需要 config
        from .mock_llm import MockLLMClient
        return MockLLMClient()

    if backend == "qwen_http":
        from .qwen_http import QwenHTTPClient
        try:
            return QwenHTTPClient(**_filter(config, _QWEN_KWARGS))
        except TypeError as err:
            # 理论上不会触发，因为已经过滤了；防御性兜底
            raise LLMError(f"QwenHTTPClient 构造参数错误：{err}") from err

    if backend == "gemma4_http":
        from .gemma4_http import Gemma4HTTPClient
        try:
            return Gemma4HTTPClient(**_filter(config, _GEMMA4_KWARGS))
        except TypeError as err:
            raise LLMError(f"Gemma4HTTPClient 构造参数错误：{err}") from err

    raise LLMError(
        f"未知 LLM backend: {backend!r}。"
        "当前支持: 'mock', 'qwen_http', 'gemma4_http'。"
    )
