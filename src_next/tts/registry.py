"""src_next/tts/registry.py

TTS backend 工厂。

core / 测试脚本只通过本模块的 ``create_tts_adapter`` 创建 adapter，不直接
import 具体后端，这样：
* 切换 backend 只改 profile，不改业务代码；
* 新增 backend 只动 registry + 新 adapter 文件，core 不感知。

懒导入：只有用到某 backend 时才 import 对应模块，避免 import src_next.tts
时把所有后端依赖都拉进来（保持 __init__.py 轻量）。
"""

from __future__ import annotations

from typing import Any

from .base import BaseTTSAdapter, TTSError


def create_tts_adapter(
    backend: str,
    **config: Any,
) -> BaseTTSAdapter:
    """根据 backend 名称创建 TTS adapter。

    Args:
        backend: 后端标识。当前支持：
            - "mock": MockTTSAdapter（离线占位）
            - "indextts": IndexTTSAdapter（subprocess 调本地 IndexTTS CLI，蓝区用）
            - "indextts_http": IndexTTSHTTPAdapter（HTTP 直连服务器，黄区用）
            - "cosyvoice_http": CosyVoiceHTTPAdapter（HTTP 直连服务器，黄区用）
            - "s2pro_http": S2ProTTSAdapter（Fish Audio S2-Pro；v1 转换验证模式）
            - "cosyvoice" / "fishpro" / "qwen_tts": 占位，未实现
        **config: 传给具体 adapter 构造函数的参数（如 engine_root / base_url）。

    Returns:
        BaseTTSAdapter 实例。

    Raises:
        TTSError: backend 未知，或占位 adapter 被调用。
    """
    if backend == "mock":
        from .mock_tts import MockTTSAdapter
        return MockTTSAdapter(**config)

    if backend == "indextts":
        from .indextts_adapter import IndexTTSAdapter
        return IndexTTSAdapter(**config)

    if backend == "indextts_http":
        from .indextts_http import IndexTTSHTTPAdapter
        return IndexTTSHTTPAdapter(**config)

    if backend == "cosyvoice_http":
        from .cosyvoice_http import CosyVoiceHTTPAdapter
        return CosyVoiceHTTPAdapter(**config)

    if backend == "s2pro_http":
        from .s2pro_adapter import S2ProTTSAdapter
        return S2ProTTSAdapter(**config)

    if backend == "cosyvoice":
        from .cosyvoice_adapter import CosyVoiceAdapter
        return CosyVoiceAdapter(**config)

    if backend == "fishpro":
        from .fishpro_adapter import FishProAdapter
        return FishProAdapter(**config)

    if backend == "qwen_tts":
        from .qwen_tts_adapter import QwenTTSAdapter
        return QwenTTSAdapter(**config)

    raise TTSError(
        f"未知 TTS backend: {backend!r}。"
        "当前支持: 'mock', 'indextts', 'indextts_http', 'cosyvoice_http', "
        "'s2pro_http', 'cosyvoice'（占位）, 'fishpro'（占位）, 'qwen_tts'（占位）。"
    )
