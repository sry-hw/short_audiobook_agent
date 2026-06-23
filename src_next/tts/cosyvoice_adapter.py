"""src_next/tts/cosyvoice_adapter.py

CosyVoice TTS adapter（占位，未实现）。

文件存在是为了让 ``registry.create_tts_adapter("cosyvoice", ...)`` 能 import
通过；构造或调用时抛 NotImplementedError，避免被误以为已可用。

实现时请参考 ``indextts_adapter.py`` 的 subprocess + 缓存 + 单条失败隔离模式。
CosyVoice2 支持 instruct 模式，因此 emotion / volume / pace 可以以自然语言
prompt 形式传给 ``instruct_text`` 字段，比 IndexTTS 表达力更强。
"""

from __future__ import annotations

from typing import Any

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult

from .base import BaseTTSAdapter, TTSError


class CosyVoiceAdapter(BaseTTSAdapter):
    """CosyVoice 后端占位 adapter。"""

    NOT_IMPLEMENTED = (
        "CosyVoiceAdapter 尚未实现。请参考 indextts_adapter.py 的结构补全："
        "构造参数（engine_root / model_dir / 端口 / python_executable），"
        "synthesize() 的 subprocess 调用 + 缓存 + 失败隔离。"
        "CosyVoice2 支持 instruct_text，emotion/volume/pace 可以走自然语言 prompt。"
    )

    def __init__(self, **_config: Any) -> None:
        # 构造先放行，便于 registry / dry-run 路径走通；
        # 真正调用 synthesize 时再抛。
        pass

    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        **kwargs: Any,
    ) -> list[AudioSegmentResult]:
        raise TTSError(self.NOT_IMPLEMENTED)
