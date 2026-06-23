"""src_next/tts/qwen_tts_adapter.py

Qwen3-TTS（合成）adapter（占位，未实现）。

文件存在是为了让 ``registry.create_tts_adapter("qwen_tts", ...)`` 能 import
通过；调用时抛 NotImplementedError。

注意：本 adapter 是 **正文合成**，不是音色生成（那是 voicebank/qwen_voicegenerator
的事）。Qwen3-TTS 支持 emotion / volume / speed 等参数，表达力比 IndexTTS 强。
"""

from __future__ import annotations

from typing import Any

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult

from .base import BaseTTSAdapter, TTSError


class QwenTTSAdapter(BaseTTSAdapter):
    """Qwen3-TTS 合成后端占位 adapter。"""

    NOT_IMPLEMENTED = (
        "QwenTTSAdapter 尚未实现。请参考 indextts_adapter.py 的结构补全。"
        "Qwen3-TTS 原生支持 emotion / volume / speed 等参数，"
        "TTSInstruction 的通用字段可以比较直接地映射过去。"
        "注意：本 adapter 是正文合成，不要和 voicebank/qwen_voicegenerator 混淆。"
    )

    def __init__(self, **_config: Any) -> None:
        pass

    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        **kwargs: Any,
    ) -> list[AudioSegmentResult]:
        raise TTSError(self.NOT_IMPLEMENTED)
