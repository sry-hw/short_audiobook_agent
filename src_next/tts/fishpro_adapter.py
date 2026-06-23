"""src_next/tts/fishpro_adapter.py

FishPro（Fish-Speech）TTS adapter（占位，未实现）。

文件存在是为了让 ``registry.create_tts_adapter("fishpro", ...)`` 能 import
通过；调用时抛 NotImplementedError。

实现时请参考 ``indextts_adapter.py``。FishPro 通常以 HTTP server 形式部署，
因此 adapter 大概率走 ``requests.post`` 而非 subprocess（与 IndexTTS 不同）。
"""

from __future__ import annotations

from typing import Any

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult

from .base import BaseTTSAdapter, TTSError


class FishProAdapter(BaseTTSAdapter):
    """FishPro 后端占位 adapter。"""

    NOT_IMPLEMENTED = (
        "FishProAdapter 尚未实现。请参考 indextts_adapter.py 的结构补全。"
        "FishPro 大概率以 HTTP server 形式部署，adapter 走 requests.post 即可，"
        "不需要 subprocess + python_executable 那套 WSL 包装逻辑。"
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
