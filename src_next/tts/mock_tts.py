"""src_next/tts/mock_tts.py

离线 MockTTSAdapter，用于验证数据流。

不调用任何真实模型，每条 instruction 返回一个 ``mock://<segment_id>.wav``
占位路径的 AudioSegmentResult。可选 dry_run / 缺 voice_ref 模拟失败路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult

from .base import BaseTTSAdapter


_DEFAULT_OUTPUT_SUBDIR = "audio_segments"


class MockTTSAdapter(BaseTTSAdapter):
    """离线占位 TTS adapter。

    构造参数仅用于和真实 adapter 保持一致（方便 profile 切换），mock 实际不消费：
        output_subdir: 伪 wav 落盘的子目录名（仅用于把占位文件写到磁盘）。

    dry_run / 缺 voice_ref 时返回 success=False 的 AudioSegmentResult。
    """

    def __init__(
        self,
        *,
        output_subdir: str = _DEFAULT_OUTPUT_SUBDIR,
        **_kwargs: Any,
    ) -> None:
        self.output_subdir = (output_subdir or _DEFAULT_OUTPUT_SUBDIR).strip() or _DEFAULT_OUTPUT_SUBDIR

    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        *,
        dry_run: bool = False,
        **_kwargs: Any,
    ) -> list[AudioSegmentResult]:
        out_dir = Path(output_dir).expanduser().resolve() / self.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[AudioSegmentResult] = []
        for inst in instructions:
            # 缺 voice_ref → 失败
            if not inst.voice_ref:
                results.append(AudioSegmentResult(
                    segment_id=inst.segment_id,
                    speaker=inst.speaker,
                    audio_path=None,
                    success=False,
                    error="mock: missing voice_ref",
                ))
                continue

            if dry_run:
                results.append(AudioSegmentResult(
                    segment_id=inst.segment_id,
                    speaker=inst.speaker,
                    audio_path=None,
                    success=False,
                    error="mock: dry_run",
                ))
                continue

            # 写一个占位 wav marker 文件（不是真实 wav，只是 mock 协议路径）
            marker_path = out_dir / inst.output_filename
            marker_path.write_text(
                f"[MOCK TTS]\nsegment_id={inst.segment_id}\nspeaker={inst.speaker}\n"
                f"text={inst.text}\n",
                encoding="utf-8",
            )
            results.append(AudioSegmentResult(
                segment_id=inst.segment_id,
                speaker=inst.speaker,
                audio_path=f"mock://{inst.output_filename}",
                success=True,
            ))
        return results
