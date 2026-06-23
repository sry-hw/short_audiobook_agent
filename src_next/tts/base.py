"""src_next/tts/base.py

统一 TTS adapter 接口。

所有 TTS 后端（Mock / IndexTTS / CosyVoice / FishPro / Qwen TTS）都要实现
``BaseTTSAdapter``。core 层和测试脚本只依赖这个抽象接口，不直接 import
具体后端。具体使用哪个后端由 ``registry.create_tts_adapter`` + profile 决定。

接口约定：
    输入：list[TTSInstruction] + VoicebankResult + output_dir
    输出：list[AudioSegmentResult]（与 instructions 一一对应，顺序一致）

实现要求：
    * 单条 instruction 失败不阻断其他 instruction；
    * 失败的 instruction 在对应 AudioSegmentResult 里 success=False + error 写明；
    * 缺少 voice_ref 的 instruction 直接 success=False，不发起后端调用；
    * 已存在的 wav 应可复用（缓存）；
    * 支持 dry_run（不真实调用模型，只产出占位 AudioSegmentResult）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult


class TTSError(Exception):
    """TTS 调用失败的统一异常。

    所有 BaseTTSAdapter 实现都应把底层错误（模型缺失、subprocess 失败、配置
    不合法、超时等）包装成 TTSError 抛出。单条 instruction 的失败不抛异常，
    而是写到对应 AudioSegmentResult.error 里——异常只用于整个 adapter 无法
    工作的情况（如配置缺失、引擎路径不存在）。
    """


class BaseTTSAdapter(ABC):
    """所有 TTS 后端的统一接口。

    只暴露一个方法：
        synthesize(instructions, voicebank_result, output_dir, **kwargs)
            -> list[AudioSegmentResult]
    """

    @abstractmethod
    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        **kwargs: Any,
    ) -> list[AudioSegmentResult]:
        """按 instruction 顺序合成每段 wav。

        Args:
            instructions: 通用 TTS 指令列表（已经过 tts_instruction_builder
                把 director_plan + voicebank 合并）。
            voicebank_result: voicebank 层产出。adapter 可以从中查 speaker
                对应的 voice_ref 作为 fallback（首选 instruction.voice_ref）。
            output_dir: 本次 pipeline 的输出根目录；adapter 在其下创建子目录
                （由 output_subdir 配置）存放 wav + log。
            **kwargs: 后端特定参数（如 dry_run / timeout_per_seg / limit）。

        Returns:
            AudioSegmentResult 列表，长度严格等于 instructions，按原顺序排列。
            单条失败不影响其他条——失败条目 success=False 且 error 字段写明原因。
        """
