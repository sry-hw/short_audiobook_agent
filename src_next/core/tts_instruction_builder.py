"""src_next/core/tts_instruction_builder.py

把 analysis 层产出的 resolved_segments / characters / director_plan / voicebank_result
合并成 ``TTSInstruction`` 列表，供后续 TTS adapter 消费。

数据流位置：
    Segment[] + CharacterProfile[] + DirectorInstruction[] + VoicebankResult
        → build_tts_instructions(...)
        → TTSInstruction[]（和 segments 一一对应，顺序一致）

输出定位：
    TTSInstruction 是**模型无关的通用合成指令**。本 builder 不关心
    IndexTTS / CosyVoice / FishPro / Qwen TTS 的具体参数格式，只把
    导演层（DirectorInstruction）的语义字段 + 音色参考（voice_ref）
    + 段落信息打包成一个稳定的中间结构。后续 ``src_next/tts/`` 各
    adapter 负责把 TTSInstruction 翻译成具体后端调用。

合并规则：
    1. 按 segment_id 把 DirectorInstruction 对齐到 Segment。
       找不到 → 按 segment_type 生成 fallback director（不抛错）。
    2. 按 speaker 把 voice_ref 对齐到 Segment。
       speaker 没拿到 → fallback narrator；narrator 也没 → 空字符串。
       fallback 路径在 metadata 里记录 ``missing_voice_ref`` +
       ``voice_ref_fallback``。
    3. 所有数值字段做范围 clamp；枚举字段做白名单校验；
       list 字段做长度 / 空值过滤。
    4. ``output_filename`` 直接用 ``<segment_id>.wav``。
"""

from __future__ import annotations

from typing import Any

from .data_models import (
    CharacterProfile,
    DirectorInstruction,
    Segment,
    TTSInstruction,
    VoicebankResult,
)


# ── 字段合法值集合（和 analysis/story_director 保持一致） ───────────────────

_VALID_VOLUMES = {"soft", "normal", "strong"}
_VALID_PITCHES = {"low", "medium_low", "medium", "medium_high", "high"}


# ── 范围常量 ────────────────────────────────────────────────────────────────

_PACE_MIN, _PACE_MAX = 0.75, 1.30
_PAUSE_MIN, _PAUSE_MAX = 0.2, 1.0
_INTENSITY_MIN, _INTENSITY_MAX = 0.0, 1.0
_MAX_STRESS_WORDS = 3


# ── DirectorInstruction fallback（按 segment_type） ─────────────────────────
#
# 当 LLM director_plan 没覆盖到某个 segment 时用这些兜底。
# 和 analysis/story_director.py 的 fallback 思路一致但更保守（这里只是
# builder 层兜底，不再做关键词推断，避免和 analysis 层重复逻辑）。

_NARRATION_FALLBACK = DirectorInstruction(
    segment_id="",  # 占位，build 时按 seg 重填
    speaker="narrator",
    emotion="calm",
    emotion_intensity=0.4,
    pace=0.95,
    tone="warm",
    volume="normal",
    pitch="medium",
    pause_hint=0.6,
    stress_words=[],
    delivery_instruction="以温和清晰的旁白语气叙述，保持自然停顿。",
)

_DIALOGUE_FALLBACK = DirectorInstruction(
    segment_id="",
    speaker="",
    emotion="neutral",
    emotion_intensity=0.5,
    pace=1.0,
    tone="normal",
    volume="normal",
    pitch="medium",
    pause_hint=0.4,
    stress_words=[],
    delivery_instruction="以自然对白语气说出，保持清晰表达。",
)

_INNER_THOUGHT_FALLBACK = DirectorInstruction(
    segment_id="",
    speaker="",
    emotion="calm",
    emotion_intensity=0.5,
    pace=0.9,
    tone="soft",
    volume="soft",
    pitch="medium",
    pause_hint=0.5,
    stress_words=[],
    delivery_instruction="以内心独白的方式轻声表达，语速稍慢。",
)


def _fallback_director(seg: Segment) -> DirectorInstruction:
    """按 segment_type 选 fallback director，并补上 segment_id / speaker。"""
    if seg.segment_type == "dialogue":
        fb = _DIALOGUE_FALLBACK
    elif seg.segment_type == "inner_thought":
        fb = _INNER_THOUGHT_FALLBACK
    else:
        fb = _NARRATION_FALLBACK
    # 拷贝一份并覆盖 id / speaker，避免改模块级常量
    return DirectorInstruction(
        segment_id=seg.segment_id,
        speaker=seg.speaker,
        emotion=fb.emotion,
        emotion_intensity=fb.emotion_intensity,
        pace=fb.pace,
        tone=fb.tone,
        volume=fb.volume,
        pitch=fb.pitch,
        pause_hint=fb.pause_hint,
        stress_words=list(fb.stress_words),
        delivery_instruction=fb.delivery_instruction,
    )


# ── 入口 ────────────────────────────────────────────────────────────────────

def build_tts_instructions(
    segments: list[Segment],
    characters: list[CharacterProfile],
    director_plan: list[DirectorInstruction],
    voicebank_result: VoicebankResult,
) -> list[TTSInstruction]:
    """组装 TTSInstruction 列表（1:1 对齐 segments）。

    Args:
        segments: resolved segments（已经过 quote_classifier + resolve_speakers）。
        characters: analyze_characters 输出（含 narrator）。当前 builder 主要
            用它做完整性参考，后续若需要按 character 调整 voice_prompt / 情绪
            再扩展。
        director_plan: generate_director_plan 输出。允许数量少于 segments
            （缺失走 fallback）。
        voicebank_result: prepare_voicebank 输出。``speaker_to_voice`` 里缺
            某个 speaker 时走 narrator / 空字符串兜底。

    Returns:
        TTSInstruction 列表，长度严格等于 segments，按 segment 顺序排列。
    """
    if not segments:
        return []

    director_by_id = {d.segment_id: d for d in director_plan}
    voice_by_speaker = (voicebank_result.speaker_to_voice or {}) if voicebank_result else {}
    _ = characters  # 预留：未来按 character 调整参数

    instructions: list[TTSInstruction] = []
    for seg in segments:
        # 1. 对齐 DirectorInstruction
        director = director_by_id.get(seg.segment_id)
        has_director = director is not None
        if director is None:
            director = _fallback_director(seg)

        # 2. 对齐 voice_ref（speaker → narrator → 空）
        voice_ref, voice_meta = _resolve_voice_ref(
            seg.speaker, voice_by_speaker
        )

        # 3. 清洗字段并打包
        instructions.append(
            TTSInstruction(
                segment_id=seg.segment_id,
                speaker=seg.speaker,
                text=seg.text,
                segment_type=seg.segment_type,
                voice_ref=voice_ref,
                emotion=str(director.emotion or "neutral"),
                emotion_intensity=_clamp(
                    _safe_float(director.emotion_intensity, 0.5),
                    _INTENSITY_MIN, _INTENSITY_MAX,
                ),
                pace=_clamp(
                    _safe_float(director.pace, 1.0),
                    _PACE_MIN, _PACE_MAX,
                ),
                tone=str(director.tone or "normal"),
                volume=_clean_volume(director.volume),
                pitch=_clean_pitch(director.pitch),
                pause_hint=_clamp(
                    _safe_float(director.pause_hint, 0.4),
                    _PAUSE_MIN, _PAUSE_MAX,
                ),
                stress_words=_clean_stress_words(director.stress_words),
                delivery_instruction=_clean_delivery(director.delivery_instruction),
                output_filename=f"{seg.segment_id}.wav",
                metadata={
                    "source_segment_type": seg.segment_type,
                    "has_director_instruction": has_director,
                    "has_voice_ref": bool(voice_ref),
                    **voice_meta,
                },
            )
        )

    return instructions


# ── voice_ref 解析 ──────────────────────────────────────────────────────────

def _resolve_voice_ref(
    speaker: str,
    voice_by_speaker: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """按 speaker → narrator → 空字符串 三级 fallback。

    Returns:
        (voice_ref, metadata_extra)
        metadata_extra 可能包含：
            - ``missing_voice_ref=True`` + ``voice_ref_fallback="narrator"|""
              （仅当 speaker 自身没匹配）
    """
    direct = voice_by_speaker.get(speaker)
    if direct:
        return str(direct), {}

    # speaker 自身没拿到，走 narrator
    narrator = voice_by_speaker.get("narrator")
    if narrator:
        return str(narrator), {
            "missing_voice_ref": True,
            "voice_ref_fallback": "narrator",
        }

    # narrator 也没有
    return "", {
        "missing_voice_ref": True,
        "voice_ref_fallback": "",
    }


# ── 字段清洗 ────────────────────────────────────────────────────────────────

def _clean_volume(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_VOLUMES else "normal"


def _clean_pitch(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_PITCHES else "medium"


def _clean_stress_words(raw: Any) -> list[str]:
    """必须 list[str]，最多 3 个，过滤空字符串和超长项。"""
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or len(s) > 12 or s in seen:
            continue
        seen.add(s)
        result.append(s)
        if len(result) >= _MAX_STRESS_WORDS:
            break
    return result


def _clean_delivery(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if len(s) > 120:
        s = s[:120]
    return s


# ── 通用工具 ────────────────────────────────────────────────────────────────

def _safe_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _clamp(v: float, low: float, high: float) -> float:
    if v < low:
        return low
    if v > high:
        return high
    return v
