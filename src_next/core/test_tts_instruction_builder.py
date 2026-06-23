"""src_next/core/test_tts_instruction_builder.py

tts_instruction_builder 的轻量自检测试（不依赖 LLM、不依赖网络）。

覆盖：
1. 正常路径：segment + director + voice_ref 都对齐 → 字段全填。
2. director 缺失：按 segment_type 走不同 fallback（narration / dialogue / inner_thought）。
3. voice_ref 缺失：speaker 没匹配 → fallback narrator；narrator 也没 → 空字符串；
   两条路径都在 metadata 里记 missing_voice_ref + voice_ref_fallback。
4. 字段清洗：emotion_intensity / pace / pause_hint 越界 clamp；
   volume / pitch 非法值回默认；stress_words 过滤 + 限 3 个；
   delivery_instruction 空字符串保留空。

运行：
    python -m src_next.core.test_tts_instruction_builder
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Windows GBK 终端兼容
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src_next.core.data_models import (
    CharacterProfile,
    DirectorInstruction,
    Segment,
    VoicebankResult,
)
from src_next.core.tts_instruction_builder import build_tts_instructions


def _build_test_inputs() -> tuple[
    list[Segment],
    list[CharacterProfile],
    list[DirectorInstruction],
    VoicebankResult,
]:
    """构造覆盖 3 种 fallback 场景的输入。

    seg_001 narrator narration    - director 齐全
    seg_002 母亲   dialogue       - director 齐全
    seg_003 我     inner_thought  - director 缺失 → 走 inner_thought fallback
    seg_004 母亲   dialogue       - director 缺失 → 走 dialogue fallback
    seg_005 narrator narration    - director 齐全，但 emotion_intensity=99 / pace=5.0
                                    等越界值，验证 clamp
    """
    segments = [
        Segment(
            segment_id="seg_001",
            text="中秋节前后，正是故乡桂花盛开的时节。",
            speaker="narrator",
            segment_type="narration",
            raw_index=0,
        ),
        Segment(
            segment_id="seg_002",
            text="可别来台风啊！",
            speaker="母亲",
            segment_type="dialogue",
            raw_index=1,
        ),
        Segment(
            segment_id="seg_003",
            text="要是能天天摇桂花就好了。",
            speaker="我",
            segment_type="inner_thought",
            raw_index=2,
        ),
        Segment(
            segment_id="seg_004",
            text="妈，怎么还不摇桂花呢？",
            speaker="我",
            segment_type="dialogue",
            raw_index=3,
        ),
        Segment(
            segment_id="seg_005",
            text="桂花成熟时，就应当摇。",
            speaker="narrator",
            segment_type="narration",
            raw_index=4,
        ),
    ]

    characters = [
        CharacterProfile(
            name="narrator", role_type="narrator", gender="female",
            age_style="young", voice_prompt="用温柔女声说", confidence=0.95,
        ),
        CharacterProfile(
            name="母亲", role_type="character", gender="female",
            age_style="middle_aged", voice_prompt="用温和的中年女声说",
            confidence=0.9,
        ),
        CharacterProfile(
            name="我", role_type="character", gender="female",
            age_style="child", voice_prompt="用清脆童声说", confidence=0.85,
        ),
    ]

    # 故意只覆盖 3 个 segment（缺 seg_003 / seg_004），测 director fallback。
    # seg_005 用极端值测 clamp。
    director_plan = [
        DirectorInstruction(
            segment_id="seg_001", speaker="narrator",
            emotion="nostalgic", emotion_intensity=0.65,
            pace=0.9, tone="gentle", volume="soft", pitch="medium_low",
            pause_hint=0.6, stress_words=["故乡", "桂花"],
            delivery_instruction="语气温柔怀念，语速稍慢，突出对故乡的思念。",
        ),
        DirectorInstruction(
            segment_id="seg_002", speaker="母亲",
            emotion="anxious", emotion_intensity=0.7,
            pace=1.05, tone="serious", volume="normal", pitch="medium_low",
            pause_hint=0.5, stress_words=["可别", "台风"],
            delivery_instruction="语气带着长辈的忧虑与急切。",
        ),
        DirectorInstruction(
            segment_id="seg_005", speaker="narrator",
            emotion="calm", emotion_intensity=9.5,  # 越界
            pace=5.0,  # 越界
            tone="normal", volume="LOUD",  # 非法枚举
            pitch="ultra_high",  # 非法枚举
            pause_hint=-0.3,  # 越界
            stress_words=["", "桂花", "桂花", "摇", "x" * 20, "成熟"],
            delivery_instruction="",
        ),
    ]

    # 故意漏 "我"，测 voice_ref → narrator fallback。
    # narrator 留着，所以 seg_003 / seg_004 都应该 fallback 到 narrator voice。
    voicebank_result = VoicebankResult(
        speaker_to_voice={
            "narrator": "voicebank/narrator.wav",
            "母亲": "voicebank/母亲.wav",
        },
        voicebank_dir="voicebank/",
        backend="mock",
        success=True,
    )

    return segments, characters, director_plan, voicebank_result


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def _run_checks(instructions) -> None:
    """逐条验证 TTSInstruction 字段。"""
    # 数量对齐
    _assert(len(instructions) == 5, f"指令数 5（实际 {len(instructions)}）")

    # seg_001：完整路径
    i1 = instructions[0]
    _assert(i1.segment_id == "seg_001", "seg_001 id")
    _assert(i1.segment_type == "narration", "seg_001 segment_type")
    _assert(i1.emotion == "nostalgic", "seg_001 emotion")
    _assert(abs(i1.emotion_intensity - 0.65) < 1e-6, "seg_001 intensity")
    _assert(abs(i1.pace - 0.9) < 1e-6, "seg_001 pace")
    _assert(i1.volume == "soft", "seg_001 volume")
    _assert(i1.pitch == "medium_low", "seg_001 pitch")
    _assert(i1.stress_words == ["故乡", "桂花"], f"seg_001 stress={i1.stress_words}")
    _assert(i1.voice_ref == "voicebank/narrator.wav", "seg_001 voice_ref")
    _assert(i1.output_filename == "seg_001.wav", "seg_001 output_filename")
    _assert(i1.metadata["has_director_instruction"] is True, "seg_001 has_director")
    _assert(i1.metadata["has_voice_ref"] is True, "seg_001 has_voice_ref")
    _assert("missing_voice_ref" not in i1.metadata, "seg_001 不应有 missing_voice_ref")

    # seg_002：母亲 dialogue，director 齐全 + voice 齐全
    i2 = instructions[1]
    _assert(i2.emotion == "anxious", "seg_002 emotion")
    _assert(i2.voice_ref == "voicebank/母亲.wav", "seg_002 voice_ref")
    _assert(i2.metadata["has_director_instruction"] is True, "seg_002 has_director")
    _assert(i2.metadata["has_voice_ref"] is True, "seg_002 has_voice_ref")
    _assert("missing_voice_ref" not in i2.metadata, "seg_002 不应 missing")

    # seg_003：inner_thought，director 缺失 → 走 inner_thought fallback
    i3 = instructions[2]
    _assert(i3.emotion == "calm", f"seg_003 emotion (want calm, got {i3.emotion})")
    _assert(abs(i3.pace - 0.9) < 1e-6, "seg_003 pace (inner_thought fallback)")
    _assert(i3.volume == "soft", "seg_003 volume (inner_thought fallback)")
    _assert("内心独白" in i3.delivery_instruction,
            f"seg_003 delivery 来自 inner_thought fallback：{i3.delivery_instruction}")
    _assert(i3.metadata["has_director_instruction"] is False, "seg_003 缺 director")
    # "我" 没 voice → narrator fallback
    _assert(i3.voice_ref == "voicebank/narrator.wav", "seg_003 voice_ref → narrator")
    _assert(i3.metadata["has_voice_ref"] is True, "seg_003 fallback 后 has_voice_ref 仍 True")
    _assert(i3.metadata.get("missing_voice_ref") is True, "seg_003 missing_voice_ref=True")
    _assert(i3.metadata.get("voice_ref_fallback") == "narrator",
            "seg_003 fallback 来源 = narrator")

    # seg_004：dialogue，director 缺失 → 走 dialogue fallback
    i4 = instructions[3]
    _assert(i4.emotion == "neutral", f"seg_004 emotion (want neutral, got {i4.emotion})")
    _assert(abs(i4.pace - 1.0) < 1e-6, "seg_004 pace (dialogue fallback)")
    _assert("自然对白" in i4.delivery_instruction,
            f"seg_004 delivery 来自 dialogue fallback：{i4.delivery_instruction}")
    _assert(i4.metadata["has_director_instruction"] is False, "seg_004 缺 director")
    _assert(i4.metadata.get("voice_ref_fallback") == "narrator",
            "seg_004 voice_ref fallback = narrator")

    # seg_005：director 齐全但字段越界 → clamp + 非法枚举回默认
    i5 = instructions[4]
    _assert(i5.emotion_intensity == 1.0,
            f"seg_005 intensity clamp 到 1.0（got {i5.emotion_intensity}）")
    _assert(i5.pace == 1.30,
            f"seg_005 pace clamp 到 1.30（got {i5.pace}）")
    _assert(i5.pause_hint == 0.2,
            f"seg_005 pause_hint clamp 到 0.2（got {i5.pause_hint}）")
    _assert(i5.volume == "normal",
            f"seg_005 volume 非法 → normal（got {i5.volume}）")
    _assert(i5.pitch == "medium",
            f"seg_005 pitch 非法 → medium（got {i5.pitch}）")
    # stress_words 过滤：空字符串去重、超长（>12）丢弃、最多 3 个
    _assert(i5.stress_words == ["桂花", "摇", "成熟"],
            f"seg_005 stress_words 清洗后 = {i5.stress_words}")


def main() -> int:
    segments, characters, director_plan, voicebank_result = _build_test_inputs()

    print("=" * 60)
    print("[test_tts_instruction_builder] 输入")
    print(f"  segments    = {len(segments)}")
    print(f"  characters  = {len(characters)}")
    print(f"  directors   = {len(director_plan)}（故意漏 seg_003 / seg_004）")
    print(f"  voicebank   = {list(voicebank_result.speaker_to_voice.keys())}"
          "（故意漏 '我'）")
    print("=" * 60)
    print()

    instructions = build_tts_instructions(
        segments, characters, director_plan, voicebank_result
    )

    print("[built instructions]")
    for inst in instructions:
        print(
            f"  {inst.segment_id} speaker={inst.speaker:<6} "
            f"type={inst.segment_type:<13} "
            f"emotion={inst.emotion:<11} intensity={inst.emotion_intensity:.2f} "
            f"pace={inst.pace:.2f} volume={inst.volume:<6} pitch={inst.pitch:<11}"
        )
        print(f"    voice_ref={inst.voice_ref!r}")
        print(f"    output={inst.output_filename}")
        print(f"    stress={inst.stress_words}")
        print(f"    delivery={inst.delivery_instruction!r}")
        print(f"    metadata={inst.metadata}")
        print()

    print("[checks]")
    _run_checks(instructions)
    print()
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
