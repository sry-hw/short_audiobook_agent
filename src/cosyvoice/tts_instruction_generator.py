"""CosyVoice instruct 模式 TTS 指令生成器。

将 director_plan + characters + voice_bank 映射为 CosyVoice /v1/cosyvoice/generate 可执行的指令。
"""

from pathlib import Path
from typing import Dict


# ----------------------------------------------------------------------
# director_plan → prompt_text 映射
# ----------------------------------------------------------------------

_EMOTION_MAP: Dict[tuple, str] = {
    # emotion → intensity → 基础语气描述
    ("平静", "轻"): "用平静的语气说",
    ("平静", "中"): "用平静略带起伏的语气说",
    ("平静", "强"): "用平静沉稳的语气说",
    ("温馨", "轻"): "用温馨的语气说",
    ("温馨", "中"): "用温馨的语气说",
    ("温馨", "强"): "用温馨感人的语气说",
    ("担忧", "轻"): "用略带担忧的语气说",
    ("担忧", "中"): "用担忧的语气说，语速稍慢",
    ("担忧", "强"): "用明显担忧的语气说，语速较慢，声音低沉",
    ("坚定", "轻"): "用坚定的语气说",
    ("坚定", "中"): "用坚定的语气说",
    ("坚定", "强"): "用坚定沉着的语气说，语速缓慢，字字清晰",
    ("恐惧", "轻"): "用略带紧张的语气说",
    ("恐惧", "中"): "用紧张的语气说",
    ("恐惧", "强"): "用恐惧紧张的语气说，语速加快，声音发紧",
    ("兴奋", "轻"): "用兴奋的语气说",
    ("兴奋", "中"): "用兴奋的语气说，语速加快",
    ("兴奋", "强"): "用非常兴奋激动的语气说，语速快，语调上扬",
    ("急切", "轻"): "用急切的语气说",
    ("急切", "中"): "用急切的语气说，语速稍快",
    ("急切", "强"): "用非常急切的语气说，语速快，有紧迫感",
    ("惊讶", "轻"): "用略带惊讶的语气说",
    ("惊讶", "中"): "用惊讶的语气说",
    ("惊讶", "强"): "用惊讶的语气说，语调上扬",
    ("愤怒", "轻"): "用愤怒的语气说，声音提高",
    ("愤怒", "中"): "用愤怒的语气说，声音提高",
    ("愤怒", "强"): "用强烈愤怒的语气说，声音明显提高，情绪激动",
    ("伤感", "轻"): "用伤感的语气说，语速偏慢",
    ("伤感", "中"): "用伤感的语气说，语速偏慢",
    ("伤感", "强"): "用悲伤的语气说，语速很慢，声音低沉",
    ("诚恳", "轻"): "用诚恳的语气说",
    ("诚恳", "中"): "用诚恳的语气说",
    ("诚恳", "强"): "用诚恳的语气说",
    ("轻松", "轻"): "用轻松的语气说",
    ("轻松", "中"): "用轻松的语气说",
    ("轻松", "强"): "用轻松愉快的语气说",
    ("紧张", "轻"): "用紧张的语气说，语速稍快",
    ("紧张", "中"): "用紧张的语气说，语速稍快",
    ("紧张", "强"): "用紧张的语气说，语速加快",
    ("骄傲", "轻"): "用骄傲的语气说",
    ("骄傲", "中"): "用骄傲的语气说",
    ("骄傲", "强"): "用骄傲自豪的语气说",
    ("轻松", "轻"): "用轻松的语气说",
}

_ELDERLY_SPEAKERS = {"老乌龟", "老爷爷", "老牛", "老奶奶", "老爷爷", "爷爷", "奶奶"}


def _build_instruct_text(emotion: str, intensity: str, pace: str, speaker: str) -> str:
    """将 director plan 映射为 CosyVoice instruct 指令。"""
    parts = []

    # 1. 情绪基础语气
    base = _EMOTION_MAP.get((emotion, intensity), f"用{emotion}的语气说")
    parts.append(base)

    # 2. 语速补充（避免与语气描述中的语速冲突）
    if "语速" not in base:
        if pace == "慢":
            parts.append("语速偏慢")
        elif pace == "稍慢":
            parts.append("语速稍慢")
        elif pace == "稍快":
            parts.append("语速稍快")
        elif pace == "快":
            parts.append("语速快")

    # 3. 角色特殊处理
    if speaker == "narrator":
        parts.insert(0, "以旁白叙述的语气")
    elif speaker in _ELDERLY_SPEAKERS:
        parts.append("语速偏慢，声音沉稳")

    return "，".join(parts) + "。"


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------


def generate_instructions(
    director_plan: Dict,
    characters: Dict,
    voice_bank: Dict[str, str],
    story_name: str = "",
) -> Dict:
    """为每个 segment 生成 CosyVoice TTS 指令。

    Args:
        director_plan: story_director.direct_story() 的输出
        characters: character_analyzer.analyze_characters() 的输出
        voice_bank: voice_bank_generator.generate_voice_bank() 的输出，speaker -> wav path
        story_name: 故事名称，用于文件命名前缀

    Returns:
        包含 instructions 列表的字典
    """
    instructions = []
    prefix = f"{story_name}_" if story_name else ""

    # narrator 的参考音频路径（voice_bank 中 key 为 "narrator"）
    narrator_audio = voice_bank.get("narrator", "")

    for seg in director_plan.get("segment_directions", []):
        speaker = seg.get("speaker", "narrator")

        # 取该角色的音色参考音频
        prompt_audio = voice_bank.get(speaker, narrator_audio)

        # 从 director plan 读取表演指导
        emotion = seg.get("emotion", "平静")
        intensity = seg.get("intensity", "轻")
        pace = seg.get("pace", "正常")
        pause_after_ms = seg.get("pause_after_ms", 400)

        # 拼接 instruct 指令（instruct 模式：固定前缀 + 动态情绪 + 分隔符，分隔符后为空）
        dynamic = _build_instruct_text(emotion, intensity, pace, speaker)
        prompt_text = (
            "You are a helpful assistant. "
            + dynamic
            + "<|endofprompt|>"
        )

        instructions.append({
            "segment_id": seg["segment_id"],
            "original_text": seg["text"],
            "text": seg["text"],
            "speaker": speaker,
            "type": seg.get("type", "narration"),
            "emotion": emotion,
            "intensity": intensity,
            "pace": pace,
            "prompt_text": prompt_text,           # instruct 指令，如"用平静的语气说。<|endofprompt|>"
            "prompt_audio": Path(prompt_audio).name,  # 相对路径如"narrator.wav"
            "pause_after_ms": pause_after_ms,
            "output_filename": f"{prefix}seg_{seg['segment_id']:03d}_{speaker}.wav",
        })

    return {
        "instructions": instructions,
        "total_instructions": len(instructions),
    }