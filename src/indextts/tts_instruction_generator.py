"""IndexTTS 指令生成器：将导演计划打包为 TTS API 指令。"""

from pathlib import Path
from typing import Dict


def generate_instructions(
    director_plan: Dict,
    characters: Dict,
    voice_bank: Dict[str, str],
    story_name: str = "",
) -> Dict:
    """为每个 segment 生成 IndexTTS 合成指令。

    Args:
        director_plan: indextts/story_director.direct_story() 的输出
        characters: character_analyzer.analyze_characters() 的输出
        voice_bank: voice_bank_generator.generate_voice_bank() 的输出，speaker -> wav path
        story_name: 故事名称，用于文件命名前缀

    Returns:
        包含 instructions 列表的字典
    """
    instructions = []
    prefix = f"{story_name}_" if story_name else ""

    narrator_audio = voice_bank.get("narrator", "")

    # narrator 固定 emotion_vector，旁白音色稳定性优先
    NARRATOR_VECTOR = [0, 0, 0, 0, 0, 0, 0, 1.0]

    for seg in director_plan.get("segment_directions", []):
        speaker = seg.get("speaker", "narrator")
        prompt_audio = voice_bank.get(speaker, narrator_audio)

        # narrator 使用固定向量 + emotion_text 控制情绪
        if speaker == "narrator":
            emotion_vector = NARRATOR_VECTOR
        else:
            emotion_vector = seg.get("emotion_vector", NARRATOR_VECTOR)

        instructions.append({
            "segment_id": seg["segment_id"],
            "original_text": seg["text"],
            "text": seg["text"],
            "speaker": speaker,
            "type": seg.get("type", "narration"),
            "emotion_vector": emotion_vector,
            "emotion_text": seg.get("emotion_text", ""),
            "interval_silence": seg.get("interval_silence", 400),
            "prompt_audio": Path(prompt_audio).name,
            "output_filename": f"{prefix}seg_{seg['segment_id']:03d}_{speaker}.wav",
        })

    return {
        "instructions": instructions,
        "total_instructions": len(instructions),
    }