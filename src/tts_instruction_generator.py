"""TTS 指令生成器：将 director_plan + characters → MOSS-TTS-v1.5 服务器可执行的 TTS 指令。

根据 segment 的 emotion 映射 temperature，根据 speaker 匹配参考音频。
冲突时同性别下换一个音频。
"""

import json
import re
from pathlib import Path
from typing import Dict


# ----------------------------------------------------------------------
# 映射规则
# ----------------------------------------------------------------------

_EMOTION_INTENSITY_TO_TEMP: Dict[tuple, float] = {
    ("平静", "轻"): 1.5,
    ("平静", "中"): 1.6,
    ("平静", "强"): 1.7,
    ("温馨", "轻"): 1.5,
    ("温馨", "中"): 1.6,
    ("温馨", "强"): 1.7,
    ("担忧", "轻"): 1.6,
    ("担忧", "中"): 1.8,
    ("担忧", "强"): 1.9,
    ("坚定", "轻"): 1.6,
    ("坚定", "中"): 1.7,
    ("坚定", "强"): 1.9,
    ("恐惧", "轻"): 1.7,
    ("恐惧", "中"): 1.8,
    ("恐惧", "强"): 2.0,
    ("兴奋", "轻"): 1.7,
    ("兴奋", "中"): 1.8,
    ("兴奋", "强"): 2.0,
    ("急切", "轻"): 1.7,
    ("急切", "中"): 1.9,
    ("急切", "强"): 2.0,
    ("惊讶", "轻"): 1.6,
    ("惊讶", "中"): 1.8,
    ("惊讶", "强"): 2.0,
    ("愤怒", "轻"): 1.7,
    ("愤怒", "中"): 1.9,
    ("愤怒", "强"): 2.0,
    ("伤感", "轻"): 1.6,
    ("伤感", "中"): 1.7,
    ("伤感", "强"): 1.8,
}


def _map_emotion_to_temperature(emotion: str, intensity: str) -> float:
    return _EMOTION_INTENSITY_TO_TEMP.get((emotion, intensity), 1.7)


# ----------------------------------------------------------------------
# 停顿标记 & 儿童故事性别调整
# ----------------------------------------------------------------------


def _add_pause_markers(text: str) -> str:
    """在中文标点前插入 [pause Xs] 标记，让 TTS 在标点处产生自然停顿。

    逗号：0.3s，句号：0.5s，感叹/问号：0.4s
    """
    # 句号前：500ms
    text = re.sub(r'。', r'[pause 1s]。', text)
    # 逗号前：300ms
    text = re.sub(r'，', r'[pause 0.6s]，', text)
    # 感叹号、问号前：400ms
    text = re.sub(r'！', r'[pause 0.8s]！', text)
    text = re.sub(r'？', r'[pause 0.8s]？', text)
    return text


_CHILDREN_GENRES = {"儿童故事", "寓言", "童话", "绘本故事", "小学课文"}


def _adjust_gender_for_children(
    speaker_profiles: Dict[str, Dict],
    director_plan: Dict,
    characters: Dict,
) -> Dict[str, Dict]:
    """如果是儿童/寓言类故事，且 confidence 不是 high，强制用女声。"""
    genre = director_plan.get("overall_style", {}).get("genre", "")
    if genre not in _CHILDREN_GENRES:
        return speaker_profiles

    for c in characters.get("characters", []):
        speaker = c.get("speaker") or c.get("name")
        if speaker not in speaker_profiles:
            continue
        confidence = c.get("confidence", "high")
        if confidence != "high":
            speaker_profiles[speaker]["gender"] = "female"
            speaker_profiles[speaker]["gender_adjust_reason"] = "children_story_low_confidence_default_female"

    return speaker_profiles


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------


def generate_instructions(
    director_plan: Dict,
    characters: Dict,
    story_name: str = "",
    config_path: str = "config/voice_preset_config.json",
) -> Dict:
    """为每个 segment 生成 TTS 指令。

    Args:
        director_plan: story_director.direct_story() 的输出，即 director_plan.json
        characters: character_analyzer.analyze_characters() 的输出，即 characters.json
        story_name: 故事名称，用于文件命名前缀
        config_path: voice_preset_config.json 的路径

    Returns:
        包含 instructions 列表的字典
    """
    config = _load_config(config_path)
    speaker_profiles = _build_speaker_map(characters)

    # 儿童/寓言类故事，低置信度角色默认女声
    speaker_profiles = _adjust_gender_for_children(speaker_profiles, director_plan, characters)

    # 第一步：为每个角色分配音频（含冲突处理）
    speaker_audio_map = _assign_audio_with_dedup(speaker_profiles, config, characters)

    instructions = []
    prefix = f"{story_name}_" if story_name else ""

    for seg in director_plan.get("segment_directions", []):
        speaker = seg.get("speaker", "narrator")

        # 从预分配结果取音频
        audio_path = speaker_audio_map.get(speaker, config.get("narrator", ""))
        profile = speaker_profiles.get(speaker) or characters.get("narrator")

        # 从 director plan 读取参数
        emotion = seg.get("emotion", "平静")
        intensity = seg.get("intensity", "轻")
        pause_after_ms = seg.get("pause_after_ms", 400)

        tts_params = {
            "temperature": _map_emotion_to_temperature(emotion, intensity),
            "top_p": 0.5,
            "pause_after_ms": pause_after_ms,
        }

        original_text = seg["text"]
        tts_text = _add_pause_markers(original_text)

        instructions.append({
            "segment_id": seg["segment_id"],
            "original_text": original_text,
            "text": tts_text,
            "speaker": speaker,
            "type": seg.get("type", "narration"),
            "voice_preset": _make_preset_key(profile) if profile else "narrator",
            "reference_audio": audio_path,
            "tts_params": tts_params,
            "pause_marker_enabled": True,
            "voice_selection_reason": profile.get("gender_adjust_reason", "") if profile else "",
            "output_filename": f"{prefix}seg_{seg['segment_id']:03d}_{speaker}.wav",
        })

    return {
        "instructions": instructions,
        "total_instructions": len(instructions),
    }


# ----------------------------------------------------------------------
# 内部工具函数
# ----------------------------------------------------------------------


def _load_config(config_path: str) -> Dict:
    """读取 voice_preset_config.json。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_speaker_map(characters: Dict) -> Dict[str, Dict]:
    """从 characters 输出建立 speaker → profile 映射。"""
    result = {}
    for c in characters.get("characters", []):
        result[c["speaker"]] = {
            "gender": c["gender"],
            "age": c["age"],
            "timbre": c["timbre"],
        }
    return result


def _make_preset_key(profile: Dict) -> str:
    """将角色标签组合为 preset key，如 female_middle_aged_warm。"""
    if not profile:
        return "narrator"
    return f"{profile['gender']}_{profile['age']}_{profile['timbre']}"


def _assign_audio_with_dedup(
    speaker_profiles: Dict[str, Dict],
    config: Dict,
    characters: Dict,
) -> Dict[str, str]:
    """为每个角色分配参考音频，冲突时同性别下换一个。"""
    speaker_audio = {}
    used_by_gender = {}

    narrator_audio = config.get("narrator", "")
    speaker_audio["narrator"] = narrator_audio
    narrator_gender = characters.get("narrator", {}).get("gender", "female")
    used_by_gender.setdefault(narrator_gender, set()).add(narrator_audio)

    presets = config.get("presets", {})

    for speaker, profile in speaker_profiles.items():
        audio_path = _find_best_audio(profile, presets, used_by_gender, narrator_audio)
        speaker_audio[speaker] = audio_path
        gender = profile.get("gender", "")
        used_by_gender.setdefault(gender, set()).add(audio_path)

    return speaker_audio


def _find_best_audio(
    profile: Dict,
    presets: Dict[str, str],
    used_by_gender: Dict[str, set],
    narrator_audio: str,
) -> str:
    """为单个角色找最佳可用音频，跳过已被同性角色使用的。"""
    gender = profile.get("gender", "")

    def candidates():
        key = _make_preset_key(profile)
        if key in presets:
            yield key, presets[key]
        key_gender_age = f"{gender}_{profile['age']}"
        for k, v in presets.items():
            if k.startswith(key_gender_age):
                yield k, v
        for k, v in presets.items():
            if k.startswith(gender):
                yield k, v
        yield "narrator", narrator_audio

    for key, audio_path in candidates():
        if audio_path not in used_by_gender.get(gender, set()):
            return audio_path

    for key, audio_path in candidates():
        return audio_path