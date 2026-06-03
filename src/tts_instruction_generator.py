"""TTS 指令生成器：将 segments + 角色标签 → MOSS-TTS-Nano 可执行的 TTS 指令。

查 voice_preset_config.json 映射表，为每个 segment 匹配参考音频和采样参数。
"""

import json
from pathlib import Path
from typing import Dict, List

_AUDIO_BASE = "M:/Users/l30083418/Documents/MOSS-TTS-Nano/assets/audio"


def generate_instructions(
    segments: Dict,
    characters: Dict,
    story_name: str = "",
    config_path: str = "config/voice_preset_config.json",
) -> Dict:
    """为每个 segment 生成 TTS 指令。

    Args:
        segments: segment_builder.build_segments() 的输出
        characters: character_analyzer.analyze_characters() 的输出
        story_name: 故事名称，用于文件命名前缀（如桂花雨）
        config_path: voice_preset_config.json 的路径

    Returns:
        包含 instructions 列表的字典
    """
    config = _load_config(config_path)
    speaker_profiles = _build_speaker_map(characters)
    instructions = []
    prefix = f"{story_name}_" if story_name else ""

    for seg in segments["segments"]:
        speaker = seg["speaker"]
        profile = speaker_profiles.get(speaker)
        audio_path = _resolve_audio(speaker, profile, characters.get("narrator"), config)
        params = _default_tts_params()

        instructions.append({
            "segment_id": seg["segment_id"],
            "text": seg["text"],
            "speaker": speaker,
            "type": seg["type"],
            "voice_preset": _make_preset_key(profile) if profile else "narrator",
            "prompt_audio_path": audio_path,
            "tts_params": params,
            "output_filename": f"{prefix}seg_{seg['segment_id']:03d}_{speaker}.wav",
        })

    return {
        "instructions": instructions,
        "total_instructions": len(instructions),
    }


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
    return f"{profile['gender']}_{profile['age']}_{profile['timbre']}"


def _resolve_audio(
    speaker: str, profile: Dict, narrator_profile: Dict, config: Dict
) -> str:
    """根据角色标签查映射表，返回参考音频路径。

    匹配优先级：精确匹配 → 去掉 timbre → 去掉 age → fallback narrator
    """
    if speaker == "narrator" or profile is None:
        return config.get("narrator", f"{_AUDIO_BASE}/zh_4.wav")

    presets = config.get("presets", {})

    # 精确匹配
    key = _make_preset_key(profile)
    if key in presets:
        return presets[key]

    # 去掉 timbre
    key_gender_age = f"{profile['gender']}_{profile['age']}"
    for k, v in presets.items():
        if k.startswith(key_gender_age):
            return v

    # 只匹配 gender
    for k, v in presets.items():
        if k.startswith(profile["gender"]):
            return v

    # fallback narrator
    return config.get("narrator", f"{_AUDIO_BASE}/zh_4.wav")


def _default_tts_params() -> Dict:
    """返回 MOSS-TTS-Nano ONNX Runtime 默认参数。"""
    return {
        "mode": "voice_clone",
        "do_sample": True,
    }
