"""Qwen3-VoiceDesign 生成角色音色参考音频。

为每个角色调用 Qwen3，根据 voice_instruction 生成音色参考 WAV。
"""

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import requests
import soundfile as sf

_API_URL = "http://10.154.39.97:8007"
os.environ["no_proxy"] = "*"

_FIXED_TEXT = "这是一个有声书音频agent项目，我会用不同的音色和语气来朗读各种类型的故事文本，包括童话、寓言、散文和小说等不同题材的内容"


def generate_voice_bank(
    characters: Dict,
    output_dir: str,
    force: bool = False,
) -> Dict[str, str]:
    """为每个角色生成音色参考音频。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    speakers = [("narrator", characters.get("narrator", {}))]
    for c in characters.get("characters", []):
        speakers.append((c["speaker"], c))

    print(f"[2/?] 生成音色参考（Qwen3-VoiceDesign）...")
    for i, (speaker, profile) in enumerate(speakers):
        voice_instruction = profile.get("voice_instruction", "")
        if not voice_instruction:
            print(f"  {speaker}: 跳过（无 voice_instruction）")
            continue

        audio_path = str(Path(output_dir) / f"{speaker}.wav")
        if not force and Path(audio_path).exists():
            print(f"  {speaker}: 已存在，跳过")
            results[speaker] = audio_path
            continue

        print(f"  {speaker}: {voice_instruction[:30]}...", end=" ", flush=True)
        t0 = time.perf_counter()
        ok = _generate_one(speaker, voice_instruction, audio_path)
        elapsed = time.perf_counter() - t0
        if ok:
            print(f"OK ({elapsed:.1f}s)")
            results[speaker] = audio_path
        else:
            print(f"FAIL")

    success = len(results)
    total = len(speakers)
    print(f"  完成，{success}/{total} 个音色")
    return results


def _post_process_wav(audio_path: str) -> str:
    """裁剪首尾静音，重新编码为标准 PCM WAV。"""
    data, sr = sf.read(audio_path)

    window = int(sr * 0.05)
    rms = np.array([
        np.sqrt(np.mean(data[i:i+window]**2))
        for i in range(0, len(data), window)
    ])
    threshold = 0.005

    start_win = next((i for i, e in enumerate(rms) if e > threshold), 0)
    end_win = next(
        (len(rms) - 1 - i for i, e in enumerate(reversed(rms)) if e > threshold),
        len(rms) - 1,
    )

    keep_start = max(0, start_win - int(0.2 / 0.05))
    keep_end = min(len(rms) - 1, end_win + int(0.2 / 0.05))

    frame_start = keep_start * window
    frame_end = min((keep_end + 1) * window, len(data))
    data = data[frame_start:frame_end]

    sf.write(audio_path, data, sr)
    return audio_path


def _generate_one(speaker: str, voice_instruction: str, output_path: str) -> bool:
    """调用 Qwen3 生成单个音色。"""
    payload = {
        "text": _FIXED_TEXT,
        "instruction": voice_instruction,
        "language": "Chinese",
    }
    try:
        resp = requests.post(
            f"{_API_URL}/v1/voicedesign/generate",
            json=payload,
            proxies={"http": None, "https": None},
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"  API error {resp.status_code}: {resp.text[:100]}")
            return False
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        _post_process_wav(output_path)
        return True
    except Exception as e:
        print(f"  Exception: {e}")
        return False