"""CosyVoice instruct 模式 TTS 引擎。

调用 /v1/cosyvoice/generate 合成音频片段，并拼接为完整 wav。
"""

import base64
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import soundfile as sf

_API_URL = "http://10.154.39.97:8005"


# ----------------------------------------------------------------------
# 核心：服务器调用
# ----------------------------------------------------------------------


def synthesize_remote(
    text: str,
    prompt_audio_path: str,
    prompt_text: str,
    output_path: str = None,
) -> str:
    """调用 CosyVoice /v1/cosyvoice/generate 生成音频。"""
    with open(prompt_audio_path, "rb") as f:
        prompt_audio = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "text": text,
        "prompt_text": prompt_text,       # e.g. "用平静的语气说。<|endofprompt|>"
        "prompt_audio": prompt_audio,
        "mode": "instruct",
        "stream": False,
    }

    response = requests.post(
        f"{_API_URL}/v1/cosyvoice/generate",
        json=payload,
        proxies={"http": None, "https": None},
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text[:200]}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        _trim_middle_silence(output_path)

    return output_path


# ----------------------------------------------------------------------
# 静音裁剪
# ----------------------------------------------------------------------


def _trim_middle_silence(
    audio_path: str,
    max_silence_ms: float = 300,
    window_ms: float = 50,
    rms_threshold: float = 0.005,
) -> str:
    """检测并裁剪音频内部过长的静音段。

    用短时窗口的 RMS 能量判断静音，超过 max_silence_ms 则裁剪到该阈值。
    """
    data, sr = sf.read(audio_path)
    window_frames = int(sr * window_ms / 1000)
    max_silence_windows = int(max_silence_ms / window_ms)

    rms = np.array([
        np.sqrt(np.mean(data[i:i+window_frames]**2))
        for i in range(0, len(data), window_frames)
    ])

    silence_windows = []
    in_silence = False
    start = 0
    for j, e in enumerate(rms):
        if e < rms_threshold and not in_silence:
            in_silence = True
            start = j
        elif e >= rms_threshold and in_silence:
            in_silence = False
            silence_windows.append((start, j))
    if in_silence:
        silence_windows.append((start, len(rms)))

    if not silence_windows:
        return audio_path

    for sw_start, sw_end in reversed(silence_windows):
        if sw_end - sw_start > max_silence_windows:
            frame_start = sw_start * window_frames
            frame_end = min(sw_end * window_frames, len(data))
            keep_frames = max_silence_windows * window_frames
            new_frame_end = frame_start + keep_frames
            data = np.concatenate([data[:frame_start], data[new_frame_end:]])

    sf.write(audio_path, data, sr)
    return audio_path


# ----------------------------------------------------------------------
# 批量合成
# ----------------------------------------------------------------------


def synthesize_all(
    instructions: Dict,
    voice_bank_dir: str,
    output_dir: str = "audio_segments",
) -> Dict:
    """遍历 instructions 逐条合成语音。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    success = 0
    failed = 0

    for i, inst in enumerate(instructions["instructions"]):
        text_preview = inst["text"][:30]
        print(f"[{i + 1}/{instructions['total_instructions']}] {inst['speaker']}: {text_preview}...", end=" ", flush=True)

        result = _synthesize_one(inst, voice_bank_dir, output_dir)
        results.append(result)

        if result["status"] == "success":
            success += 1
            print(f"OK ({result.get('elapsed_seconds', 0):.1f}s)")
        else:
            failed += 1
            print(f"FAIL: {result['error']}")

    return {
        "results": results,
        "total": len(results),
        "success": success,
        "failed": failed,
    }


def _synthesize_one(instruction: Dict, voice_bank_dir: str, output_dir: str) -> Dict:
    """合成单条指令。"""
    prompt_audio_path = str(Path(voice_bank_dir) / instruction["prompt_audio"])

    if not Path(prompt_audio_path).exists():
        return {
            "segment_id": instruction["segment_id"],
            "speaker": instruction["speaker"],
            "audio_path": "",
            "pause_after_ms": instruction.get("pause_after_ms", 400),
            "elapsed_seconds": 0,
            "status": "failed",
            "error": f"prompt_audio not found: {prompt_audio_path}",
        }

    output_path = str(Path(output_dir) / instruction["output_filename"])

    try:
        t0 = time.perf_counter()
        synthesize_remote(
            text=instruction["text"],
            prompt_audio_path=prompt_audio_path,
            prompt_text=instruction["prompt_text"],
            output_path=output_path,
        )
        elapsed = time.perf_counter() - t0

        return {
            "segment_id": instruction["segment_id"],
            "speaker": instruction["speaker"],
            "audio_path": output_path,
            "pause_after_ms": instruction.get("pause_after_ms", 400),
            "elapsed_seconds": round(elapsed, 2),
            "status": "success",
        }
    except Exception as e:
        return {
            "segment_id": instruction["segment_id"],
            "speaker": instruction["speaker"],
            "audio_path": "",
            "pause_after_ms": instruction.get("pause_after_ms", 400),
            "elapsed_seconds": 0,
            "status": "failed",
            "error": str(e),
        }


# ----------------------------------------------------------------------
# 音频拼接
# ----------------------------------------------------------------------


def stitch_audio(
    results: List[Dict],
    output_path: str = "audio_final/final.wav",
    default_pause_ms: float = 400,
) -> Dict:
    """按 segment_id 顺序拼接所有成功生成的 wav 文件。"""
    successful = [r for r in results if r["status"] == "success"]
    successful.sort(key=lambda r: r["segment_id"])

    if not successful:
        return {"output_path": "", "total_duration_seconds": 0, "total_segments": 0}

    first = successful[0]["audio_path"]
    audio_data, sr = sf.read(first)
    channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_frames = len(audio_data)
    pause_samples = 0

    with sf.SoundFile(output_path, "w", samplerate=sr, channels=channels) as out_sf:
        out_sf.write(audio_data)

        for i in range(len(successful)):
            if i == 0:
                continue
            prev_r = successful[i - 1]
            pause_ms = prev_r.get("pause_after_ms", default_pause_ms)
            pause_frames = int(sr * pause_ms / 1000)
            if pause_frames > 0:
                if channels > 1:
                    pause_arr = np.zeros((pause_frames, channels))
                else:
                    pause_arr = np.zeros(pause_frames)
                out_sf.write(pause_arr)
                pause_samples += pause_frames

            seg_data, _ = sf.read(successful[i]["audio_path"])
            out_sf.write(seg_data)
            total_frames += len(seg_data)

    duration = (total_frames + pause_samples) / sr

    return {
        "output_path": str(Path(output_path).resolve()),
        "total_duration_seconds": round(duration, 2),
        "total_segments": len(successful),
    }


def synthesize_and_stitch(
    instructions: Dict,
    voice_bank_dir: str,
    output_dir: str = "audio_segments",
    final_output: str = "audio_final/final.wav",
) -> Dict:
    """合成所有片段并拼接为完整音频。"""
    synth_result = synthesize_all(instructions, voice_bank_dir, output_dir)
    stitch_result = stitch_audio(synth_result["results"], final_output)

    return {
        "synthesis": synth_result,
        "stitch": stitch_result,
    }